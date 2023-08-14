import copy
import time

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch

from utilities.logger import add_file_handler_to_logger, logger

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        print("binding ", engine.get_binding_shape(binding), engine.get_binding_dtype(binding))
        size = trt.volume(engine.get_binding_shape(binding))    # * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream


def build_engine(engine_path):
    trt.init_libnvinfer_plugins(None, "")
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    return engine


def load_engine(engine_path):
    logger.info("emptying cuda cache")
    torch.cuda.empty_cache()

    logger.info("load engine from " + engine_path)
    engine = build_engine(engine_path)
    logger.info("build_engine done")
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    logger.info("allocate_buffers done")
    context = engine.create_execution_context()
    logger.info("create_execution_context done")

    sess = {
        "context": context,
        "inputs": inputs,
        "outputs": outputs,
        "bindings": bindings,
        "stream": stream
    }
    return sess


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    return [out.host for out in outputs]


def predict(sess, dummy_input=None, save_npy=False):
    if dummy_input is None:
        # dummy_input = np.ones((B, Ci, D, H, W), dtype=np.float32)
        dummy_input = np.random.randn(B, Ci, D, H, W).astype(np.float32)
    # sess["inputs"][0].host = dummy_input
    np.copyto(sess["inputs"][0].host, dummy_input.ravel())

    trt_outputs = do_inference(sess["context"], bindings=sess["bindings"], inputs=sess["inputs"],
                               outputs=sess["outputs"], stream=sess["stream"])  # [o]
    print("network output shape:{}".format(trt_outputs[0].shape))
    trt_output = copy.deepcopy(trt_outputs[0])     # shape: (B*Co*D*H*W)
    Co = trt_output.shape[0] // (B * D * H * W)
    pred = trt_output.reshape(B, Co, D, H, W)

    if save_npy:
        np.save(base_dir + "trt_output.npy", pred)
        print(pred.shape, pred[0, :, 0, 0, 0])
    return pred


def diff_output():
    torch_path = base_dir + "torch_output.npy"
    onnx_path = base_dir + "onnx_output.npy"
    trt_path = base_dir + "trt_output.npy"

    onnx_re = np.load(onnx_path)
    torch_re = np.load(torch_path)
    trt_re = np.load(trt_path)

    diff = onnx_re - torch_re
    diff_abs = np.abs(diff)

    print("onnx-torch diff: ", np.min(diff), np.max(diff), np.mean(diff))
    print("onnx-torch diff_abs: ", np.min(diff_abs), np.max(diff_abs), np.mean(diff_abs))

    diff = trt_re - onnx_re
    diff_abs = np.abs(diff)

    print("trt-onnx diff: ", np.min(diff), np.max(diff), np.mean(diff))
    print("trt-onnx diff_abs: ", np.min(diff_abs), np.max(diff_abs), np.mean(diff_abs))



if __name__ == "__main__":
    print('CUDA device query (PyCUDA version) \n')

    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/predict_temp/tensorrt_test/torch_model/" \
               "Task030_FLARE23OARTumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium/"
    engine_pth = base_dir + "nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium_fp16.trt"

    add_file_handler_to_logger(name="main", dir_path=base_dir + "/logs/", level="DEBUG")
    loop_test = False
    B, Ci, D, H, W = 1, 1, 32, 128, 192

    t1 = time.time()
    session = load_engine(engine_pth)
    t2 = time.time()
    print(">>>load engine time: {}s".format(t2 - t1))
    if loop_test:
        for i in range(200):
            predict(session)
    else:
        d_input = np.ones((B, Ci, D, H, W), dtype=np.float32)
        predict(session, d_input, save_npy=True)
    t3 = time.time()
    print(">>>predict time: {}s".format(t3 - t2))

    # diff_output()




