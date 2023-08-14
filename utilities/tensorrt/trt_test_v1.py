import copy
import os
import sys
import time
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

import utilities.engine as engine_utils

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):
        """Initializes TensorRT objects needed for model inference.
        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))
        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = engine_utils.load_engine(
                self.trt_runtime, trt_engine_path)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = \
            engine_utils.allocate_buffers(self.trt_engine)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

    def infer(self, dummy_input=None):
        B, Ci, D, H, W = 1, 1, 32, 128, 192
        if dummy_input is None:
            # dummy_input = np.ones((B, Ci, D, H, W), dtype=np.float32)
            dummy_input = np.random.randn(B, Ci, D, H, W).astype(np.float32)
        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, dummy_input.ravel())
        # Output shapes expected by the post-processor
        # output_shapes = [(1, 11616, 4), (11616, 21)]
        # When infering on single image, we measure inference time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        trt_outputs = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)
        print("network output shape:{}".format(trt_outputs[0].shape))
        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))
        trt_output = copy.deepcopy(trt_outputs[0])  # shape: (B*Co*D*H*W)
        Co = trt_output.shape[0] // (B * D * H * W)
        pred = trt_output.reshape(B, Co, D, H, W)
        # And return results
        return pred


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == "__main__":
    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/tensorrt_test/predict_temp/torch_model/" \
               "Task030_FLARE23OARTumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium/"
    engine_pth = base_dir + "nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium.trt"

    t1 = time.time()
    trt_inference = TRTInference(engine_pth)
    t2 = time.time()
    print(">>>init engine time: {}s".format(t2 - t1))

    for i in range(200):
        trt_inference.infer()
    t3 = time.time()
    print(">>>predict time: {}s".format(t3 - t2))

