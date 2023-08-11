import copy
import time

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch

from typing import Union, Tuple, List

from batchgenerators.augmentations.utils import pad_nd_image
from scipy.ndimage import gaussian_filter

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def numpy_softmax(x, axis=-1):
    """
    softmax numpy version
    :param x: input shape [B,D,H,W,C]
    :return: [B,D,H,W,C]
    """
    x = x - np.expand_dims(x.max(axis=axis), axis=axis)
    x_exp = np.exp(x)
    x_exp_row_sum = np.expand_dims(x_exp.sum(axis=axis), axis=axis)
    soft_max = x_exp / x_exp_row_sum

    return soft_max


class TensorRTSession(object):

    def __init__(self, engine_path, num_classes):
        self.engine = load_engine(engine_path)
        self.num_classes = num_classes

        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self.batch_data_size = None

        self.inference_apply_nonlin = lambda x: numpy_softmax(x, 1)  # softmax_helper

    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None,
                   regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True, window_type='fast') -> Tuple[
        np.ndarray, np.ndarray]:
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            # 3D
            if max(mirror_axes) > 2:
                raise ValueError("mirror axes. duh")

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        res = self._internal_predict_3D_3Dconv_tiled_full_window(x, step_size, do_mirroring,
                                                                 mirror_axes,
                                                                 patch_size,
                                                                 regions_class_order, use_gaussian,
                                                                 pad_border_mode,
                                                                 pad_kwargs=pad_kwargs,
                                                                 all_in_gpu=all_in_gpu,
                                                                 verbose=verbose)
        return res

    def _internal_predict_3D_3Dconv_tiled_full_window(
            self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
            patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
            pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
            verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose: print("done")
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d
        else:
            gaussian_importance_map = None

        if use_gaussian and num_tiles > 1:
            add_for_nb_of_preds = self._gaussian_3d
        else:
            add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
        aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32) + 1e-7

        # is_empty = False
        num_window = 0
        for z in steps[0]:

            lb_z = z
            ub_z = z + patch_size[0]

            if len(steps[1]) == 1:
                lb_y = 0
            else:
                lb_y = steps[1][1]
            ub_y = lb_y + patch_size[1]
            if len(steps[2]) == 1:
                lb_x = 0
            else:
                lb_x = steps[2][1]
            ub_x = lb_x + patch_size[2]
            predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                data[None, :, lb_z:ub_z, lb_y:ub_y, lb_x:ub_x], mirror_axes, do_mirroring,
                gaussian_importance_map)[0]
            num_window += 1

            aggregated_results[:, lb_z:ub_z, lb_y:ub_y, lb_x:ub_x] += predicted_patch
            aggregated_nb_of_predictions[:, lb_z:ub_z, lb_y:ub_y, lb_x:ub_x] += add_for_nb_of_preds

            for y in steps[1]:
                lb_y = y
                ub_y = lb_y + patch_size[1]
                for x in steps[2]:
                    lb_x = x
                    ub_x = lb_x + patch_size[2]

                    if len(steps[1]) == 1 and len(steps[2]) == 1:
                        continue
                    idx_y = 0 if len(steps[1]) == 1 else 1
                    idx_x = 0 if len(steps[2]) == 1 else 1

                    if y == steps[1][idx_y] and x == steps[2][idx_x]:
                        continue

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_z:ub_z, lb_y:ub_y, lb_x:ub_x], mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]
                    num_window += 1
                    aggregated_results[:, lb_z:ub_z, lb_y:ub_y, lb_x:ub_x] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_z:ub_z, lb_y:ub_y, lb_x:ub_x] += add_for_nb_of_preds
        print('num tiles:  ', num_tiles)
        print('num_window: ', num_window)

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        predicted_segmentation = aggregated_results.argmax(0)

        if verbose: print("prediction done")
        return predicted_segmentation, aggregated_results

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'
        self.batch_data_size = x.shape

        result = np.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=np.float)

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        batchTTA = False
        TTA4 = True
        print(">>>Debug _internal_maybe_mirror_and_pred_3D TTA4, batchTTA, mirror_idx: ", TTA4, batchTTA, mirror_idx)
        if batchTTA:
            # x -> torch.Size([1, 1, 32, 128, 192]), torch.float32, [2, 3, 4] are index of z, y, x-axis
            # partial tta for batch inference
            tta_batch_data = np.concat((x, np.flip(x, (4,)), np.flip(x, (3,)), np.flip(x, (2,)),
                                        # torch.flip(x, (4, 3)), torch.flip(x, (4, 2)), torch.flip(x, (3, 2)),
                                        # torch.flip(x, (4, 3, 2))
                                        ), dim=0)
            tta_batch_pred = self.inference_apply_nonlin(self._predict_patch(tta_batch_data))
            tta_batch_pred[1:2] = np.flip(tta_batch_pred[1:2], (4,))
            tta_batch_pred[2:3] = np.flip(tta_batch_pred[2:3], (3,))
            tta_batch_pred[3:4] = np.flip(tta_batch_pred[3:4], (2,))
            # tta_batch_pred[4:5] = torch.flip(tta_batch_pred[4:5], (4, 3))
            # tta_batch_pred[5:6] = torch.flip(tta_batch_pred[5:6], (4, 2))
            # tta_batch_pred[6:7] = torch.flip(tta_batch_pred[6:7], (3, 2))
            # tta_batch_pred[7:8] = torch.flip(tta_batch_pred[7:8], (4, 3, 2))
            result = np.mean(tta_batch_pred, dim=0, keepdim=True)
        elif TTA4:
            pred = self.inference_apply_nonlin(self._predict_patch(x))
            result += 1 / num_results * pred
            pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (4,))))
            result += 1 / num_results * np.flip(pred, (4,))
            pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (3,))))
            result += 1 / num_results * np.flip(pred, (3,))
            pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (2,))))
            result += 1 / num_results * np.flip(pred, (2,))
        else:
            for m in range(mirror_idx):
                if m == 0:
                    pred = self.inference_apply_nonlin(self._predict_patch(x))
                    result += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (4,))))
                    result += 1 / num_results * np.flip(pred, (4,))

                if m == 2 and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (3,))))
                    result += 1 / num_results * np.flip(pred, (3,))

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (4, 3))))
                    result += 1 / num_results * np.flip(pred, (4, 3))

                if m == 4 and (0 in mirror_axes):
                    pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (2,))))
                    result += 1 / num_results * np.flip(pred, (2,))

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (4, 2))))
                    result += 1 / num_results * np.flip(pred, (4, 2))

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (3, 2))))
                    result += 1 / num_results * np.flip(pred, (3, 2))

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self._predict_patch(np.flip(x, (4, 3, 2))))
                    result += 1 / num_results * np.flip(pred, (4, 3, 2))

        if mult is not None:
            result[:, :] *= mult

        return result

    def _predict_patch(self, x):
        np.copyto(self.engine["inputs"][0].host, x.ravel())
        trt_output = do_inference(self.engine["context"], bindings=self.engine["bindings"],
                                  inputs=self.engine["inputs"], outputs=self.engine["outputs"],
                                  stream=self.engine["stream"])[0]
        B, Ci, D, H, W = self.batch_data_size
        Co = trt_output.shape[0] // (B * D * H * W)
        pred = trt_output.reshape(B, Co, D, H, W)
        return pred

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> \
            List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in
                     zip(image_size, target_step_sizes_in_voxels, patch_size)]

        # 3 × 3
        num_steps[1] = 3
        num_steps[2] = 3

        # Decide steps of z axis
        steps = []
        max_step_value = image_size[0] - patch_size[0]
        if num_steps[0] > 1:
            actual_step_size = max_step_value / (num_steps[0] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0
        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[0])]
        steps.append(steps_here)

        #  fix steps of y axis to be 3
        if image_size[1] > 2.0 * patch_size[1]:
            steps_here = [int(np.round(image_size[1] / 2) - 1.0 * patch_size[1]),
                          int(np.round(image_size[1] / 2) - 0.5 * patch_size[1]),
                          int(np.round(image_size[1] / 2) + 0. * patch_size[1])]
        elif image_size[1] > patch_size[1]:
            actual_step_size = (image_size[1] - patch_size[1]) / 2
            steps_here = [int(np.round(actual_step_size * i)) for i in range(3)]
        else:
            steps_here = [0]
        steps.append(steps_here)

        #  fix steps of x axis to be 3
        if image_size[2] > 2.0 * patch_size[2]:
            steps_here = [int(np.round(image_size[2] / 2) - 1.0 * patch_size[2]),
                          int(np.round(image_size[2] / 2) - 0.5 * patch_size[2]),
                          int(np.round(image_size[2] / 2) + 0. * patch_size[2])]
        elif image_size[2] > patch_size[2]:
            actual_step_size = (image_size[2] - patch_size[2]) / 2
            steps_here = [int(np.round(actual_step_size * i)) for i in range(3)]
        else:
            steps_here = [0]
        steps.append(steps_here)
        return steps

    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map


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
        # print("binding ", engine.get_binding_shape(binding), engine.get_binding_dtype(binding))
        size = trt.volume(engine.get_binding_shape(binding))  # * engine.max_batch_size
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
    engine = build_engine(engine_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    sess = {
        "context": context,
        "inputs": inputs,
        "outputs": outputs,
        "bindings": bindings,
        "stream": stream
    }
    return sess


def do_inference(context, bindings, inputs, outputs, stream):
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
    trt_output = copy.deepcopy(trt_outputs[0])  # shape: (B*Co*D*H*W)
    Co = trt_output.shape[0] // (B * D * H * W)
    pred = trt_output.reshape(B, Co, D, H, W)

    if save_npy:
        np.save(base_dir + "trt_output.npy", pred)
        print(pred.shape, pred[0, :, 0, 0, 0])
    return pred


if __name__ == "__main__":
    print('CUDA device query (PyCUDA version) \n')

    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/tensorrt_test/torch_model/" \
               "Task030_FLARE23OARTumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium/"
    engine_pth = base_dir + "nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium.trt"
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
