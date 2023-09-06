import time
import argparse
from typing import Tuple, Union, List

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
import torch
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from scipy.ndimage import gaussian_filter

from nnunet.utilities.nd_softmax import numpy_softmax, softmax_helper


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt.__version__ >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TensorRTwithTorchPredictor(torch.nn.Module):
    def __init__(self, engine_path, num_classes, input_names=None, output_names=None):
        super(TensorRTwithTorchPredictor, self).__init__()
        self.context = None
        self.engine = None
        self.logger = None
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        self.engine_path = engine_path
        self.build_engine()
        self.check_engine()

        self.num_classes = num_classes
        self.input_names = input_names
        self.output_names = output_names

        self._gaussian_3d = self._patch_size_for_gaussian_3d = None

        self.inference_apply_nonlin = softmax_helper

    def build_engine(self):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        print("build engine from ", self.engine_path)

    def check_engine(self):
        for idx in range(self.engine.num_bindings):  # 查看输入输出的名字，类型，大小
            is_input = self.engine.binding_is_input(idx)
            name = self.engine.get_binding_name(idx)
            op_type = self.engine.get_binding_dtype(idx)
            shape = self.engine.get_binding_shape(idx)
            print('input id:', idx, ' is input: ', is_input, ' binding name:', name, ' shape:', shape, 'type: ',
                  op_type)

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))  # set shape
            # contiguous() Returns a contiguous in memory tensor containing the same data as self tensor.
            # data_ptr() Returns the address of the first element of self tensor.
            bindings[idx] = inputs[i].contiguous().data_ptr()

        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.engine.get_binding_shape(idx))
            # get_location() a pointer to device or host memory.
            device = torch_device_from_trt(self.engine.get_location(idx))
            # torch.empty() Returns a tensor filled with uninitialized data.
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()
        # torch.cuda.current_stream() Returns the currently selected Stream for a given device.
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

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

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            # predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)
        else:
            gaussian_importance_map = None

        if np.prod(data.shape[1:]) > 8e6:
            all_in_gpu = False
        else:
            all_in_gpu = True

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(list(patch_size), device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device()) + 1e-7

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]),
                                                    dtype=np.float32) + 1e-7

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

            if all_in_gpu:
                predicted_patch = predicted_patch.half()
            else:
                predicted_patch = predicted_patch.cpu().numpy()

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
                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()
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

        if all_in_gpu:
            if verbose: print("copying results to CPU")
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            aggregated_results = aggregated_results.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, aggregated_results

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        x = maybe_to_torch(x)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        batchTTA = False
        TTA4 = True
        print(">>>TensorRTwithTorchPredictor TTA4, batchTTA, mirror_idx: ", TTA4, batchTTA,
              mirror_idx)
        if batchTTA:
            # x -> torch.Size([1, 1, 32, 128, 192]), torch.float32, [2, 3, 4] are index of z, y, x-axis
            # partial tta for batch inference
            tta_batch_data = torch.concat((x, torch.flip(x, (4,)), torch.flip(x, (3,)), torch.flip(x, (2,)),
                                           # torch.flip(x, (4, 3)), torch.flip(x, (4, 2)), torch.flip(x, (3, 2)),
                                           # torch.flip(x, (4, 3, 2))
                                           ), dim=0)
            tta_batch_pred = self.inference_apply_nonlin(self(tta_batch_data))
            tta_batch_pred[1:2] = torch.flip(tta_batch_pred[1:2], (4,))
            tta_batch_pred[2:3] = torch.flip(tta_batch_pred[2:3], (3,))
            tta_batch_pred[3:4] = torch.flip(tta_batch_pred[3:4], (2,))
            # tta_batch_pred[4:5] = torch.flip(tta_batch_pred[4:5], (4, 3))
            # tta_batch_pred[5:6] = torch.flip(tta_batch_pred[5:6], (4, 2))
            # tta_batch_pred[6:7] = torch.flip(tta_batch_pred[6:7], (3, 2))
            # tta_batch_pred[7:8] = torch.flip(tta_batch_pred[7:8], (4, 3, 2))
            result_torch = torch.mean(tta_batch_pred, dim=0, keepdim=True)
        elif TTA4:
            pred = self.inference_apply_nonlin(self(x))
            result_torch += 1 / num_results * pred
            pred = self.inference_apply_nonlin(self(torch.flip(x, (4,))))
            result_torch += 1 / num_results * torch.flip(pred, (4,))
            pred = self.inference_apply_nonlin(self(torch.flip(x, (3,))))
            result_torch += 1 / num_results * torch.flip(pred, (3,))
            pred = self.inference_apply_nonlin(self(torch.flip(x, (2,))))
            result_torch += 1 / num_results * torch.flip(pred, (2,))
        else:
            for m in range(mirror_idx):
                if m == 0:
                    pred = self.inference_apply_nonlin(self(x))
                    result_torch += 1 / num_results * pred

                if m == 1 and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4,))))
                    result_torch += 1 / num_results * torch.flip(pred, (4,))

                if m == 2 and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3,))))
                    result_torch += 1 / num_results * torch.flip(pred, (3,))

                if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3))

                if m == 4 and (0 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (2,))))
                    result_torch += 1 / num_results * torch.flip(pred, (2,))

                if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 2))

                if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                    result_torch += 1 / num_results * torch.flip(pred, (3, 2))

                if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                    pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                    result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...],
                                          step_size: float) -> \
            List[List[int]]:
        assert [i >= j for i, j in
                zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
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


class TensorRTPredictor:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """

    def __init__(self, engine_path, num_classes):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        """ engine params """
        self.engine_path = engine_path
        self.logger = None
        self.engine = None
        self.context = None
        self.allocations = None
        self.outputs = None
        self.inputs = None
        self.batch_size = None

        """ load engine """
        self.load_engine()

        """ prediction params """
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
        TTA4 = False
        print(">>>TensorRTPredictor TTA4, batchTTA, mirror_idx: ", TTA4, batchTTA, mirror_idx)
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

    def _predict_patch(self, batch):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """
        # Copy I/O and Execute
        cuda.memcpy_htod(self.inputs[0]['allocation'], batch.ravel())
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        preds = [o['host_allocation'] for o in self.outputs]
        pred = preds[0]  # [B, Co, D, H, W]
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

    def load_engine(self):
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context
        print("load engine from ", self.engine_path)

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs
