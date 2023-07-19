#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import shutil
from typing import Tuple, Union, Callable

import numpy as np
import SimpleITK as sitk
from batchgenerators.augmentations.color_augmentations import augment_brightness_multiplicative, augment_contrast, \
    augment_gamma
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise, augment_gaussian_blur
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug

TRANSFORM_DEBUG = False


def augment_spatial_multi_data(data, seg, patch_size, patch_center_dist_from_border=30,
                               do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                               do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                               do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0,
                               order_data=3,
                               border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True,
                               p_el_per_sample=1,
                               p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                               p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1,
                               modalities=None, ori_shp=None):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)
    if ori_shp is None:
        ori_shp = data.shape

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)
    if modalities is None:
        modalities = []
        for i in range(data.shape[0]):
            temp = [f"{j}" for j in range(data.shape[1])]
            modalities.append(temp)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = data.shape[d + 2] / 2. - 0.5
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                channel_group = channel_id // ori_shp[2]
                # print(channel_group, channel_id)
                if modalities[sample_id][channel_group] == "seg":
                    data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_seg,
                                                                         border_mode_seg, cval=border_cval_seg,
                                                                         is_seg=True)
                else:
                    data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords,
                                                                         order_data,
                                                                         border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg,
                                                                        is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result


class SpatialTransformMultiData(SpatialTransform):

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis: float = 1,
                 p_independent_scale_per_axis: int = 1):
        super().__init__(patch_size, patch_center_dist_from_border,
                         do_elastic_deform, alpha, sigma,
                         do_rotation, angle_x, angle_y, angle_z,
                         do_scale, scale, border_mode_data, border_cval_data, order_data,
                         border_mode_seg, border_cval_seg, order_seg, random_crop, data_key,
                         label_key, p_el_per_sample, p_scale_per_sample, p_rot_per_sample,
                         independent_scale_for_each_axis, p_rot_per_axis, p_independent_scale_per_axis)

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        shp = data_dict['orig_shape_data']
        modalities = [list(i["all_modalities"].values()) for i in data_dict.get("properties")]
        if TRANSFORM_DEBUG:
            pids = [os.path.basename(i["seg_file"]).split(".nii.gz")[0] for i in data_dict.get("properties")]
            print(">>>SpatialTransformMultiData __cal__: ", pids, modalities, data.shape, seg.shape, shp)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial_multi_data(data, seg, patch_size=patch_size,
                                             patch_center_dist_from_border=self.patch_center_dist_from_border,
                                             do_elastic_deform=self.do_elastic_deform, alpha=self.alpha,
                                             sigma=self.sigma,
                                             do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                             angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                             border_mode_data=self.border_mode_data,
                                             border_cval_data=self.border_cval_data, order_data=self.order_data,
                                             border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                             order_seg=self.order_seg, random_crop=self.random_crop,
                                             p_el_per_sample=self.p_el_per_sample,
                                             p_scale_per_sample=self.p_scale_per_sample,
                                             p_rot_per_sample=self.p_rot_per_sample,
                                             independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                             p_rot_per_axis=self.p_rot_per_axis,
                                             p_independent_scale_per_axis=self.p_independent_scale_per_axis,
                                             modalities=modalities, ori_shp=shp)
        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]

        return data_dict


class GaussianNoiseTransformMultiData(GaussianNoiseTransform):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel: float = 1, per_channel: bool = False,
                 data_key="data"):
        super().__init__(noise_variance, p_per_sample, p_per_channel, per_channel, data_key)

    def __call__(self, **data_dict):
        modalities = [list(i["all_modalities"].values()) for i in data_dict.get("properties")]
        if TRANSFORM_DEBUG:
            data = data_dict.get(self.data_key)
            pids = [os.path.basename(i["seg_file"]).split(".nii.gz")[0] for i in data_dict.get("properties")]
            print(">>>GaussianNoiseTransformMultiData __cal__: ", pids, modalities, data.shape)
        for b in range(len(data_dict[self.data_key])):
            for c in range(len(data_dict[self.data_key][0])):
                if modalities[b][c] != "seg" and np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b][c:c + 1] = augment_gaussian_noise(
                        data_dict[self.data_key][b][c:c + 1],
                        self.noise_variance,
                        self.p_per_channel, self.per_channel)
        return data_dict


class GaussianBlurTransformMultiData(GaussianBlurTransform):
    def __init__(self, blur_sigma: Tuple[float, float] = (1, 5), different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False, p_isotropic: float = 0, p_per_channel: float = 1,
                 p_per_sample: float = 1, data_key: str = "data"):
        super().__init__(blur_sigma, different_sigma_per_channel, different_sigma_per_axis, p_isotropic, p_per_channel,
                         p_per_sample, data_key)

    def __call__(self, **data_dict):
        modalities = [list(i["all_modalities"].values()) for i in data_dict.get("properties")]
        if TRANSFORM_DEBUG:
            data = data_dict.get(self.data_key)
            pids = [os.path.basename(i["seg_file"]).split(".nii.gz")[0] for i in data_dict.get("properties")]
            print(">>>GaussianBlurTransformMultiData __cal__: ", pids, modalities, data.shape)
        for b in range(len(data_dict[self.data_key])):
            for c in range(len(data_dict[self.data_key][0])):
                if modalities[b][c] != "seg" and np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b][c:c + 1] = \
                        augment_gaussian_blur(data_dict[self.data_key][b][c:c + 1],
                                              self.blur_sigma,
                                              self.different_sigma_per_channel,
                                              self.p_per_channel,
                                              different_sigma_per_axis=self.different_sigma_per_axis,
                                              p_isotropic=self.p_isotropic)
        return data_dict


class BrightnessMultiplicativeTransformMultiData(BrightnessMultiplicativeTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1):
        super().__init__(multiplier_range, per_channel, data_key, p_per_sample)

    def __call__(self, **data_dict):
        modalities = [list(i["all_modalities"].values()) for i in data_dict.get("properties")]
        if TRANSFORM_DEBUG:
            data = data_dict.get(self.data_key)
            pids = [os.path.basename(i["seg_file"]).split(".nii.gz")[0] for i in data_dict.get("properties")]
            print(">>>BrightnessMultiplicativeTransformMultiData __cal__: ", pids, modalities, data.shape)
        for b in range(len(data_dict[self.data_key])):
            for c in range(len(data_dict[self.data_key][0])):
                if modalities[b][c] != "seg" and np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b][c:c + 1] = \
                        augment_brightness_multiplicative(data_dict[self.data_key][b][c:c + 1],
                                                          self.multiplier_range,
                                                          self.per_channel)
        return data_dict


class ContrastAugmentationTransformMultiData(ContrastAugmentationTransform):
    def __init__(self, contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                 preserve_range: bool = True, per_channel: bool = True, data_key: str = "data", p_per_sample: float = 1,
                 p_per_channel: float = 1):
        super().__init__(contrast_range, preserve_range, per_channel, data_key, p_per_sample, p_per_channel)

    def __call__(self, **data_dict):
        modalities = [list(i["all_modalities"].values()) for i in data_dict.get("properties")]
        if TRANSFORM_DEBUG:
            data = data_dict.get(self.data_key)
            pids = [os.path.basename(i["seg_file"]).split(".nii.gz")[0] for i in data_dict.get("properties")]
            print(">>>ContrastAugmentationTransformMultiData __cal__: ", pids, modalities, data.shape)
        for b in range(len(data_dict[self.data_key])):
            for c in range(len(data_dict[self.data_key][0])):
                if modalities[b][c] != "seg" and np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b][c:c + 1] = \
                        augment_contrast(data_dict[self.data_key][b][c:c + 1],
                                         contrast_range=self.contrast_range,
                                         preserve_range=self.preserve_range,
                                         per_channel=self.per_channel,
                                         p_per_channel=self.p_per_channel)
        return data_dict


class SimulateLowResolutionTransformMultiData(SimulateLowResolutionTransform):
    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1, channels=None, order_downsample=1,
                 order_upsample=0, data_key="data", p_per_sample=1, ignore_axes=None):
        super().__init__(zoom_range, per_channel, p_per_channel, channels, order_downsample, order_upsample, data_key,
                         p_per_sample, ignore_axes)

    def __call__(self, **data_dict):
        modalities = [list(i["all_modalities"].values()) for i in data_dict.get("properties")]
        if TRANSFORM_DEBUG:
            data = data_dict.get(self.data_key)
            pids = [os.path.basename(i["seg_file"]).split(".nii.gz")[0] for i in data_dict.get("properties")]
            print(">>>SimulateLowResolutionTransformMultiData __cal__: ", pids, modalities, data.shape)
        for b in range(len(data_dict[self.data_key])):
            for c in range(len(data_dict[self.data_key][0])):
                if modalities[b][c] != "seg" and np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b][c:c + 1] = \
                        augment_linear_downsampling_scipy(data_dict[self.data_key][b][c:c + 1],
                                                          zoom_range=self.zoom_range,
                                                          per_channel=self.per_channel,
                                                          p_per_channel=self.p_per_channel,
                                                          channels=self.channels,
                                                          order_downsample=self.order_downsample,
                                                          order_upsample=self.order_upsample,
                                                          ignore_axes=self.ignore_axes)
        return data_dict


class GammaTransformMultiData(GammaTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data",
                 retain_stats: Union[bool, Callable[[], bool]] = False, p_per_sample=1):
        super().__init__(gamma_range, invert_image, per_channel, data_key, retain_stats, p_per_sample)

    def __call__(self, **data_dict):
        modalities = [list(i["all_modalities"].values()) for i in data_dict.get("properties")]
        if TRANSFORM_DEBUG:
            data = data_dict.get(self.data_key)
            pids = [os.path.basename(i["seg_file"]).split(".nii.gz")[0] for i in data_dict.get("properties")]
            print(">>>GammaTransformMultiData __cal__: ", pids, modalities, data.shape)
        for b in range(len(data_dict[self.data_key])):
            for c in range(len(data_dict[self.data_key][0])):
                if modalities[b][c] != "seg" and np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b][c:c + 1] = \
                        augment_gamma(data_dict[self.data_key][b][c:c + 1], self.gamma_range,
                                      self.invert_image,
                                      per_channel=self.per_channel,
                                      retain_stats=self.retain_stats)
        return data_dict

class SaveTransform(AbstractTransform):
    def __init__(self, save_base, prefix, data_key="data", seg_key="seg"):
        self.save_base = save_base
        self.prefix = prefix
        self.data_key = data_key
        self.seg_key = seg_key

    def __call__(self, **data_dict):
        if not TRANSFORM_DEBUG:
            return data_dict

        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.seg_key)
        seg_path = [i["seg_file"] for i in data_dict.get("properties")]
        data_path = [i["list_of_data_files"] for i in data_dict.get("properties")]
        pid = [os.path.basename(s).split(".nii.gz")[0] for s in seg_path]
        print("saving training file...: ", pid, self.prefix)
        for b in range(data.shape[0]):
            for c in range(data.shape[1]):
                save_path = os.path.join(self.save_base, pid[b] +
                                         "_{}_patch{}_channel{}_data.nii.gz".format(self.prefix, b, c))
                if not os.path.exists(save_path):
                    sitk.WriteImage(sitk.GetImageFromArray(data[b][c]),
                                    save_path)
            save_path = os.path.join(self.save_base, pid[b] + "_{}_patch{}_seg.nii.gz".format(self.prefix, b))
            if not os.path.exists(save_path):
                sitk.WriteImage(sitk.GetImageFromArray(seg[b][0]), save_path)
        if self.prefix == "before":
            for s in seg_path:
                shutil.copy(s, os.path.join(self.save_base, os.path.basename(s)))
            for fl in data_path:
                for f in fl:
                    shutil.copy(f, os.path.join(self.save_base, os.path.basename(f)))
        print("saved training file: ", pid, self.prefix)

        return data_dict


class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(self, regions: dict, seg_key: str = "seg", output_key: str = "seg", seg_channel: int = 0):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r, k in enumerate(self.regions.keys()):
                    for l in self.regions[k]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict
