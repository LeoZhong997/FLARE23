import os
from typing import Tuple

import nnunet.preprocessing.preprocessing
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, load_pickle

# from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
#     default_2D_augmentation_params

from torch.backends import cudnn

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from collections import OrderedDict

import torchsummary as summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class nnUNetPredictor(object):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        self.fp16 = fp16

        print("deterministic, torch, cuda, cudnn", deterministic, torch.__version__,
              torch.version.cuda, torch.backends.cudnn.version())
        print("default cudnn:", cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            # torch.backends.cudnn.benchmark = True
        print("new cudnn:", cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)

        ################# SET THESE IN self.initialize() ###################################
        self.network: None
        self.was_initialized = False
        self.was_processed_plans = False

        ################# SET THESE IN self.initialize() ###################################
        self.trt_path = ""
        self.trt_mode = False
        self.trt_session = None

        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16)
        # set through arguments from init
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder

        self.plans = None
        self.num_input_channels = self.num_classes = self.net_pool_per_axis = self.patch_size = self.batch_size = \
            self.threeD = self.base_num_features = self.intensity_properties = self.normalization_schemes = \
            self.net_num_pool_op_kernel_sizes = self.net_conv_kernel_sizes = None  # loaded automatically from plans_file
        self.basic_generator_patch_size = self.data_aug_params = self.transpose_forward = self.transpose_backward = None

        self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = self.only_keep_largest_connected_component = \
            self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        # self.update_fold(fold)
        self.pad_all_sides = None

        self.conv_per_stage = None
        self.regions_class_order = None

        self.deep_supervision_scales = None

        self.pin_memory = True

    def collect_network_info_on_cpu(self):
        input_tensor = torch.rand((1, 1, 32, 128, 192))
        flops = FlopCountAnalysis(self.network, input_tensor)
        print("Flops info")
        print("flops total:", flops.total())
        print("parameter_count_table: ")
        print(parameter_count_table(self.network))

    def collect_network_info_on_gpu(self):
        print("summary info")
        summary.summary(self.network, input_size=(self.num_input_channels, 32, 128, 192))

    def process_plans(self, plans):
        if self.was_processed_plans:
            return

        self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            print("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            print("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

        self.was_processed_plans = True

    def initialize(self, training=True, force_load_plans=False, trt_mode=False, trt_path=""):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            if trt_mode:
                self.trt_path = trt_path
                self.trt_mode = trt_mode
                self.load_trt_engine()
            else:
                self.initialize_network()
        else:
            print('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True,
                                                         window_type='fast',
                                                         trt_mode=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """

        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        if trt_mode:
            ret = self.trt_session.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                              use_sliding_window=use_sliding_window, step_size=step_size,
                                              patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                              use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                              pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                              mixed_precision=mixed_precision, window_type=window_type)
        else:
            self.network.do_ds = False
            self.network.eval()
            ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                          use_sliding_window=use_sliding_window, step_size=step_size,
                                          patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                          use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                          pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                          mixed_precision=mixed_precision, window_type=window_type)
        return ret

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        preprocessor_class = nnunet.preprocessing.preprocessing.GenericPreprocessor
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                          self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties

    def initialize_network(self):
        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 32
        self.max_num_features = 512
        self.max_num_epochs = 1500

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    self.max_num_features)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.data_aug_params = {}
        self.data_aug_params['mirror_axes'] = (0, 1, 2)

    def load_trt_engine(self):
        from nnunet.utilities.trt_utils import TensorRTSession
        self.trt_session = TensorRTSession(self.trt_path, self.num_classes)
        # from nnunet.utilities.trt_utils import TRTModule
        # self.trt_session = TRTModule(self.trt_path, self.num_classes)

        # from nnunet.utilities.trt_new_utils import TensorRTPredictor
        # self.trt_session = TensorRTPredictor(self.trt_path, self.num_classes)
        # from nnunet.utilities.trt_new_utils import TensorRTwithTorchPredictor
        # self.trt_session = TensorRTwithTorchPredictor(self.trt_path, self.num_classes)

    def load_checkpoint(self, fname, train=True):
        print("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.network.load_state_dict(new_state_dict)

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)


class nnUNetTrainerV2_FLARE_Medium_Pure(nnUNetPredictor):
    def initialize_network(self):
        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 32
        self.max_num_features = 512
        self.max_num_epochs = 1500

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                                    self.max_num_features)

        self.collect_network_info_on_cpu()
        if torch.cuda.is_available():
            self.network.cuda()
            self.collect_network_info_on_gpu()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = True
        self.data_aug_params["do_elastic"] = True


class nnUNetTrainerV2_FLARE_Large_Pure(nnUNetPredictor):
    def initialize_network(self):
        self.conv_per_stage = 3
        self.base_num_features = 32
        self.max_num_features = 512
        self.max_num_epochs = 1500

        # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, self.max_num_features)

        self.collect_network_info_on_cpu()
        if torch.cuda.is_available():
            self.network.cuda()
            self.collect_network_info_on_gpu()
        self.network.inference_apply_nonlin = softmax_helper

    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = True
        self.data_aug_params["do_elastic"] = True
