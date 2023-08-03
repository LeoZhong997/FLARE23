import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import time

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss_weighted


class nnUNetTrainerV2_FLARE_Mixed(nnUNetTrainerV2):
    # only used in training phase
    # def initialize(self, training=True, force_load_plans=False):
    #     super().initialize(training, force_load_plans)
    #     self.singleloss = DC_and_CE_loss_weighted({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
    #     self.loss = MultipleOutputLoss2(self.singleloss, self.ds_loss_weights)

    def initialize_network(self):
        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 16
        self.max_num_features = 256
        self.max_num_epochs = 1500
        
        # accerlerate training
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False
        

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num-1]

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
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = True
        self.data_aug_params["do_elastic"] = True
