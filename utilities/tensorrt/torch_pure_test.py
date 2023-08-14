import os.path
import time
from collections import OrderedDict

import torch
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from torch.cuda.amp import GradScaler, autocast

from nnunet.utilities.nd_softmax import softmax_helper

from nnunet.utilities.random_stuff import no_op

from nnunet.network_architecture.initialization import InitWeights_He
from torch import nn

from nnunet.network_architecture.generic_UNet import Generic_UNet


class PureNNUNet(object):

    def __init__(self, pkl_file, model_path, mixed_precision):
        self.fp16 = self.mixed_precision = mixed_precision
        self.amp_grad_scaler = None
        
        info = load_pickle(pkl_file)
        # init = info['init']
        # name = info['name']
        plans = info['plans']
        self.plans = plans
        self.model_path = model_path
        # print("plans: ", self.plans)

        self.stage = list(plans['plans_per_stage'].keys())[0]
        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']

        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
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

        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 32
        self.max_num_features = 512

        # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num - 1]

        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d

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

        self.load_params()

        self.set_eval()

    def load_params(self):
        checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())

        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.network.load_state_dict(new_state_dict)

        if self.fp16:
            self._maybe_init_amp()

    def set_eval(self):
        self.network.do_ds = False
        self.network.eval()

        torch.cuda.empty_cache()

    def predict(self):
        dummy_input = torch.ones((1, 1, 32, 128, 192), device="cuda")
        if self.mixed_precision:
            context = autocast
        else:
            context = no_op
        # dummy_input = torch.randn((1, 1, 32, 128, 192), device="cuda").half()
        with context():
            with torch.no_grad():
                torch_output = self.network(dummy_input)
        # print(torch_output.shape)
        return torch_output

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()


if __name__ == "__main__":
    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/predict_temp/tensorrt_test/torch_model/" \
               "Task030_FLARE23OARTumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium/"

    folds = ["all"]
    mixed_precision = True
    checkpoint_name = "model_final_checkpoint"

    model_path = os.path.join(base_dir, folds[0], checkpoint_name + ".model")
    plan_path = os.path.join(base_dir, folds[0], "%s.model.pkl" % checkpoint_name)
    print("model_path", model_path)
    print("plan_path", plan_path)

    t1 = time.time()
    network = PureNNUNet(plan_path, model_path, mixed_precision)
    t2 = time.time()
    print(">>>load model time: {}s".format(t2 - t1))

    for i in range(400):
        network.predict()
    t3 = time.time()
    print(">>>predict time: {}s".format(t3 - t2))
