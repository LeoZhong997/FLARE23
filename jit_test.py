import os
import time

import torch
from nnunet.training.model_restore import load_model_and_checkpoint_files


class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h


def tutorial():
    my_cell = MyCell()
    x, h = torch.rand(3, 4), torch.rand(3, 4)
    print("y1: ", my_cell(x, h))

    traced_cell = torch.jit.trace(my_cell, (x, h))
    print("traced_cell: ", traced_cell)
    print("y2: ", traced_cell(x, h))
    print("graph: ", traced_cell.graph)
    print("code: ", traced_cell.code)

    save_path = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/jit_test/MeCell.pt"
    traced_cell.save(save_path)

    loaded_cell = torch.jit.load(save_path)
    print("loaded_cell: ", loaded_cell)
    print("code: ", loaded_cell.code)
    print("y3: ", loaded_cell(x, h))


def nnUNet_model_transfer_jit():
    folds = "all"
    mixed_precision = True
    checkpoint_name = "model_final_checkpoint"
    model_folder = "/data/result/zhongzhiqiang/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/" \
                   "Task026_FLARE23TumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium"
    jit_save_path = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/jit_test/" \
                    "nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium.pt"

    input_patch_size = [1, 1, 32, 128, 192]
    x = torch.rand(input_patch_size, device="cuda:0")   # , dtype=torch.float32
    print("x: ", x.shape, x.get_device())

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model_folder, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    print("trainer: ", trainer)      # <nnunet.training.network_training.nnUNetTrainerV2_FLARE.nnUNetTrainerV2_FLARE_Medium object at 0x7f5156f88250>
    print("params: ", len(params), params[0]["epoch"])       # dict{'epoch': 1500, 'state_dict': ..., 'optimizer_state_dict': ..., ,,,}

    for i, p in enumerate(params):
        trainer.load_checkpoint_ram(p, False)
        model = trainer.network
        model.eval()
        model_re = model(x)     # (torch.Size([1, 15, 32, 128, 192]))

        if os.path.exists(jit_save_path):
            traced_model = torch.jit.load(jit_save_path)
            print("loaded traced_model: ", traced_model)
        else:
            traced_model = torch.jit.trace(model, x)
            traced_model.save(jit_save_path)
            print("saved traced_model: ", traced_model)
        traced_model_re = traced_model(x)
        print("model_re: ", model_re[0].shape, model_re[0][0, :, 16, 64, 96])
        print("traced_model_re: ", traced_model_re[0].shape, traced_model_re[0][0, :, 16, 64, 96])


def torch_model_test():
    folds = "all"
    mixed_precision = True
    checkpoint_name = "model_final_checkpoint"
    model_folder = "/data/result/zhongzhiqiang/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/" \
                   "Task026_FLARE23TumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium"

    input_patch_size = [1, 1, 32, 128, 192]
    x = torch.rand(input_patch_size, device="cuda:0")  # , dtype=torch.float32
    print("x: ", x.shape, x.get_device())

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model_folder, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    # <nnunet.training.network_training.nnUNetTrainerV2_FLARE.nnUNetTrainerV2_FLARE_Medium object at 0x7f5156f88250>
    print("trainer: ", trainer)
    # [dict{'epoch': 1500, 'state_dict': ..., 'optimizer_state_dict': ..., ,,,}]
    print("params: ", len(params), params[0]["epoch"])

    for i, p in enumerate(params):
        t0 = time.time()
        trainer.load_checkpoint_ram(p, False)
        model = trainer.network
        model.eval()
        print("loaded model: ", model)

        t1 = time.time()
        for n in range(1000):
            model_re = model(x)  # (torch.Size([1, 15, 32, 128, 192]))
        t2 = time.time()
        print("model time: ", t2 - t1, t2 - t0)
        print("model_re: ", model_re[0].shape, model_re[0][0, :, 16, 64, 96])


def jit_model_test():
    input_patch_size = [1, 1, 32, 128, 192]
    x = torch.rand(input_patch_size, device="cuda:0")  # , dtype=torch.float32
    print("x: ", x.shape, x.get_device())

    jit_save_path = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/jit_test/" \
                    "nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium.pt"

    t0 = time.time()
    traced_model = torch.jit.load(jit_save_path)
    print("loaded traced_model: ", traced_model)

    t1 = time.time()
    for n in range(1000):
        traced_model_re = traced_model(x)
    t2 = time.time()
    print("traced_model time: ", t2 - t1, t2 - t0)
    print("traced_model_re: ", traced_model_re[0].shape, traced_model_re[0][0, :, 16, 64, 96])


if __name__ == "__main__":

    # tutorial()

    # nnUNet_model_transfer_jit()
    # torch_model_test()
    jit_model_test()
