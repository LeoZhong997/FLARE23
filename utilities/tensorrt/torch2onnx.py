import os.path

import numpy as np
import onnx
import torch
from nnunet.training.model_restore import load_model_and_checkpoint_files
import onnxruntime as ort
from nnunet.utilities.random_stuff import no_op

from torch.cuda.amp import autocast


def torch_convert_onnx():
    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(base_dir, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    trainer.load_checkpoint_ram(params[0], False)
    trainer.network.do_ds = False
    trainer.network.eval()
    print("net: ", trainer.network, trainer.network.get_device(), trainer.network.training)

    # handle set norm op to false
    # for m in trainer.network.modules():
    #     print(m.__class__.__name__.lower(), m.training)
    #     if "instancenorm" in m.__class__.__name__.lower():
    #         m.train(False)

    # convert torch format to onnx
    if mixed_precision:
        context = autocast
    else:
        context = no_op
    inputs_name = ["input"]
    output_name = ["output"]
    with context():
        with torch.no_grad():
            torch.onnx.export(trainer.network, dummy_input, onnx_save_path,
                              verbose=True, input_names=inputs_name, output_names=output_name)
            print("convert torch format model to onnx...")

    # confirm the onnx file
    # net = onnx.load(onnx_save_path)
    # # check the onnx model
    # onnx.checker.check_model(net)
    # onnx.helper.printable_graph(net.graph)


def torch_predict():
    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(base_dir, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    trainer.load_checkpoint_ram(params[0], False)
    trainer.network.do_ds = False
    trainer.network.eval()

    # convert torch format to onnx
    if mixed_precision:
        context = autocast
    else:
        context = no_op
    with context():
        with torch.no_grad():
            torch_output = trainer.network(dummy_input)
        np.save(base_dir + "torch_output.npy", torch_output.detach().cpu().numpy())


def onnx_predict():
    ort_session = ort.InferenceSession(onnx_save_path, providers=torch.device("cuda"))
    onnx_output = ort_session.run(None, {"input": dummy_input_array})
    print(len(onnx_output))
    onnx_output = np.array(onnx_output[0])
    np.save(base_dir + "onnx_output.npy", onnx_output)
    onnx_output1 = torch.from_numpy(onnx_output)
    print("ONNX output: ", onnx_output1.get_device(), onnx_output1.shape, onnx_output1[0, :, 0, 0, 0])


def diff_output():
    onnx_path = base_dir + "onnx_output.npy"
    torch_path = base_dir + "torch_output.npy"

    onnx_re = np.load(onnx_path)
    torch_re = np.load(torch_path)

    diff = onnx_re - torch_re
    diff_abs = np.abs(diff)

    print("diff: ", np.min(diff), np.max(diff), np.mean(diff), np.sum(diff))
    print("diff_abs: ", np.min(diff_abs), np.max(diff_abs), np.mean(diff_abs), np.sum(diff_abs))


if __name__ == "__main__":
    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/predict_temp/tensorrt_test/torch_model/" \
               "Task030_FLARE23OARTumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium/"
    onnx_save_path = base_dir + "nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium.onnx"

    folds = "all"
    mixed_precision = False
    checkpoint_name = "model_final_checkpoint"
    # dummy_input_array = np.random.randn(1, 1, 32, 128, 192).astype(np.float32)
    # dummy_input = torch.from_numpy(dummy_input_array).float().to(torch.device("cuda"))
    # dummy_input = torch.randn((1, 1, 32, 128, 192), device="cuda")
    dummy_input = torch.ones((1, 1, 32, 128, 192), device="cuda")
    dummy_input_array = dummy_input.detach().cpu().numpy()
    print(torch.sum(dummy_input), np.sum(dummy_input_array))

    # torch_convert_onnx()

    # onnx_predict()
    # torch_predict()
    diff_output()



