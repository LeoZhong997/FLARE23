import os.path

import numpy as np
import SimpleITK as sitk


def pred2label(pred):
    """
    pred: [N, C, D, H, W]
    return: [N, C, D, H]
    """
    pred = np.transpose(pred, (0, 2, 3, 4, 1))  # [N,D,H,W,C]

    # softmax -- numpy version
    # B, D, H, W, C = pred.shape
    pred = numpy_softmax(pred)
    pos = np.argmax(pred, axis=-1)      # [N,D,H,W]

    return pos


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


def diff_output():
    torch_path = base_dir + "torch_output.npy"
    onnx_path = base_dir + "onnx_output.npy"
    trt_path = base_dir + "trt_output.npy"

    onnx_re = np.load(onnx_path)
    torch_re = np.load(torch_path)
    torch_re = np.expand_dims(torch_re, axis=0)
    trt_re = np.load(trt_path)
    print(onnx_re.shape, torch_re.shape, trt_re.shape)

    onnx_label = pred2label(onnx_re)
    sitk.WriteImage(sitk.GetImageFromArray(onnx_label[0]), os.path.join(base_dir, "onnx_label.nii.gz"))
    print("onnx_label: ", np.min(onnx_label), np.max(onnx_label), np.mean(onnx_label), np.sum(onnx_label))
    torch_label = pred2label(torch_re)
    sitk.WriteImage(sitk.GetImageFromArray(torch_label[0]), os.path.join(base_dir, "torch_label.nii.gz"))
    print("torch_label: ", np.min(torch_label), np.max(torch_label), np.mean(torch_label), np.sum(torch_label))
    trt_label = pred2label(trt_re)
    sitk.WriteImage(sitk.GetImageFromArray(trt_label[0]), os.path.join(base_dir, "trt_label.nii.gz"))
    print("trt_label: ", np.min(trt_label), np.max(trt_label), np.mean(trt_label), np.sum(trt_label))

    diff = onnx_re - torch_re
    diff_abs = np.abs(diff)
    print("onnx-torch diff: ", np.min(diff), np.max(diff), np.mean(diff), np.sum(diff))
    print("onnx-torch diff_abs: ", np.min(diff_abs), np.max(diff_abs), np.mean(diff_abs), np.sum(diff_abs))
    diff = trt_re - onnx_re
    diff_abs = np.abs(diff)
    print("trt-onnx diff: ", np.min(diff), np.max(diff), np.mean(diff),  np.sum(diff))
    print("trt-onnx diff_abs: ", np.min(diff_abs), np.max(diff_abs), np.mean(diff_abs), np.sum(diff_abs))

    diff = onnx_label - torch_label
    diff_abs = np.abs(diff)
    print("onnx-torch label diff: ", np.min(diff), np.max(diff), np.mean(diff), np.sum(diff))
    print("onnx-torch label diff_abs: ", np.min(diff_abs), np.max(diff_abs), np.mean(diff_abs), np.sum(diff_abs))
    diff = trt_label - onnx_label
    diff_abs = np.abs(diff)
    print("trt-onnx label diff: ", np.min(diff), np.max(diff), np.mean(diff))
    print("trt-onnx label diff_abs: ", np.min(diff_abs), np.max(diff_abs), np.mean(diff_abs), np.sum(diff_abs))


if __name__ == "__main__":

    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/tensorrt_test/predict_temp/torch_model/" \
               "Task030_FLARE23OARTumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium/"

    diff_output()

