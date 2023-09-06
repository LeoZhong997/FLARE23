import glob
import multiprocessing
import os
import shutil
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.paths import nnUNet_raw_data


def sitk_correct_direction(img_list, orientation="LPS"):
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)
    for i, img in enumerate(img_list):
        img = orient_filter.Execute(img)
        img_list[i] = img
    return img_list


def copy_FLARE2023_tumor_segmentation_and_convert_labels(in_file, out_file):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    if 14 not in uniques:
        raise RuntimeError("unexpected label")

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 14] = 1
    img_coor = sitk.GetImageFromArray(seg_new)
    img_coor.CopyInformation(img)
    sitk.WriteImage(img_coor, out_file)


def copy_gt_label_without_tumor(in_file, out_file):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    img_npy[img_npy == 14] = 0
    new_img = sitk.GetImageFromArray(img_npy)
    new_img.CopyInformation(img)
    sitk.WriteImage(new_img, out_file)


def process(pid, image_dir, oar_tumor_pred_dir, target_imagesTr, target_labelsTr):
    ct = join(image_dir, pid + "_0000.nii.gz")
    for img_sub_dir in ["1-500", "501-1000", "1001-1500", "1501-2000", "FLARE23_1001-1500", "FLARE23_2001-2200"]:
        if isfile(ct):
            break
        else:
            ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
    oar_tumor_pred = join(oar_tumor_pred_dir, pid + ".nii.gz")  # oar and tumor pred
    assert all([isfile(ct), isfile(oar_tumor_pred)]), "{} has wrong paths".format(pid)
    print("processing: ", pid)

    new_ct = join(target_imagesTr, pid + "_0000.nii.gz")
    new_label = join(target_labelsTr, pid + ".nii.gz")
    if not (isfile(new_ct) and isfile(new_label)):
    # if True:
        ct_img = sitk.ReadImage(ct)
        ct_direction = ct_img.GetDirection()
        oar_tumor_pred_img = sitk.ReadImage(oar_tumor_pred)
        oar_tumor_pred_direction = oar_tumor_pred_img.GetDirection()

        oar_tumor_pred_npy = sitk.GetArrayFromImage(oar_tumor_pred_img).astype(np.uint8)
        oar_tumor_pred_unique = np.unique(oar_tumor_pred_npy)
        print(pid, ct_direction, oar_tumor_pred_direction)
        print(pid, ct_img.GetOrigin(), oar_tumor_pred_img.GetOrigin())
        print(pid, "oar_tumor_pred_npy: ", oar_tumor_pred_unique)
        if -1 not in ct_direction and -1 not in oar_tumor_pred_direction:
            shutil.copy(ct, new_ct)
            shutil.copy(oar_tumor_pred, new_label)
        else:
            [new_ct_img, new_oar_tumor_pred_img] = sitk_correct_direction(
                [ct_img, oar_tumor_pred_img], "LPS")

            sitk.WriteImage(new_ct_img, new_ct)
            sitk.WriteImage(new_oar_tumor_pred_img, new_label)


if __name__ == "__main__":
    """
    2200 cases with partial-label dataset can refer to Task033.
    If use model without retrained, you can just copy results from Task033 to Task034
    this code is for 1800 cases with no-label dataset
    """
    num_processes = 8

    task_name = "Task034_FLARE23noLabeledOARTumorMultiTask"
    data_dir = "/data/result/herongxuan/dataset/Release"
    oar_tumor_pred_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/training_dataset/" \
                          "Task030__nnUNetPlansFLARE23TumorMedium__all_do_mirror_fullWindow_TTA4_0808"  # oar and tumor
    image_dir = join(data_dir, "unlabelTr1800")

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")
    print("target_base", target_base)
    print("target_imagesTr", target_imagesTr)

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    patient_names = sorted(glob.glob(os.path.join(image_dir, "*", "*.nii.gz")))
    patient_names = [os.path.basename(p).split("_0000.nii.gz")[0] for p in patient_names][:]
    print(len(patient_names), patient_names[:5])

    for pat in patient_names[:10]:
        process(pat, image_dir, oar_tumor_pred_dir, target_imagesTr, target_labelsTr)
    # with multiprocessing.get_context("spawn").Pool(num_processes) as p:
    #     results = []
    #     results.append(
    #         p.starmap_async(
    #             process, zip(patient_names,
    #                          [image_dir] * len(patient_names),
    #                          [oar_tumor_pred_dir] * len(patient_names),
    #                          [target_imagesTr] * len(patient_names),
    #                          [target_labelsTr] * len(patient_names))
    #         )
    #     )
    #
    #     [i.get() for i in results]

    json_dict = OrderedDict()
    json_dict["name"] = "FLARE23noLabeledOARTumorMultiTask"
    json_dict["description"] = "FLARE2023"
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = "see FLARE2023"
    json_dict["licence"] = "see FLARE2023 licence"
    json_dict["release"] = "0.0"
    json_dict["modality"] = {
        "0": "CT"
    }
    json_dict["labels"] = {
        "0": "Background",
        "1": "Liver",
        "2": "Right_kidney",
        "3": "Spleen",
        "4": "Pancreas",
        "5": "Aorta",
        "6": "Inferior_vena_cava",
        "7": "Right_adrenal_gland",
        "8": "Left_adrenal_gland",
        "9": "Gallbladder",
        "10": "Esophagus",
        "11": "Stomach",
        "12": "Duodenum",
        "13": "Left_kidney",
        "14": "Tumor",
    }
    json_dict["numTraining"] = len(patient_names)
    json_dict["numTest"] = 0
    json_dict["training"] = [{"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                             for i in patient_names]
    json_dict["test"] = []

    save_json(json_dict, os.path.join(target_base, "dataset.json"))
