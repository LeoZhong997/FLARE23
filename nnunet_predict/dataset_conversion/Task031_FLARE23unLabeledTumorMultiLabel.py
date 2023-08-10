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


def process(pid, image_dir, label_dir, pseudo_tumor_dir, mixed_oar_label_dir, target_imagesTr, target_labelsTr):
    ct = join(image_dir, pid + "_0000.nii.gz")
    for img_sub_dir in ["1-500", "501-1000", "1001-1500", "1501-2000", "FLARE23_1001-1500", "FLARE23_2001-2200"]:
        if isfile(ct):
            break
        else:
            ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
    mixed_oar_label = join(mixed_oar_label_dir, pid + ".nii.gz")  # without tumor label
    pseudo_label = join(pseudo_tumor_dir, pid + ".nii.gz")  # tumor pseudo with multi label
    gt_label = join(label_dir, pid + ".nii.gz")  # tumor gt label
    assert all([isfile(ct), isfile(mixed_oar_label)]), "{} has wrong paths".format(pid)
    if os.path.exists(pseudo_label):
        label_mode = "pseudo"
        gt_img = sitk.ReadImage(pseudo_label)
    else:
        label_mode = "gt"
        gt_img = sitk.ReadImage(gt_label)
    print("processing: ", pid, label_mode)

    new_ct = join(target_imagesTr, pid + "_0000.nii.gz")
    tumor_label = join(target_labelsTr, pid + ".nii.gz")
    # if not (isfile(new_ct) and isfile(tumor_label)):
    if True:
        ct_img = sitk.ReadImage(ct)
        ct_direction = ct_img.GetDirection()
        mixed_oar_label_img = sitk.ReadImage(mixed_oar_label)
        mixed_oar_direction = mixed_oar_label_img.GetDirection()

        gt_npy = sitk.GetArrayFromImage(gt_img).astype(np.uint8)
        mixed_oar_npy = sitk.GetArrayFromImage(mixed_oar_label_img).astype(np.uint8)
        print(pid, ct_direction, mixed_oar_direction)
        print(pid, ct_img.GetOrigin(), mixed_oar_label_img.GetOrigin())
        print(pid, "label_mode, gt_npy, mixed_oar_npy: ", label_mode, np.unique(gt_npy), np.unique(mixed_oar_npy))
        if -1 not in ct_direction and -1 not in mixed_oar_direction:
            shutil.copy(ct, new_ct)

            if label_mode == "gt":
                tumor_npy = gt_npy * 0
                mixed_oar_npy[gt_npy != 14] = 0
                tumor_npy[gt_npy == 14] = 14
                for i in np.unique(mixed_oar_npy):
                    if i == 0:
                        continue
                    tumor_npy[mixed_oar_npy == i] = i
            else:
                tumor_npy = gt_npy * 1

            tumor_img = sitk.GetImageFromArray(tumor_npy)
            tumor_img.CopyInformation(ct_img)
            sitk.WriteImage(tumor_img, tumor_label)
        else:
            ct_spacing = ct_img.GetSpacing()
            ct_origin = ct_img.GetOrigin()

            if -1 not in mixed_oar_direction:
                [new_ct_img, new_gt_img] = sitk_correct_direction(
                    [ct_img, gt_img], "LPS")
            else:
                [new_ct_img, new_gt_img, new_mixed_oar_label_img] = sitk_correct_direction(
                    [ct_img, gt_img, mixed_oar_label_img], "LPS")
                mixed_oar_npy = sitk.GetArrayFromImage(new_mixed_oar_label_img)
            print(pid, "correct_direction: \nbefore: {} \nafter: {}".format(
                [ct_direction, ct_origin, ct_spacing],
                [new_ct_img.GetDirection(), new_ct_img.GetOrigin(), new_ct_img.GetSpacing()]))

            sitk.WriteImage(new_ct_img, new_ct)

            gt_npy = sitk.GetArrayFromImage(new_gt_img).astype(np.uint8)
            if label_mode == "gt":
                tumor_npy = gt_npy * 0
                mixed_oar_npy[gt_npy != 14] = 0
                tumor_npy[gt_npy == 14] = 14
                for i in np.unique(mixed_oar_npy):
                    if i == 0:
                        continue
                    tumor_npy[mixed_oar_npy == i] = i
            else:
                tumor_npy = gt_npy * 1

            new_gt_img = sitk.GetImageFromArray(tumor_npy)
            new_gt_img.CopyInformation(new_ct_img)
            sitk.WriteImage(new_gt_img, tumor_label)
        print(pid, "multi_labeled tumor_npy: ", np.unique(tumor_npy))


if __name__ == "__main__":
    num_processes = 8

    task_name = "Task031_FLARE23unLabeledTumorMultiLabel"
    data_dir = "/data/result/herongxuan/dataset/Release"
    label_dir = join(data_dir, "labelsTr2200")
    mixed_oar_label_dir = "/home/zhongzhiqiang/Dataset/FLARE2023/nnUNet_raw_data/" \
                          "Task016_FLARE23BigAllMixed/labelsTr"  # mixed oar label
    pseudo_tumor_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesUnlabeledTumor/pseudoLabelsTr"
    image_dir = join(data_dir, "imagesTr2200")
    val_image_dir = join(data_dir, "validation")

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

    # for pat in patient_names[:10]:
    #     process(pat, image_dir, label_dir, pseudo_tumor_dir, mixed_oar_label_dir, target_imagesTr, target_labelsTr)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []
        results.append(
            p.starmap_async(
                process, zip(patient_names,
                             [image_dir] * len(patient_names),
                             [label_dir] * len(patient_names),
                             [pseudo_tumor_dir] * len(patient_names),
                             [mixed_oar_label_dir] * len(patient_names),
                             [target_imagesTr] * len(patient_names),
                             [target_labelsTr] * len(patient_names))
            )
        )

        [i.get() for i in results]

    json_dict = OrderedDict()
    json_dict["name"] = "FLARE23unLabeledTumorMultiLabel"
    json_dict["description"] = "FLARE2023"
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = "see FLARE2023"
    json_dict["licence"] = "see FLARE2023 licence"
    json_dict["release"] = "0.0"
    json_dict["modality"] = {
        "0": "CT"
    }
    json_dict["labels"] = {
        "0": "background",
        "1": "tumor_1",
        "2": "tumor_2",
        "3": "tumor_3",
        "4": "tumor_4",
        "5": "tumor_5",
        "6": "tumor_6",
        "7": "tumor_7",
        "8": "tumor_8",
        "9": "tumor_9",
        "10": "tumor_10",
        "11": "tumor_11",
        "12": "tumor_12",
        "13": "tumor_13",
        "14": "tumor",
    }
    json_dict["numTraining"] = len(patient_names)
    json_dict["numTest"] = 0
    json_dict["training"] = [{"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                             for i in patient_names]
    json_dict["test"] = []

    save_json(json_dict, os.path.join(target_base, "dataset.json"))
