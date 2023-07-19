import multiprocessing
import os
import shutil
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.paths import nnUNet_raw_data


def correct_direction(img_npy, direction, origin, spacing):
    if -1 not in direction:
        return img_npy, direction, origin
    direction_temp = list(direction)
    direction_temp[8] = direction_temp[0] = direction_temp[4] = 1
    assert -1 not in direction_temp, \
        "direction {} correction failed, direction_temp is {}".format(direction, direction_temp)

    if not isinstance(img_npy, list):
        img_npy = [img_npy]

    new_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    for i, npy in enumerate(img_npy):
        if int(direction[8]) == -1:
            npy = npy[::int(direction[8])]
        if int(direction[0]) == -1:
            npy = npy[:, ::int(direction[0]), :]
        if int(direction[4]) == -1:
            npy = npy[:, :, ::int(direction[4])]
        img_npy[i] = npy

    img_size = img_npy[0].shape
    new_origin = list(origin)
    if int(direction[8]) == -1:
        new_origin[0] = origin[0] + spacing[0] * (img_size[0] - 1) * int(direction[8])
    if int(direction[0]) == -1:
        new_origin[1] = origin[1] + spacing[1] * (img_size[1] - 1) * int(direction[0])
    if int(direction[4]) == -1:
        new_origin[2] = origin[2] + spacing[2] * (img_size[2] - 1) * int(direction[4])
    return img_npy, new_direction, new_origin


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


def process(pid, image_dir, label_dir, pseudo_label_dir, target_imagesTr, target_labelsTr):
    second_channel_name = "pseudo"  # "pseudo" or "gt"

    ct = join(image_dir, pid + "_0000.nii.gz")
    for img_sub_dir in ["1-500", "501-1000", "1001-1500", "1501-2000"]:
        if isfile(ct):
            break
        else:
            ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
    pseudo_label = join(pseudo_label_dir, pid + ".nii.gz")  # without tumor label
    gt_label = join(label_dir, pid + ".nii.gz")  # with tumor label
    assert all([isfile(ct), isfile(gt_label)]), "{} has wrong paths".format(pid)
    print("pass: ", pid)

    new_ct = join(target_imagesTr, pid + "_0000.nii.gz")
    aux_label = join(target_imagesTr, pid + "_0001.nii.gz")
    tumor_label = join(target_labelsTr, pid + ".nii.gz")
    if not (isfile(new_ct) and isfile(aux_label) and isfile(tumor_label)):
        ct_img = sitk.ReadImage(ct)
        ct_direction = ct_img.GetDirection()
        print(pid, ct_direction, ct, gt_label, pseudo_label)
        if -1 not in ct_direction:
            shutil.copy(ct, new_ct)
            if second_channel_name == "pseudo":
                shutil.copy(pseudo_label, aux_label)
            else:
                copy_gt_label_without_tumor(gt_label, aux_label)

            copy_FLARE2023_tumor_segmentation_and_convert_labels(gt_label, tumor_label)
        else:
            ct_npy = sitk.GetArrayFromImage(ct_img)
            ct_spacing = ct_img.GetSpacing()
            ct_origin = ct_img.GetOrigin()

            gt_npy = sitk.GetArrayFromImage(sitk.ReadImage(gt_label))
            pseudo_npy = sitk.GetArrayFromImage(sitk.ReadImage(pseudo_label))

            [ct_npy, gt_npy, pseudo_npy], new_direction, new_origin = correct_direction(
                [ct_npy, gt_npy, pseudo_npy], ct_direction, ct_origin, ct_spacing)
            print(pid, "correct_direction: \nbefore: {} \nafter: {}".format(
                [ct_direction, ct_origin, ct_spacing], [new_direction, new_origin]))

            new_ct_img = sitk.GetImageFromArray(ct_npy)
            new_ct_img.SetSpacing(ct_spacing)
            new_ct_img.SetOrigin(new_origin)
            new_ct_img.SetDirection(new_direction)
            sitk.WriteImage(new_ct_img, new_ct)

            tumor_npy = np.zeros_like(gt_npy)
            tumor_npy[gt_npy == 14] = 1
            new_gt_img = sitk.GetImageFromArray(tumor_npy)
            new_gt_img.SetSpacing(ct_spacing)
            new_gt_img.SetOrigin(new_origin)
            new_gt_img.SetDirection(new_direction)
            sitk.WriteImage(new_gt_img, tumor_label)

            new_pseudo_img = sitk.GetImageFromArray(pseudo_npy)
            new_pseudo_img.SetSpacing(ct_spacing)
            new_pseudo_img.SetOrigin(new_origin)
            new_pseudo_img.SetDirection(new_direction)
            sitk.WriteImage(new_pseudo_img, aux_label)

            print(pid, "origin check: {}, {}, {}".format(
                new_ct_img.GetOrigin(), new_gt_img.GetOrigin(), new_pseudo_img.GetOrigin()))


if __name__ == "__main__":
    num_processes = 8

    task_name = "Task025_FLARE23TumorWithOAR"
    data_dir = "/data/result/herongxuan/dataset/Release"
    label_dir = join(data_dir, "labelsTr2200")
    aladdin5_pseudo_label_dir = join(data_dir, "aladdin5-pseudo-labels-FLARE23")
    blackbean_pseudo_label_dir = join(data_dir, "blackbean-pseudo-labels-FLARE23")  # method without postprocess
    image_dir = join(data_dir, "imagesTr2200")
    val_image_dir = join(data_dir, "validation")
    pid_table_path = "/home/zhongzhiqiang/PreResearch/FLARE2023/tables/labelsTr1497_fullTumor.xlsx"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    df = pd.read_excel(pid_table_path)
    patient_names = df["case_id"].values.tolist()[:]
    # patient_names = ["FLARE23_0609"]
    print(len(patient_names), patient_names[:5])

    # for pat in patient_names[:10]:
    #     process(pat, image_dir, label_dir, aladdin5_pseudo_label_dir, target_imagesTr, target_labelsTr)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        results = []

        results.append(
            p.starmap_async(
                process, zip(patient_names,
                             [image_dir] * len(patient_names),
                             [label_dir] * len(patient_names),
                             [aladdin5_pseudo_label_dir] * len(patient_names),
                             [target_imagesTr] * len(patient_names),
                             [target_labelsTr] * len(patient_names))
            )
        )

        [i.get() for i in results]

    json_dict = OrderedDict()
    json_dict["name"] = "FLARE23TumorWithOAR"
    json_dict["description"] = "FLARE2023"
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = "see FLARE2023"
    json_dict["licence"] = "see FLARE2023 licence"
    json_dict["release"] = "0.0"
    json_dict["modality"] = {
        "0": "CT",
        "1": "seg"
    }
    json_dict["labels"] = {
        "0": "background",
        "1": "tumor"
    }
    json_dict["numTraining"] = len(patient_names)
    json_dict["numTest"] = 0
    json_dict["training"] = [{"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                             for i in patient_names]
    json_dict["test"] = []

    save_json(json_dict, os.path.join(target_base, "dataset.json"))

    # val_pids = subfiles(val_image_dir, join=False)
    # print(len(val_pids), val_pids[:10])
    # for p in val_pids:
    #     pid = p.split("_0000.nii.gz")[0]
    #     ct = join(val_image_dir, p)
    #     assert isfile(ct), "%s" % pid
    #
    #     shutil.copy(ct, join(target_imagesVal, p))
