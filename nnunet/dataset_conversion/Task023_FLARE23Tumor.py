import multiprocessing
import os
import shutil
from collections import OrderedDict

import SimpleITK as sitk
import numpy as np

import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.paths import nnUNet_raw_data


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


def process(pid, image_dir, label_dir, target_imagesTr, target_labelsTr):
    ct = join(image_dir, pid + "_0000.nii.gz")
    for img_sub_dir in ["1-500", "501-1000", "1001-1500", "1501-2000"]:
        if isfile(ct):
            break
        else:
            ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
    seg = join(label_dir, pid + ".nii.gz")
    assert all([isfile(ct), isfile(seg)]), "{} has wrong paths".format(pid)
    print(pid, ct, seg)

    shutil.copy(ct, join(target_imagesTr, pid + "_0000.nii.gz"))

    copy_FLARE2023_tumor_segmentation_and_convert_labels(seg, join(target_labelsTr, pid + ".nii.gz"))


if __name__ == "__main__":
    num_processes = 8

    task_name = "Task023_FLARE23Tumor"
    data_dir = "/data/result/herongxuan/dataset/Release"
    label_dir = join(data_dir, "labelsTr2200")
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
    patient_names = df["case_id"].values.tolist()
    print(len(patient_names), patient_names[:10])

    # with multiprocessing.get_context("spawn").Pool(num_processes) as p:
    #     results = []
    #
    #     results.append(
    #         p.starmap_async(
    #             process, zip(patient_names,
    #                          [image_dir] * len(patient_names),
    #                          [label_dir] * len(patient_names),
    #                          [target_imagesTr] * len(patient_names),
    #                          [target_labelsTr] * len(patient_names))
    #         )
    #     )
    #
    #     [i.get() for i in results]

        # for pid in patient_names[:]:
        #     ct = join(image_dir, pid + "_0000.nii.gz")
        #     for img_sub_dir in ["1-500", "501-1000", "1001-1500", "1501-2000"]:
        #         if isfile(ct):
        #             break
        #         else:
        #             ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
        #     seg = join(label_dir, pid + ".nii.gz")
        #     assert all([isfile(ct), isfile(seg)]), "{} has wrong paths".format(pid)
        #     print(pid, ct, seg)
        #
        #     shutil.copy(ct, join(target_imagesTr, pid + "_0000.nii.gz"))
        #
        #     copy_FLARE2023_tumor_segmentation_and_convert_labels(seg, join(target_labelsTr, pid + ".nii.gz"))
        # break

    json_dict = OrderedDict()
    json_dict["name"] = "FLARE23Tumor"
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
        "1": "tumor"
    }
    json_dict["numTraining"] = len(patient_names)
    json_dict["numTest"] = 0
    json_dict["training"] = [{"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                             for i in patient_names]
    json_dict["test"] = []

    save_json(json_dict, os.path.join(target_base, "dataset.json"))

    val_pids = subfiles(val_image_dir, join=False)
    print(len(val_pids), val_pids[:10])
    for p in val_pids:
        pid = p.split("_0000.nii.gz")[0]
        ct = join(val_image_dir, p)
        assert isfile(ct), "%s" % pid

        shutil.copy(ct, join(target_imagesVal, p))

        # break





