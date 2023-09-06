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


def process(pid, image_dir, label_dir, oar_tumor_pred_dir, pseudo_label_dir, target_imagesTr, target_labelsTr):
    ct = join(image_dir, pid + "_0000.nii.gz")
    for img_sub_dir in ["1-500", "501-1000", "1001-1500", "1501-2000", "FLARE23_1001-1500", "FLARE23_2001-2200"]:
        if isfile(ct):
            break
        else:
            ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
    pseudo_label = join(pseudo_label_dir, pid + ".nii.gz")  # without tumor label
    oar_tumor_pred = join(oar_tumor_pred_dir, pid + ".nii.gz")  # oar and tumor pred
    gt_label = join(label_dir, pid + ".nii.gz")  # gt
    assert all([isfile(ct), isfile(oar_tumor_pred), isfile(gt_label), isfile(pseudo_label)]), "{} has wrong paths".format(pid)

    new_ct = join(target_imagesTr, pid + "_0000.nii.gz")
    new_label = join(target_labelsTr, pid + ".nii.gz")
    if not (isfile(new_ct) and isfile(new_label)):
        # if True:
        print("processing: ", pid)
        ct_img = sitk.ReadImage(ct)
        ct_direction = ct_img.GetDirection()
        ct_spacing = ct_img.GetSpacing()
        gt_img = sitk.ReadImage(gt_label)
        oar_tumor_pred_img = sitk.ReadImage(oar_tumor_pred)
        pseudo_label_img = sitk.ReadImage(pseudo_label)
        oar_tumor_pred_direction = oar_tumor_pred_img.GetDirection()

        gt_npy = sitk.GetArrayFromImage(gt_img).astype(np.uint8)
        oar_tumor_pred_npy = sitk.GetArrayFromImage(oar_tumor_pred_img).astype(np.uint8)
        pseudo_label_npy = sitk.GetArrayFromImage(pseudo_label_img).astype(np.uint8)
        gt_unique = np.unique(gt_npy)
        oar_tumor_pred_unique = np.unique(oar_tumor_pred_npy)
        pseudo_label_unique = np.unique(pseudo_label_npy)
        print(pid, "ct", ct_direction, ct_spacing, ct_img.GetOrigin())
        # print(pid, "oar_tumor_pred_img", oar_tumor_pred_img.GetDirection(), oar_tumor_pred_img.GetSpacing(),
        #       oar_tumor_pred_img.GetOrigin())
        print(pid, "gt_npy: {}\noar_tumor_pred_npy: {}\npseudo_label_npy: {}".format(gt_unique, oar_tumor_pred_unique,
              pseudo_label_unique))
        if oar_tumor_pred_npy.shape != gt_npy.shape or ct_spacing != oar_tumor_pred_img.GetSpacing():
            print("!!!wrong pred img", pid)
            return
        if -1 not in ct_direction and -1 not in oar_tumor_pred_direction:
            if not isfile(new_ct):
                shutil.copy(ct, new_ct)

            oar_tumor_npy = gt_npy * 1
            oar_merged = False
            for l in pseudo_label_unique:
                if l == 0 or l == 14 or l in gt_unique:
                    continue
                oar_merged = True
                oar_tumor_npy[pseudo_label_npy == l] = l  # just merge pseudo oar
            if oar_merged:
                for l in gt_unique:
                    if l == 0:
                        continue
                    oar_tumor_npy[gt_npy == l] = l  # gt first
                print("merged oar: ", np.unique(oar_tumor_npy))

            if 14 not in gt_unique and 14 in oar_tumor_pred_unique:
                pseudo_tumor = oar_tumor_pred_npy * 1
                pseudo_tumor[oar_tumor_npy == 0] = 0  # remove pseudo oar and tumor according to new gt background
                oar_tumor_npy[pseudo_tumor == 14] = 14  # merge pseudo tumor
                print("merged tumor: ", np.unique(oar_tumor_npy))

            tumor_img = sitk.GetImageFromArray(oar_tumor_npy)
            tumor_img.CopyInformation(ct_img)
            sitk.WriteImage(tumor_img, new_label)
        else:
            ct_spacing = ct_img.GetSpacing()
            ct_origin = ct_img.GetOrigin()

            [new_ct_img, new_gt_img, new_oar_tumor_pred_img, new_pseudo_label_img] = sitk_correct_direction(
                [ct_img, gt_img, oar_tumor_pred_img, pseudo_label_img], "LPS")
            oar_tumor_pred_npy = sitk.GetArrayFromImage(new_oar_tumor_pred_img).astype(np.uint8)
            pseudo_label_npy = sitk.GetArrayFromImage(new_pseudo_label_img).astype(np.uint8)
            gt_npy = sitk.GetArrayFromImage(new_gt_img).astype(np.uint8)
            print(pid, "correct_direction: \nbefore: {} \nafter: {}".format(
                [ct_direction, ct_origin, ct_spacing],
                [new_ct_img.GetDirection(), new_ct_img.GetOrigin(), new_ct_img.GetSpacing()]))

            oar_tumor_npy = gt_npy * 1
            oar_merged = False
            for l in pseudo_label_unique:
                if l == 0 or l == 14 or l in gt_unique:
                    continue
                oar_merged = True
                oar_tumor_npy[pseudo_label_npy == l] = l  # just merge pseudo oar
            if oar_merged:
                for l in gt_unique:
                    if l == 0:
                        continue
                    oar_tumor_npy[gt_npy == l] = l  # gt first
                print("corrected merged oar: ", np.unique(oar_tumor_npy))

            if 14 not in gt_unique and 14 in oar_tumor_pred_unique:
                pseudo_tumor = oar_tumor_pred_npy * 1
                pseudo_tumor[oar_tumor_npy == 0] = 0  # remove pseudo oar and tumor according to new gt background
                oar_tumor_npy[pseudo_tumor == 14] = 14  # merge pseudo tumor
                print("corrected merged tumor: ", np.unique(oar_tumor_npy))

            new_gt_img = sitk.GetImageFromArray(oar_tumor_npy)
            new_gt_img.CopyInformation(new_ct_img)
            sitk.WriteImage(new_gt_img, new_label)

            if not isfile(new_ct):
                sitk.WriteImage(new_ct_img, new_ct)
        print(pid, "final oar_tumor_npy: ", np.unique(oar_tumor_npy))
    else:
        print("skip: ", pid)


def process1(pid, image_dir, oar_tumor_pred_dir, pseudo_label_dir, target_imagesTr, target_labelsTr):
    ct = join(image_dir, pid + "_0000.nii.gz")
    for img_sub_dir in ["unlabel3101-4000", "unlabelTr2201-3100"]:
        if isfile(ct):
            break
        else:
            ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
    pseudo_label = join(pseudo_label_dir, pid + ".nii.gz")      # without tumor label
    oar_tumor_pred = join(oar_tumor_pred_dir, pid + ".nii.gz")  # oar and tumor pred
    assert all([isfile(ct), isfile(oar_tumor_pred)]), "{} has wrong paths\n{}\n{}".format(pid, ct, oar_tumor_pred)

    new_ct = join(target_imagesTr, pid + "_0000.nii.gz")
    new_label = join(target_labelsTr, pid + ".nii.gz")
    if not (isfile(new_ct) and isfile(new_label)):
        # if True:
        print("processing: ", pid)
        ct_img = sitk.ReadImage(ct)
        ct_direction = ct_img.GetDirection()
        oar_tumor_pred_img = sitk.ReadImage(oar_tumor_pred)
        pseudo_label_img = sitk.ReadImage(pseudo_label)
        oar_tumor_pred_direction = oar_tumor_pred_img.GetDirection()

        oar_tumor_pred_npy = sitk.GetArrayFromImage(oar_tumor_pred_img).astype(np.uint8)
        pseudo_label_npy = sitk.GetArrayFromImage(pseudo_label_img).astype(np.uint8)
        oar_tumor_pred_unique = np.unique(oar_tumor_pred_npy)
        pseudo_label_unique = np.unique(pseudo_label_npy)
        print(pid, "ct: ", ct_direction, ct_img.GetOrigin(), ct_img.GetSpacing())
        # print(pid, ct_img.GetOrigin(), oar_tumor_pred_img.GetOrigin())
        print(pid, "oar_tumor_pred: {}\npseudo_label: {}".format(oar_tumor_pred_unique, pseudo_label_unique))
        if -1 not in ct_direction and -1 not in oar_tumor_pred_direction:
            if not isfile(new_ct):
                shutil.copy(ct, new_ct)

            oar_tumor_npy = pseudo_label_npy * 1    # use pseudo oar label
            pseudo_tumor = oar_tumor_pred_npy * 1
            pseudo_tumor[oar_tumor_npy == 0] = 0    # remove pseudo oar and tumor according to new oar background
            oar_tumor_npy[pseudo_tumor == 14] = 14  # relabel tumor label
            print("multi labeled tumor: ", np.unique(oar_tumor_npy))

            tumor_img = sitk.GetImageFromArray(oar_tumor_npy)
            tumor_img.CopyInformation(ct_img)
            sitk.WriteImage(tumor_img, new_label)
        else:
            [new_ct_img, new_oar_tumor_pred_img, new_pseudo_label_img] = sitk_correct_direction(
                [ct_img, oar_tumor_pred_img, pseudo_label_img], "LPS")
            oar_tumor_pred_npy = sitk.GetArrayFromImage(new_oar_tumor_pred_img).astype(np.uint8)
            pseudo_label_npy = sitk.GetArrayFromImage(new_pseudo_label_img).astype(np.uint8)

            oar_tumor_npy = pseudo_label_npy * 1  # use pseudo oar label
            pseudo_tumor = oar_tumor_pred_npy * 1
            pseudo_tumor[oar_tumor_npy == 0] = 0  # remove pseudo oar and tumor according to new oar background
            oar_tumor_npy[pseudo_tumor == 14] = 14  # relabel tumor label
            print("corrected multi labeled tumor: ", np.unique(oar_tumor_npy))

            new_gt_img = sitk.GetImageFromArray(oar_tumor_npy)
            new_gt_img.CopyInformation(new_ct_img)
            sitk.WriteImage(new_gt_img, new_label)
            if not isfile(new_ct):
                sitk.WriteImage(new_ct_img, new_ct)
        print(pid, "final oar_tumor_npy: ", np.unique(oar_tumor_npy))
    else:
        print("skip: ", pid)

if __name__ == "__main__":
    num_processes = 8

    task_name = "Task039_FLARE23noLabeledOARTumorMultiTask"
    data_dir = "/data/result/herongxuan/dataset/Release"
    label_dir = join(data_dir, "labelsTr2200")  # gt
    oar_tumor_pred_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/training_dataset/" \
                         "Task034__nnUNetPlansFLARE23TumorMedium__all_do_mirror_fullWindow_TTA4_0814"  # oar and tumor
    aladdin5_pseudo_label_dir = join(data_dir, "aladdin5-pseudo-labels-FLARE23")
    blackbean_pseudo_label_dir = join(data_dir, "blackbean-pseudo-labels-FLARE23")  # method without postprocess
    image_dir = join(data_dir, "imagesTr2200")
    image_dir1 = join(data_dir, "unlabelTr1800")
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

    take_labeled_data = False
    patient_names1 = sorted(glob.glob(os.path.join(image_dir, "*", "*.nii.gz")))
    patient_names1 = [os.path.basename(p).split("_0000.nii.gz")[0] for p in patient_names1][:]
    patient_names2 = sorted(glob.glob(os.path.join(image_dir1, "*", "*.nii.gz")))
    patient_names2 = [os.path.basename(p).split("_0000.nii.gz")[0] for p in patient_names2][:]
    total_names = patient_names1 + patient_names2
    print("total: ", len(total_names), total_names[:5], total_names[-5:])
    if take_labeled_data:
        # for partial labeled dataset 2200 cases
        patient_names = patient_names1
        print(len(patient_names), patient_names[:5])
        # for pat in patient_names[:10]:
        #     process(pat, image_dir, label_dir, oar_tumor_pred_dir, aladdin5_pseudo_label_dir, target_imagesTr, target_labelsTr)
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            results = []
            results.append(
                p.starmap_async(
                    process, zip(patient_names,
                                 [image_dir] * len(patient_names),
                                 [label_dir] * len(patient_names),
                                 [oar_tumor_pred_dir] * len(patient_names),
                                 [aladdin5_pseudo_label_dir] * len(patient_names),
                                 [target_imagesTr] * len(patient_names),
                                 [target_labelsTr] * len(patient_names))
                )
            )

            [i.get() for i in results]
    else:
        patient_names = patient_names2
        print(len(patient_names), patient_names[:5])

        # for pat in patient_names[:10]:
        #     process1(pat, image_dir1, oar_tumor_pred_dir, aladdin5_pseudo_label_dir, target_imagesTr, target_labelsTr)
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            results = []
            results.append(
                p.starmap_async(
                    process1, zip(patient_names,
                                  [image_dir1] * len(patient_names),
                                  [oar_tumor_pred_dir] * len(patient_names),
                                  [aladdin5_pseudo_label_dir] * len(patient_names),
                                  [target_imagesTr] * len(patient_names),
                                  [target_labelsTr] * len(patient_names))
                )
            )

            [i.get() for i in results]

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
    patient_names = total_names
    json_dict["numTraining"] = len(patient_names)
    json_dict["numTest"] = 0
    json_dict["training"] = [{"image": "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i}
                             for i in patient_names]
    json_dict["test"] = []

    save_json(json_dict, os.path.join(target_base, "dataset.json"))
