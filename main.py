import glob
import os
import shutil
import SimpleITK as sitk
import numpy as np
import pandas as pd

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.evaluation.metrics import dice
from nnunet.network_architecture.neural_network import SegmentationNetwork
from utils import small_volume_filter_2d

cls_oar = ["Background", "Liver", "Right kidney", "Spleen", "Pancreas", "Aorta", "Inferior vena cava",
           "Right adrenal gland", "Left adrenal gland", "Gallbladder", "Esophagus", "Stomach",
           "Duodenum", "Left Kidney", "Tumor"]


def write_modality_info_to_pkl(output_dir):
    pid_pkl_list = sorted(glob.glob(os.path.join(output_dir, "*.pkl")))
    print(len(pid_pkl_list), pid_pkl_list[:5])

    all_modalities = {0: "CT", 1: "seg"}
    for pid_pkl in pid_pkl_list:
        pkl_info = load_pickle(pid_pkl)
        pkl_info['all_modalities'] = all_modalities

        write_pickle(pkl_info, pid_pkl)
        # break


def merge_oar_tumor_label(oar_re_dir, tumor_re_dir, result_dir):
    name = tumor_re_dir.split("/")[-1]
    result_dir = os.path.join(result_dir, name)
    maybe_mkdir_p(result_dir)

    pid_list = sorted(glob.glob(os.path.join(tumor_re_dir, "*.nii.gz")))
    pid_list = [os.path.basename(p).split(".nii.gz")[0] for p in pid_list]
    print(len(pid_list), pid_list[:5])

    for pid in pid_list:
        print(pid)
        re1 = os.path.join(oar_re_dir, pid + ".nii.gz")
        re2 = os.path.join(tumor_re_dir, pid + ".nii.gz")
        re1_img = sitk.ReadImage(re1)
        re1_npy = sitk.GetArrayFromImage(re1_img)
        re2_img = sitk.ReadImage(re2)
        re2_npy = sitk.GetArrayFromImage(re2_img)

        print(np.unique(re1_npy), np.unique(re2_npy))
        re1_npy[re2_npy > 0] = 14
        print("merged: ", np.unique(re1_npy))
        merge_img = sitk.GetImageFromArray(re1_npy)
        merge_img.CopyInformation(re1_img)
        sitk.WriteImage(merge_img, os.path.join(result_dir, pid + ".nii.gz"))

        # break


def compare_nnUNet_outputs(r1_dir, r2_dir, folds=None, save_table=False, rename=None,
                           match_img_label=False, img_dir=None, save_dir=None, pid_list=None,
                           class_id=None):
    if rename is None:
        rename = ["re1", "re2"]
    if folds is None:
        folds = []
    if class_id is None:
        class_id = [1]
    if match_img_label and img_dir is None:
        # this img folder copied from ori image directly
        img_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE23Tumor/imagesTr"
    if save_dir is None:
        save_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/training_dataset/compare_result/" + \
                   os.path.basename(r1_dir)
    maybe_mkdir_p(save_dir)
    table_path = os.path.join(save_dir, "dice_compare.xlsx")
    if pid_list is None:
        pid_list = sorted(glob.glob(os.path.join(r1_dir, "*.nii.gz")))
        pid_list = [os.path.basename(p).split(".nii.gz")[0] for p in pid_list]
        pid_list = pid_list[:]
    print(len(pid_list), pid_list[:5])

    re1_zero_num, re2_zero_num = [0] * len(class_id), [0] * len(class_id)
    re2_fold_zero_num = [0] * len(folds)
    pid_metric_dict = {}
    for pid in pid_list:
        print(pid)
        re1 = os.path.join(r1_dir, pid + ".nii.gz")
        re2 = os.path.join(r2_dir, pid + ".nii.gz")
        re1_img = sitk.ReadImage(re1)
        re1_npy = sitk.GetArrayFromImage(re1_img).astype(np.uint8)
        re2_img = sitk.ReadImage(re2)
        re2_npy = sitk.GetArrayFromImage(re2_img).astype(np.uint8)

        assert re1_npy.shape == re2_npy.shape, f"{pid} shape dont match"

        metric = {}
        for i, c in enumerate(class_id):
            re1_npy_c = re1_npy * 0
            re1_npy_c[re1_npy == c] = 1
            re2_npy_c = re2_npy * 0
            re2_npy_c[re2_npy == c] = 1
            re1_num, re2_num = np.sum(re1_npy_c), np.sum(re2_npy_c)
            if re1_num != 0:
                re1_zero_num[i] += 1
            if re2_num != 0:
                re2_zero_num[i] += 1
            dsc = dice(re1_npy_c, re2_npy_c)
            dsc = round(dsc, 4)
            print(pid, i, c, re1_num, re2_num, dsc)
            metric[rename[0] + f"_pixel{c}"] = re1_num
            metric[rename[1] + f"_pixel{c}"] = re2_num
            metric[f"dice{c}"] = dsc
        pid_metric_dict[pid] = metric

        if match_img_label:
            img = sitk.ReadImage(os.path.join(img_dir, pid + "_0000.nii.gz"))
            re1_img_ = sitk.GetImageFromArray(re1_npy)
            re1_img_.CopyInformation(img)
            sitk.WriteImage(re1_img_, os.path.join(save_dir, pid + "_{}.nii.gz".format(rename[0])))
            re2_npy[re2_npy == 1] = 14
            re2_img_ = sitk.GetImageFromArray(re2_npy)
            re2_img_.CopyInformation(img)
            sitk.WriteImage(re2_img_, os.path.join(save_dir, pid + "_{}.nii.gz".format(rename[1])))

        for i, f in enumerate(folds):
            re1 = os.path.join(r1_dir, pid + ".nii.gz")
            re2 = os.path.join(r2_dir + f"_fold{f}", pid + ".nii.gz")
            re1_img = sitk.ReadImage(re1)
            re1_npy = sitk.GetArrayFromImage(re1_img)
            re2_img = sitk.ReadImage(re2)
            re2_npy = sitk.GetArrayFromImage(re2_img)

            assert re1_npy.shape == re2_npy.shape, f"{pid} shape dont match"

            re1_num, re2_num = np.sum(re1_npy), np.sum(re2_npy)
            if re2_num != 0:
                re2_fold_zero_num[i] += 1
            dsc = dice(re1_npy, re2_npy)
            print(f, pid, re1_num, re2_num, round(dsc, 2))
    print("total: ", re1_zero_num, re2_zero_num, re2_fold_zero_num)
    print("metric: ", pid_metric_dict)
    if save_table:
        df = pd.DataFrame.from_dict(pid_metric_dict, orient="index")
        df.to_excel(table_path)


def compare_val_results(r1_dir, r2_dir, save_table=False, rename=None,
                        match_img_label=False, img_dir=None, save_dir=None, pid_list=None,
                        class_id=None):
    if rename is None:
        rename = ["re1", "re2"]
    if class_id is None:
        class_id = [i for i in range(1, 15)]
    if match_img_label and img_dir is None:
        # this img folder copied from ori image directly
        img_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE23Tumor/imagesVal"
    if save_dir is None:
        save_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/merge/compare_result/" + \
                   os.path.basename(r1_dir) + f"-{rename[1]}"
    maybe_mkdir_p(save_dir)
    table_path = os.path.join(save_dir, "dice_compare.xlsx")
    if pid_list is None:
        pid_list = sorted(glob.glob(os.path.join(r1_dir, "*.nii.gz")))
        pid_list = [os.path.basename(p).split(".nii.gz")[0] for p in pid_list]
        pid_list = pid_list[:]
    print(len(pid_list), pid_list[:5])

    re1_zero_num, re2_zero_num = [0] * len(class_id), [0] * len(class_id)
    pid_metric_dict = {}
    for i, pid in enumerate(pid_list):
        print(i, pid)
        re1 = os.path.join(r1_dir, pid + ".nii.gz")
        re2 = os.path.join(r2_dir, pid + ".nii.gz")
        re1_img = sitk.ReadImage(re1)
        re1_npy = sitk.GetArrayFromImage(re1_img).astype(np.uint8)
        re2_img = sitk.ReadImage(re2)
        re2_npy = sitk.GetArrayFromImage(re2_img).astype(np.uint8)

        # just check tumor
        re2_npy[re2_npy > 0] = 14
        re1_npy[re1_npy != 14] = 0

        assert re1_npy.shape == re2_npy.shape, f"{pid} shape dont match"

        metric = {}
        for i, c in enumerate(class_id):
            oar = cls_oar[c]
            re1_npy_c = re1_npy * 0
            re1_npy_c[re1_npy == c] = 1
            re2_npy_c = re2_npy * 0
            re2_npy_c[re2_npy == c] = 1
            re1_num, re2_num = np.sum(re1_npy_c), np.sum(re2_npy_c)
            if re1_num != 0:
                re1_zero_num[i] += 1
            if re2_num != 0:
                re2_zero_num[i] += 1
            dsc = dice(re1_npy_c, re2_npy_c)
            dsc = round(dsc, 4)
            print(pid, c, oar, re1_num, re2_num, dsc)
            metric[f"{oar}_pix-" + rename[0]] = re1_num
            metric[f"{oar}_pix-" + rename[1]] = re2_num
            metric[f"{oar}_dice"] = dsc
        pid_metric_dict[pid] = metric
        print(pid, metric)

        if match_img_label:
            img = sitk.ReadImage(os.path.join(img_dir, pid + "_0000.nii.gz"))
            re1_img_ = sitk.GetImageFromArray(re1_npy)
            re1_img_.CopyInformation(img)
            sitk.WriteImage(re1_img_, os.path.join(save_dir, pid + "_{}.nii.gz".format(rename[0])))
            re2_img_ = sitk.GetImageFromArray(re2_npy)
            re2_img_.CopyInformation(img)
            sitk.WriteImage(re2_img_, os.path.join(save_dir, pid + "_{}.nii.gz".format(rename[1])))
    print("total: ", re1_zero_num, re2_zero_num)
    print("metric: ", pid_metric_dict)
    if save_table:
        df = pd.DataFrame.from_dict(pid_metric_dict, orient="index")
        df.to_excel(table_path)


def modify_batch_size_in_plan(plan_file: str, batch_size: int, back_up=True):
    plan_conf = load_pickle(plan_file)
    print(plan_conf['plans_per_stage'][1]['batch_size'])

    # if back_up:
    #     shutil.copy(plan_file, plan_file.replace('.pkl', '_bk.pkl'))
    #
    # for i in [0, 1]:
    #     plan_conf['plans_per_stage'][i]['batch_size'] = batch_size
    # write_pickle(plan_conf, plan_file)


def create_tumor_with_OAR_test_dataset(image_dir, oar_result_dir, output_dir):
    pid_list = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    pid_list = [os.path.basename(p).split(".nii.gz")[0][:-5] for p in pid_list]
    print(len(pid_list), pid_list[:5])
    maybe_mkdir_p(output_dir)

    for pid in pid_list:
        src_path = os.path.join(image_dir, pid + "_0000.nii.gz")
        target_path = os.path.join(output_dir, pid + "_0000.nii.gz")
        shutil.copy(src_path, target_path)

        src_path = os.path.join(oar_result_dir, pid + ".nii.gz")
        target_path = os.path.join(output_dir, pid + "_0001.nii.gz")
        shutil.copy(src_path, target_path)


def compute_uncertainty_of_pseudo_label_for_5_folds(base_dir, folds_postfix=None):
    if folds_postfix is None:
        folds_postfix = [f"_fold{i}" for i in range(5)]
    pid_list = sorted(glob.glob(os.path.join(base_dir + "_fold0", "*.nii.gz")))
    pid_list = [os.path.basename(p).split(".nii.gz")[0] for p in pid_list]
    print(len(pid_list), pid_list[:5])

    K = len(folds_postfix)
    uncertainties_dict = {}
    dice_dict = {}
    for p, pid in enumerate(pid_list):
        print("process: ", p, pid)
        uncertainties = []
        dice_list = []
        for f, fold_postfix in enumerate(folds_postfix):
            if f == 0:
                continue
            src_fold_dir = os.path.join(base_dir + folds_postfix[f])
            tgt_fold_dir = os.path.join(base_dir + folds_postfix[f - 1])
            src_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(src_fold_dir, pid + ".nii.gz")))
            tgt_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(tgt_fold_dir, pid + ".nii.gz")))
            unequal_mask = src_mask != tgt_mask
            if np.sum(src_mask) != 0:
                uncertainties.append(round(np.sum(unequal_mask) / np.sum(src_mask), 4))
            elif np.sum(tgt_mask) != 0:
                uncertainties.append(1)
            else:
                uncertainties.append(0)
            dsc = round(dice(src_mask, tgt_mask), 4)
            dice_list.append(dsc)
            # print(f, fold_postfix, re[-1])
            # print("fold: ", f, fold_postfix,
            #       # "\nunique:", np.unique(src_mask), np.unique(tgt_mask), np.unique(unequal_mask),
            #       "\nsum:", np.sum(src_mask), np.sum(tgt_mask), np.sum(unequal_mask),
            #       "\nuncertainty:", uncertainties[-1],
            #       )
        if len(uncertainties) > 0:
            avg = round(np.average(uncertainties), 4)
            print("uncertainties: ", pid, uncertainties, avg)
            print("dice_list: ", pid, dice_list)
            uncertainties_dict[pid] = avg
            dice_dict[pid] = dice_list
    print(uncertainties_dict)
    print(dice_dict)


def copy_images_with_unlabeled_tumor():
    target_base = "/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesUnlabeledTumor"
    info_table_save_name = "pid_path_info.xlsx"
    data_dir = "/data/result/herongxuan/dataset/Release"
    label_dir = join(data_dir, "labelsTr2200")
    image_dir = join(data_dir, "imagesTr2200")
    unlabeled_image_dir = join(data_dir, "unlabelTr1800")
    val_image_dir = join(data_dir, "validation")
    without_tumor_pid_table_path = "/home/zhongzhiqiang/Dataset/FLARE2023/labelsTr703_withoutTumorLabel.xlsx"

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "pseudoLabelsTr")
    target_OARlabelsTr = join(target_base, "oarLabelsTr")
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_OARlabelsTr)

    df = pd.read_excel(without_tumor_pid_table_path)
    pid_lst = df["case_id"].values.tolist()[:]
    # patient_names = ["FLARE23_0609"]
    print(len(pid_lst), pid_lst[:5])

    unlabeled_pid_lst = sorted(glob.glob(os.path.join(unlabeled_image_dir, "*", "*_0000.nii.gz")))
    unlabeled_pid_lst = [os.path.basename(pid).split("_0000.nii.gz")[0] for pid in unlabeled_pid_lst]
    print(len(unlabeled_pid_lst), unlabeled_pid_lst[:5])

    info_dict = {}

    # copy image without tumor from label dataset with 2200 cases
    for pid in pid_lst:
        info_dict[pid] = {}
        ct = join(image_dir, pid + "_0000.nii.gz")
        for img_sub_dir in ["1-500", "501-1000", "1001-1500", "1501-2000"]:
            if isfile(ct):
                break
            else:
                ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
        info_dict[pid]["oar_ct_path"] = ct
        if not os.path.exists(os.path.join(target_imagesTr, pid + "_0000.nii.gz")):
            shutil.copy(ct, os.path.join(target_imagesTr, pid + "_0000.nii.gz"))

        if not os.path.exists(os.path.join(target_OARlabelsTr, pid + ".nii.gz")):
            oar_label = join(label_dir, pid + ".nii.gz")
            ct_img = sitk.ReadImage(ct)
            oar_label_arr = sitk.GetArrayFromImage(sitk.ReadImage(oar_label))
            new_oar_label_img = sitk.GetImageFromArray(oar_label_arr)
            new_oar_label_img.CopyInformation(ct_img)
            sitk.WriteImage(new_oar_label_img, os.path.join(target_OARlabelsTr, pid + ".nii.gz"))
        print(pid, ct)
    print("copy OAR labeled image done.")

    # copy 1800 unlabeled images
    # for pid in unlabeled_pid_lst:
    #     if os.path.exists(os.path.join(target_imagesTr, pid + "_0000.nii.gz")):
    #         continue
    #     info_dict[pid] = {}
    #     ct = join(unlabeled_image_dir, pid + "_0000.nii.gz")
    #     for img_sub_dir in ["unlabel3101-4000", "unlabelTr2201-3100"]:
    #         if isfile(ct):
    #             break
    #         else:
    #             ct = join(unlabeled_image_dir, img_sub_dir, pid + "_0000.nii.gz")
    #     info_dict[pid]["ori_ct_path"] = ct
    #     shutil.copy(ct, os.path.join(target_imagesTr, pid + "_0000.nii.gz"))
    #     print(pid, ct)
    # print("copy unlabeled image done.")
    #
    # df = pd.DataFrame.from_dict(info_dict, orient="index").rename(columns={0: "PID"})
    # df.to_excel(os.path.join(target_base, info_table_save_name))
    # print(df)


def copy_checkpoint_from_temp_to_final():
    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/" \
               "Task028_FLARE23TumorMultiLabelWithOAR/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium"
    # folds = [f"fold_{i}" for i in range(5)]
    folds = ["all"]
    overwrite = True
    for f in folds:
        src_final_path = os.path.join(base_dir, f, "model_latest.model")
        src_final_pkl_path = os.path.join(base_dir, f, "model_latest.model.pkl")
        final_path = os.path.join(base_dir, f, "model_final_checkpoint.model")
        final_pkl_path = os.path.join(base_dir, f, "model_final_checkpoint.model.pkl")

        if not os.path.exists(final_path) or overwrite:
            shutil.copy(src_final_path, final_path)
            shutil.copy(src_final_pkl_path, final_pkl_path)


def tumor_multi_label_transpose_single_label():
    multi_label_result = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/" \
                         "Task026__nnUNetPlansFLARE23TumorMedium__all_do_mirror_0724"
    single_label_result = multi_label_result + "_single_label"
    maybe_mkdir_p(single_label_result)

    pid_list = sorted(glob.glob(os.path.join(multi_label_result, "*.nii.gz")))
    pid_list = [os.path.basename(p).split(".nii.gz")[0] for p in pid_list]
    print(len(pid_list), pid_list[:5])

    for i, pid in enumerate(pid_list):
        print("processing: ", i, pid)
        multi_label_img = sitk.ReadImage(os.path.join(multi_label_result, pid + ".nii.gz"))
        multi_label = sitk.GetArrayFromImage(multi_label_img)

        single_label = multi_label * 1
        single_label[single_label > 0] = 1
        single_label_img = sitk.GetImageFromArray(single_label)
        single_label_img.CopyInformation(multi_label_img)
        sitk.WriteImage(single_label_img, os.path.join(single_label_result, pid + ".nii.gz"))

        # break


def compute_slide_window_mask():
    temp_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/predict_temp/slide_window_test"
    table_name = "label_slide_window_info_0_8.xlsx"
    data_dir = "/data/result/herongxuan/dataset/Release"
    label_dir = join(data_dir, "labelsTr2200")
    cropped_label_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_preprocessed/" \
                        "Task026_FLARE23TumorMultiLabel/nnUNetData_plans_FLARE23TumorMedium_stage0"     # tumor multi label
    image_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE23Tumor/imagesTr"
    maybe_mkdir_p(temp_dir)

    pid_path = "/home/zhongzhiqiang/result/predict_temp/slide_window_test/label_slide_window_info_check.xlsx"
    df = pd.read_excel(pid_path, sheet_name="check")
    pid_list = df.pid.values.tolist()
    # pid_list = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    # pid_list = [os.path.basename(p).split("_0000.nii.gz")[0] for p in pid_list][:]
    print(len(pid_list), pid_list[:5])

    patch_size = (32, 128, 192)
    # patch_size = (32, 192, 192)
    img_size = (120, 520, 520)
    step_size = 0.5

    total_info = {}
    patch_mask = np.ones(patch_size, dtype=np.uint8)
    for i, pid in enumerate(pid_list):
        print(i, pid)
        # label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, pid + ".nii.gz"))).astype(np.uint8)
        label = np.load(os.path.join(cropped_label_dir, pid + ".npy"))[-1].astype(np.uint8)
        _z, _y, _x = np.where(label > 0)
        bbox = [min(_z), max(_z), min(_y), max(_y), min(_x), max(_x)]
        img_size = label.shape

        steps = SegmentationNetwork._compute_steps_for_sliding_window(patch_size, img_size, step_size)  # [z, y, x]
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])
        pid_info = {
            "patch_size": patch_size,
            "step_size": step_size,
            "img_size": img_size,
            "label_bbox": bbox,
            "window_bbox": [],
            "num_tiles": num_tiles,
            "steps": steps,
        }
        print("pid_info: ", pid_info)

        window_mask = np.zeros(img_size, dtype=np.uint8)
        for z in steps[0]:
            lb_z = z
            ub_z = min(z + patch_size[0], img_size[0])
            for y in steps[1]:
                lb_y = y
                ub_y = min(y + patch_size[1], img_size[1])
                for x in steps[2]:
                    lb_x = x
                    ub_x = min(x + patch_size[2], img_size[2])
                    window_mask[lb_z:ub_z, lb_y:ub_y, lb_x:ub_x] += patch_mask[0:ub_z-lb_z, 0:ub_y-lb_y, 0:ub_x-lb_x]
                    # print(">>>Debug: ", lb_z, ub_z, lb_y, ub_y, lb_x, ub_x)

        # print(np.max(window_mask), np.sum(window_mask == 0), np.sum(window_mask > 0))
        _z, _y, _x = np.where(window_mask > 0)
        window_bbox = [min(_z), max(_z), min(_y), max(_y), min(_x), max(_x)]
        pid_info["window_bbox"] = window_bbox
        print("window_bbox:", pid_info["window_bbox"])

        label_outer = label * 1
        label_outer[window_mask > 0] = 0
        pid_info["label_before"] = np.unique(label)
        pid_info["label_after"] = np.unique(label_outer)
        print("label before and after:", pid_info["label_before"], pid_info["label_after"])

        pid_info["label_full"] = True
        if len(np.unique(label_outer)) > 1:
            pid_info["label_full"] = False
            sitk.WriteImage(sitk.GetImageFromArray(window_mask), os.path.join(temp_dir, pid + "_window_mask.nii.gz"))
            sitk.WriteImage(sitk.GetImageFromArray(label), os.path.join(temp_dir, pid + "_label.nii.gz"))
            print("!!!check label and window mask")

        total_info[pid] = pid_info
        # break

    df = pd.DataFrame.from_dict(total_info, orient="index")
    df.to_excel(os.path.join(temp_dir, table_name))


def create_pseudo_tumor_use_oar_label():
    img_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesUnlabeledTumor/imagesTr"
    oar_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesUnlabeledTumor/oarLabelsTr"
    tumor_result_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesUnlabeledTumor/" \
                       "Task026_all_do_mirror_fullWindow_closeTTA_0727"
    tumor_final_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesUnlabeledTumor/pseudoLabelsTr"
    table_save_path = os.path.join("/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesUnlabeledTumor",
                                   "pseudo_tumor_info-Task026_all_do_mirror_fullWindow_closeTTA_0727.xlsx")

    pid_list = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
    pid_list = [os.path.basename(p).split("_0000.nii.gz")[0] for p in pid_list][:]
    print(len(pid_list), pid_list[:5])

    total_info = {}
    for i, pid in enumerate(pid_list):
        img_path = os.path.join(img_dir, pid + "_0000.nii.gz")
        oar_path = os.path.join(oar_dir, pid + ".nii.gz")
        tumor_path = os.path.join(tumor_result_dir, pid + ".nii.gz")
        save_path = os.path.join(tumor_final_dir, pid + ".nii.gz")
        if not (os.path.exists(oar_path) and os.path.join(tumor_path)):
            print("skip: ", i, pid)
            continue

        pid_info = {}
        img = sitk.ReadImage(img_path)
        img_size = img.GetSize()
        img_spacing = img.GetSpacing()
        plane_res = img_spacing[0] * img_spacing[1]
        oar_npy = sitk.GetArrayFromImage(sitk.ReadImage(oar_path)).astype(np.uint8)
        tumor_npy = sitk.GetArrayFromImage(sitk.ReadImage(tumor_path)).astype(np.uint8)
        oar_unique = np.unique(oar_npy)
        tumor_unique = np.unique(tumor_npy)

        new_tumor_npy = tumor_npy * 1
        new_tumor_npy[oar_npy == 0] = 0
        new_tumor_npy[new_tumor_npy == 14] = 0

        tumor_sum = []
        for t in tumor_unique:
            tumor_sum.append(np.sum(tumor_npy == t))

        oar_sum = []
        for o in oar_unique:
            oar_sum.append(np.sum(oar_npy == o))
            if o == 0:
                continue
            new_tumor_npy[(oar_npy == o) & (new_tumor_npy != 0)] = o
        new_tumor_unique = np.unique(new_tumor_npy)
        new_tumor_sum = []
        for t in new_tumor_unique:
            new_tumor_sum.append(np.sum(new_tumor_npy == t))

        # postprocess for new tumor
        vol_threshold = np.ceil(10 / plane_res)
        pp_new_tumor_npy = small_volume_filter_2d(new_tumor_npy, threshold=vol_threshold)
        pp_new_tumor_unique = np.unique(pp_new_tumor_npy)
        pp_new_tumor_sum = []
        for t in pp_new_tumor_unique:
            pp_new_tumor_sum.append(np.sum(pp_new_tumor_npy == t))

        pid_info["size"] = img_size
        pid_info["spacing"] = img_spacing
        pid_info["vol_threshold"] = vol_threshold
        pid_info["oar"] = oar_unique.tolist()
        pid_info["oar_sum"] = oar_sum
        pid_info["tumor"] = tumor_unique.tolist()
        pid_info["tumor_sum"] = tumor_sum
        pid_info["new_tumor"] = new_tumor_unique.tolist()
        pid_info["new_tumor_sum"] = new_tumor_sum
        pid_info["pp_new_tumor"] = pp_new_tumor_unique.tolist()
        pid_info["pp_new_tumor_sum"] = pp_new_tumor_sum
        total_info[pid] = pid_info
        print(">>>processing: ", i, pid, pid_info)

        new_tumor_img = sitk.GetImageFromArray(pp_new_tumor_npy)
        new_tumor_img.CopyInformation(img)
        sitk.WriteImage(new_tumor_img, save_path)

    df = pd.DataFrame.from_dict(total_info, orient="index")
    df.to_excel(table_save_path)


if __name__ == "__main__":
    # base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_preprocessed/"
    # task_name = "Task023_FLARE23Tumor"
    # plan_name = "nnUNetPlansFLARE23TumorSmall_plans_3D.pkl"
    # modify_batch_size_in_plan(join(base_dir, task_name, plan_name), batch_size=2)

    # nnUNetPlansFLARE23TumorSmall__5folds_noMirror_0713
    # nnUNetPlansFLARE23TumorSmall__all_do_mirror_0713
    # nnUNetPlansFLARE23TumorSmall__5folds_do_mirror_0713
    # compare_nnUNet_outputs(
    #     r1_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/Task026__nnUNetPlansFLARE23TumorMedium__all_do_mirror_0724",
    #     r2_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/Task026__nnUNetPlansFLARE23TumorMedium__all_do_mirror_fullWindow_closeTTA_0724",
    #     # folds=[i for i in range(5)]
    # )
    # compare_nnUNet_outputs(
    #     r1_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/training_dataset/nnUNetPlansFLARE23TumorSmall__5folds_do_mirror_0713_full_window",
    #     r2_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE23Tumor/labelsTr",
    #     save_table=True, rename=["pred", "gt"], match_img_label=True
    #     # folds=[i for i in range(5)]
    # )
    # compare_nnUNet_outputs(
    #     r1_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/merge/nnUNetPlansFLARE23TumorSmall__5folds_do_mirror_0713",
    #     r2_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/gt-validation-for-sanity-check",
    #     img_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE23Tumor/imagesVal",
    #     save_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/nnUNetPlansFLARE23TumorSmall__5folds_do_mirror_0713",
    #     save_table=True, rename=["pred", "gt"], match_img_label=True, class_id=[i for i in range(1, 15)],
    #     pid_list=["FLARE23Ts_0038", "FLARE23Ts_0048", "FLARE23Ts_0083"]
    # )

    # compare_val_results(
    #     r1_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/merge/Task026__nnUNetPlansFLARE23TumorMedium__all_do_mirror_0724_single_label",
    #     r2_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/merge/Task026__nnUNetPlansFLARE23TumorMedium__all_do_mirror_fullWindow_0724",
    #     save_table=True, rename=["fullWindow_TTA", "fullWindow_TTA_again"], match_img_label=True
    # )
    # compare_val_results(
    #     r1_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/gt-validation-for-sanity-check/ground_true",
    #     r2_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/gt-validation-for-sanity-check/Task028__nnUNetPlansFLARE23TumorMedium__all_do_mirror_0727",
    #     save_table=True, rename=["GT", "Task028__nnUNetPlansFLARE23TumorMedium__all_do_mirror_0727"],
    #     match_img_label=True, class_id=[14]
    # )

    # compute_uncertainty_of_pseudo_label_for_5_folds(
    #     base_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/nnUNetPlansFLARE23TumorSmall__5folds_do_mirror_0713",
    #     folds_postfix=[f"_fold{i}" for i in range(5)]
    # )

    # create_tumor_with_OAR_test_dataset(
    #     image_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE23Tumor/imagesVal",
    #     # oar_result_dir="/home/zhongzhiqiang/result/nnUNet_small_mirroring_0713_postprocessing",
    #     oar_result_dir="/home/zhongzhiqiang/result/Mixed_mirroring_postprocess_0_5",
    #     output_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesVal100_withOAR-Mixed_mirroring_postprocess_0_5"
    # )

    # merge_oar_tumor_label(
    #     # oar_re_dir="/home/zhongzhiqiang/result/nnUNet_small_mirroring_0713_postprocessing",
    #     oar_re_dir="/home/zhongzhiqiang/result/Mixed_mirroring_postprocess_0_5",
    #     tumor_re_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/val_dataset/Task028__nnUNetPlansFLARE23TumorMedium__all_do_mirror_fullWindow_0728",
    #     result_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/merge"
    # )

    # write_modality_info_to_pkl(
    #     "/data/result/zhongzhiqiang/nnUNet/nnUNet_preprocessed/Task025_FLARE23TumorWithOAR/nnUNetData_plans_FLARE23TumorSmall_stage0"
    # )

    # copy_images_with_unlabeled_tumor()

    # copy_checkpoint_from_temp_to_final()

    # tumor_multi_label_transpose_single_label()

    # create_pseudo_tumor_use_oar_label()

    # compute_slide_window_mask()

    print("hello")
    

