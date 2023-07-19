import glob
import os
import shutil
import SimpleITK as sitk
import numpy as np
import pandas as pd

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.evaluation.metrics import dice


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
        re1_npy[re2_npy == 1] = 14
        sitk.WriteImage(sitk.GetImageFromArray(re1_npy), os.path.join(result_dir, pid + ".nii.gz"))

        # break


def compare_nnUNet_outputs(r1_dir, r2_dir, folds=None):
    if folds is None:
        folds = []
    pid_list = sorted(glob.glob(os.path.join(r1_dir, "*.nii.gz")))
    pid_list = [os.path.basename(p).split(".nii.gz")[0] for p in pid_list]
    print(len(pid_list), pid_list[:5])

    re1_zero_num, re2_zero_num = 0, 0
    re2_fold_zero_num = [0] * len(folds)
    for pid in pid_list:
        print(pid)
        re1 = os.path.join(r1_dir, pid + ".nii.gz")
        re2 = os.path.join(r2_dir, pid + ".nii.gz")
        re1_img = sitk.ReadImage(re1)
        re1_npy = sitk.GetArrayFromImage(re1_img)
        re2_img = sitk.ReadImage(re2)
        re2_npy = sitk.GetArrayFromImage(re2_img)

        assert re1_npy.shape == re2_npy.shape, f"{pid} shape dont match"

        re1_num, re2_num = np.sum(re1_npy), np.sum(re2_npy)
        if re1_num != 0:
            re1_zero_num += 1
        if re2_num != 0:
            re2_zero_num += 1
        dsc = dice(re1_npy, re2_npy)
        print(pid, re1_num, re2_num, round(dsc, 2))

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
    without_tumor_pid_table_path = "/home/zhongzhiqiang/PreResearch/FLARE2023/tables/labelsTr703_withoutTumor.xlsx"

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "pseudoLabelsTr")
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)

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
        shutil.copy(ct, os.path.join(target_imagesTr, pid + "_0000.nii.gz"))
    print("copy label image done.")

    # copy 1800 unlabeled images
    for pid in unlabeled_pid_lst:
        info_dict[pid] = {}
        ct = join(unlabeled_image_dir, pid + "_0000.nii.gz")
        for img_sub_dir in ["unlabel3101-4000", "unlabelTr2201-3100"]:
            if isfile(ct):
                break
            else:
                ct = join(image_dir, img_sub_dir, pid + "_0000.nii.gz")
        info_dict[pid]["ori_ct_path"] = ct
        shutil.copy(ct, os.path.join(target_imagesTr, pid + "_0000.nii.gz"))
    print("copy unlabeled image done.")

    df = pd.DataFrame.from_dict(info_dict, orient="index").rename(columns={0: "PID"})
    df.to_excel(os.path.join(target_base, info_table_save_name))
    print(df)


if __name__ == "__main__":

    # base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_preprocessed/"
    # task_name = "Task023_FLARE23Tumor"
    # plan_name = "nnUNetPlansFLARE23TumorSmall_plans_3D.pkl"
    # modify_batch_size_in_plan(join(base_dir, task_name, plan_name), batch_size=2)


    # nnUNetPlansFLARE23TumorSmall__5folds_noMirror_0713
    # nnUNetPlansFLARE23TumorSmall__all_do_mirror_0713
    # nnUNetPlansFLARE23TumorSmall__5folds_do_mirror_0713
    # compare_nnUNet_outputs(
    #     r1_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/nnUNetPlansFLARE23TumorSmall__all_do_mirror_0713",
    #     r2_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/Task025__nnUNetPlansFLARE23TumorSmall__all_do_mirror_0718",
    #     # folds=[i for i in range(5)]
    # )

    # compute_uncertainty_of_pseudo_label_for_5_folds(
    #     base_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/nnUNetPlansFLARE23TumorSmall__5folds_do_mirror_0713",
    #     folds_postfix=[f"_fold{i}" for i in range(5)]
    # )

    # create_tumor_with_OAR_test_dataset(
    #     image_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task023_FLARE23Tumor/imagesVal",
    #     oar_result_dir="/home/zhongzhiqiang/result/nnUNet_small_mirroring_0713_postprocessing",
    #     output_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_inputs/imagesVal100_withOAR"
    # )

    # merge_oar_tumor_label(
    #     oar_re_dir="/home/zhongzhiqiang/result/nnUNet_small_mirroring_0713_postprocessing",
    #     tumor_re_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/Task025__nnUNetPlansFLARE23TumorSmall__all_do_mirror_0718",
    #     result_dir="/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/merge"
    # )

    # write_modality_info_to_pkl(
    #     "/data/result/zhongzhiqiang/nnUNet/nnUNet_preprocessed/Task025_FLARE23TumorWithOAR/nnUNetData_plans_FLARE23TumorSmall_stage0"
    # )

    copy_images_with_unlabeled_tumor()


