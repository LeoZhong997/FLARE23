#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from batchgenerators.utilities.file_and_folder_operations import load_pickle, join
from nnunet.training.network_training.nnUNetTrainerV3 import nnUNetTrainerV2_FLARE_Medium_Pure


def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info['init']
    print("trainer init:", init)

    trainer = nnUNetTrainerV2_FLARE_Medium_Pure(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_model_and_checkpoint_files(folder, folds=None, mixed_precision=None, checkpoint_name="model_best",
                                    trt_mode=False):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them from disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    folds = [join(folder, "all")]

    trainer = restore_model(join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    if trt_mode:
        trt_path = [join(i, "%s.trt" % checkpoint_name) for i in folds]
        trainer.initialize(False, trt_mode=trt_mode, trt_path=trt_path[0])
        print("using the following TensorRT files: ", trt_path)
        all_params = trt_path
    else:
        trainer.initialize(False)
        all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
        print("using the following model files: ", all_best_model_files)
        all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params


if __name__ == "__main__":
    pkl = "/home/fabian/PhD/results/nnUNetV2/nnUNetV2_3D_fullres/Task004_Hippocampus/fold0/model_best.model.pkl"
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
