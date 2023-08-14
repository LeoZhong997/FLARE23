import time

import torch

from nnunet.training.model_restore_pure import load_model_and_checkpoint_files
from nnunet.utilities.random_stuff import no_op

from torch.cuda.amp import autocast

def predict():
    dummy_input = torch.ones((1, 1, 32, 128, 192), device="cuda")
    if mixed_precision:
        context = autocast
    else:
        context = no_op
    # dummy_input = torch.randn((1, 1, 32, 128, 192), device="cuda").half()
    with context():
        with torch.no_grad():
            torch_output = trainer.network(dummy_input)[0]
            # np.save(base_dir + "torch_output.npy", torch_output.detach().cpu().numpy())
            # print("Torch output: ", torch_output.get_device(), torch_output.shape, torch_output[0, :, 0, 0, 0])


def load_model():
    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(base_dir, folds, mixed_precision=mixed_precision,
                                                      checkpoint_name=checkpoint_name)
    trainer.load_checkpoint_ram(params[0], False)
    trainer.network.do_ds = False
    trainer.network.eval()

    print("emptying cuda cache")
    torch.cuda.empty_cache()
    return trainer


if __name__ == "__main__":
    base_dir = "/data/result/zhongzhiqiang/nnUNet/nnUNet_outputs/predict_temp/tensorrt_test/torch_model/" \
               "Task030_FLARE23OARTumorMultiLabel/nnUNetTrainerV2_FLARE_Medium__nnUNetPlansFLARE23TumorMedium/"

    folds = ["all"]
    mixed_precision = False
    checkpoint_name = "model_final_checkpoint"

    t1 = time.time()
    trainer = load_model()
    t2 = time.time()
    print(">>>load model time: {}s".format(t2 - t1))

    for i in range(400):
        predict()
    t3 = time.time()
    print(">>>predict time: {}s".format(t3 - t2))
