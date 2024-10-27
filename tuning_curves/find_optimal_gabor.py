import argparse
import pickle

import numpy as np
from tqdm import tqdm

from .make_gabors import GaborSet
from .utils import *

MAX_CONTRAST = 1.75
MIN_CONTRAST = 0.01 * MAX_CONTRAST
MIN_SIZE = 4
MAX_SIZE = 25

num_contrasts = 6
contrast_increment = (MAX_CONTRAST / MIN_CONTRAST) ** (1 / (num_contrasts - 1))
num_sizes = 8
size_increment = (MAX_SIZE / MIN_SIZE) ** (1 / (num_sizes - 1))


def initializa_response_dict(keys, dataloaders):
    max_response_dict = {}
    max_idx_dict = {}
    for k in keys:
        max_response = np.asarray(
            [0] * dataloaders["train"][k].dataset.__getitem__(0).responses.shape[0]
        )
        max_idx = np.asarray(
            [0] * dataloaders["train"][k].dataset.__getitem__(0).responses.shape[0]
        )

        max_response_dict[k] = max_response
        max_idx_dict[k] = max_idx
    return max_response_dict, max_idx_dict


def compute_gabor_responses(
    gabor_path, num_of_batches, model, dataloaders, median_beh, device, factorized
):
    keys = list(model.readout.keys())
    max_response_dict, max_idx_dict = initializa_response_dict(keys, dataloaders)
    with torch.no_grad():
        for batch_idx in tqdm(range(num_of_batches)):
            batch_start = batch_idx * num_of_batches
            images = np.load(f"{gabor_path}{batch_idx}.npy")
            batch_end = batch_start + images.shape[0]
            images_or = torch.tensor(np.array(images)).to(device)
            output = []
            images = torch.tensor(images_or).unsqueeze(1)
            images = add_behaviour_as_channel(median_beh[k], images, device)
            loc_out = model.core(images)
            if not factorized:
                sel_idx = loc_out.shape[-1] // 2 + loc_out.shape[-1] % 2
                loc_out = loc_out[:, :, sel_idx, sel_idx].unsqueeze(-1).unsqueeze(-1)
            for k in keys:
                if factorized:
                    out = model.readout[k].forward(loc_out.clone())
                else:
                    out = fake_forward(model.readout[k], loc_out.clone())
                if model.nonlinearity_type == "elu":
                    out = model.nonlinearity_fn(out + model.offset) + 1
                else:
                    out = model.nonlinearity_fn(out)
                output = out.detach().cpu()

                loc_maxes = torch.max(output, axis=0)
                max_r = loc_maxes.values.numpy()
                max_i = batch_idx * BATCH_SIZE + loc_maxes.indices.numpy()
                new_max = max_response_dict[k] < max_r
                max_response_dict[k] = np.maximum(max_response_dict[k], max_r)
                max_idx_dict[k] = ~new_max * max_idx_dict[k] + new_max * max_i
    return max_response_dict, max_idx_dict


if __main__ == "main":
    ## TODO - add argparse with median_behaviour_path, data_path, device, model_chekpoint_path, factorized, regularizer_type, phase_inv_path, coi_path, orient_path, ss_path

    parser = argparse.ArgumentParser(description="Script for finding optimal gabors")

    parser.add_argument(
        "--median_behaviour_path",
        type=str,
        required=True,
        help="Path to the median behaviour file",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data folder"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for computation (e.g., 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--factorized",
        type=bool,
        default=False,
        help="Flag to indicate whether to use a factorized model",
    )
    parser.add_argument(
        "--regularizer_type",
        type=str,
        choices=["L1", "L2", "adaptive_log_norm"],
        default="L1",
        help="Type of regularizer to use, only for gaussian model",
    )
    parser.add_argument(
        "--gabor_path",
        type=str,
        required=True,
        help="Path to the precomputed gabors",
    )
    parser.add_argument(
        "--num_of_batches",
        type=int,
        default=128,
        help="batch size for optimal gabor search",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the tuning dicts"
    )

    args = parser.parse_args()

    if args.factorized:
        canvas_size = [64, 36]
        x_start = 0
        x_end = 64
        y_start = 0
        y_end = 36
    else:
        x_start = 0
        x_end = MAX_SIZE
        y_start = 0
        y_end = MAX_SIZE
        canvas_size = [MAX_SIZE, MAX_SIZE]

    min_size = MIN_SIZE
    min_sf = 1.3**-1
    num_sf = 10
    sf_increment = 1.3
    min_contrast = MIN_CONTRAST
    num_orientations = 12
    num_phases = 8
    center_range = [x_start, x_end, y_start, y_end]
    sizes = min_size * size_increment ** np.arange(num_sizes)
    sfs = min_sf * sf_increment ** np.arange(num_sf)
    c = min_contrast * contrast_increment ** np.arange(num_contrasts)
    params_gabor = GaborSet(
        canvas_size,
        center_range,
        sizes,
        sfs,
        c,
        num_orientations,
        num_phases,
    )
    with open(args.median_behaviour_path, "rb") as f:
        median_beh = pickle.load(f)
    dataloaders = load_dataloader(args.data_path, args.device)
    model = load_model(
        args.model_path,
        dataloaders,
        args.device,
        args.factorized,
        args.regularizer_type,
    )
    max_response_dict, max_idx_dict = compute_gabor_responses(
        args.gabor_path,
        args.num_of_batches,
        model,
        dataloaders,
        median_beh,
        device,
        factorized,
    )
    save_lst = [max_response_dict, max_idx_dict]
    for idx in range(len(save_lst)):
        name = get_variable_name(save_lst[idx])
        save_file(f"{args.save_path}/{name}.pkl", save_lst[idx])
