import argparse
import pickle

import numpy as np
from tqdm import tqdm

from .gabor_stimuli import GaborSet
from .surround_suppresion_stimuli import CenterSurround
from .utils import *

# TODO - this file does not handle factorized readout models

MAX_CONTRAST = 1.75
MIN_CONTRAST = 0.01 * MAX_CONTRAST
MIN_SIZE = 4
MAX_SIZE = 25

num_contrasts = 6
contrast_increment = (MAX_CONTRAST / MIN_CONTRAST) ** (1 / (num_contrasts - 1))
num_sizes = 8
size_increment = (MAX_SIZE / MIN_SIZE) ** (1 / (num_sizes - 1))
canvas_size = [MAX_SIZE, MAX_SIZE]

def phase_invariance_images_batch(optimal_gabor_params, phases):
    phases_g = GaborSet(
        canvas_size=canvas_size,
        center_range=(
            optimal_gabor_params["location"][0],
            optimal_gabor_params["location"][0] + 1,
            optimal_gabor_params["location"][1],
            optimal_gabor_params["location"][1] + 1,
        ),
        sizes=[optimal_gabor_params["size"]],
        spatial_frequencies=[optimal_gabor_params["spatial_frequency"]],
        contrasts=[optimal_gabor_params["contrast"]],
        orientations=[optimal_gabor_params["orientation"]],
        phases=list(phases),
        relative_sf=False,
    )
    return  phases_g.images()

def orientation_images_batch(optimal_gabor_params, orientations, n_phases):
    orientations_g = GaborSet(
        canvas_size=canvas_size,
        center_range=(
            optimal_gabor_params["location"][0],
            optimal_gabor_params["location"][0] + 1,
            optimal_gabor_params["location"][1],
            optimal_gabor_params["location"][1] + 1,
        ),
        sizes=[optimal_gabor_params["size"]],
        spatial_frequencies=[optimal_gabor_params["spatial_frequency"]],
        contrasts=[optimal_gabor_params["contrast"]],
        orientations=list(orientations),
        phases=n_phases,
        relative_sf=False,
    )
    return orientations_g.images()


def surround_suppresion_images_batch(optimal_gabor_params):
    center_range = [optimal_gabor_params['location'][0], 
                    optimal_gabor_params['location'][0] + 1,
                    optimal_gabor_params['location'][1],
                    optimal_gabor_params['location'][1] + 1]

    sizes_center = [-0.01] + list(
        p["min_size_center"] * p["size_center_increment"] ** np.arange(p["num_sizes_center"])
    )
    sizes_surround = [1]
    contrasts_center = p["min_contrast_center"] * p["contrast_center_increment"] ** np.arange(
        p["num_contrasts_center"]
    )
    contrasts_surround = p["min_contrast_surround"] * p["contrast_surround_increment"] ** np.arange(
        p["num_contrasts_surround"]
    )

    center_surround = CenterSurround(
        canvas_size=canvas_size,
        center_range=center_range,
        sizes_total=[p["total_size"]],
        sizes_center=sizes_center,
        sizes_surround=sizes_surround,
        contrasts_center=contrasts_center,
        contrasts_surround=contrasts_surround,
        orientations_center=[optimal_gabor_params['orientation']],
        orientations_surround=[optimal_gabor_params['orientation']],
        spatial_frequencies=[optimal_gabor_params['spatial_frequency']],
        phases=[optimal_gabor_params['phase']],
    )
    return center_surround.images()


def add_behaviour_as_channel(beh, images, device):
    behavior_chosen = torch.stack(
        [torch.ones(3, MAX_SIZE, MAX_SIZE) *  torch.Tensor(beh).reshape(3, 1, 1)] * images.shape[0]
    ).to(device)
    images = torch.concat([images, behavior_chosen], axis=1)
    return images.float()


def gaussian_forward(model, images, session_k):
    loc_out = model.core(images)
    sel_idx = loc_out.shape[-1] // 2 + loc_out.shape[-1] % 2
    loc_out = loc_out[:, :, sel_idx, sel_idx].unsqueeze(-1).unsqueeze(-1)
    out = fake_forward(model.readout[session_k], loc_out)
    if model.nonlinearity_type == "elu":
        out = model.nonlinearity_fn(out + model.offset) + 1
    else:
        out = model.nonlinearity_fn(out)
    return out


def surround_suppresion_tuning(model, dataloaders, max_idx_dict_path, params_gabor, median_beh, device):
    ss_curve = {}
    ss_idx = {}
    with torch.no_grad():
        with open(max_idx_dict_path, 'rb') as f:
            max_idxs_dict = pickle.load(f)
        for k in  dataloaders['train'].keys():
            session = k.replace('-', '_')
            max_idxs = max_idxs_dict[k]
            max_idxs_res_ss = {}
            max_idxs_res_idxs_ss = {}
            for i in tqdm(range(max_idxs.shape[0])):
                optimal_gabor_params = params_gabor.params_dict_from_idx(max_idxs[i])
                images = surround_suppresion_images_batch(optimal_gabor_params)
                images = torch.tensor(images).unsqueeze(1).to(device)
                images = add_behaviour_as_channel(median_beh[k], images, device)

                out = gaussian_forward(model, images, k)
                tuning_curve_center = out.detach().cpu().numpy()[:, i]
                center_max_global = np.max(tuning_curve_center)
                center_max_global_i = np.argmax(tuning_curve_center)

                for center_max_i, center_max in enumerate(tuning_curve_center):
                    if center_max >= 0.95 * center_max_global:
                        break

                center_asymptote = tuning_curve_center[-1]
                suppression_index = (center_max - center_asymptote) / center_max
                max_idxs_res_ss[i] = tuning_curve_center
                max_idxs_res_idxs_ss[i] = suppression_index
            max_idxs_res_global_ss[k] = max_idxs_res_ss
            max_idxs_res_global_idxs_ss[k] = max_idxs_res_idxs_ss
    return max_idxs_res_global_ss, max_idxs_res_global_idxs_ss


def orientation_tuning(model, dataloaders, max_idx_dict_path, params_gabor, device, median_beh, n_orientations=24, n_phases=8):
    res_global_idxs_orient = {}
    res_global_orient = {}
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
    with torch.no_grad():
        res_global_orient = {}
        res_global_idxs_orient = {}
        with open(max_idx_dict_path, 'rb') as f:
            max_idxs_dict = pickle.load(f)
        for k in  dataloaders['train'].keys():
            session = k.replace('-', '_')
            max_idxs = max_idxs_dict[k]
            res_orient = {}
            res_idxs_orient = {}
            for i in tqdm(range(max_idxs.shape[0])):

                optimal_gabor_params = params_gabor.params_dict_from_idx(max_idxs[i])
                images = orientation_images_batch(optimal_gabor_params, orientations, n_phases)
                images = torch.tensor(images).unsqueeze(1).to(device)
                images = add_behaviour_as_channel(median_beh[k], images, device)

                out = gaussian_forward(model, images, k)
                output = out.detach().cpu().numpy()[:, i]
                r = output
                r = np.reshape(r, (n_orientations, n_phases))  
                r = np.max(r, axis=1)
                assert len(r) == n_orientations
                f0 = r.mean()
                f2 = np.linalg.norm((r * np.exp(orientations * 2j)).mean())
                orientation_tuning = f2 / f0
                res_idxs_orient[i] = orientation_tuning
                res_orient[i] = r
                
            res_global_idxs_orient[k] = res_idxs_orient
            res_global_orient[k] = res_orient
    return res_global_orient, res_global_idxs_orient


def cross_orientation_inhibition_tuning(model, dataloaders, max_idx_dict_path, params_gabor, device, median_beh, num_contrasts=9):
    contrast_increment = (0.5 * MAX_CONTRAST / MIN_CONTRAST) ** (1 / (num_contrasts - 1))
    contents = [[1, MIN_CONTRAST, num_contrasts, contrast_increment, 8]]
    orth_phase_shifts = np.linspace(start=0, stop=2 * np.pi, endpoint=False, num=num_phases)
    with torch.no_grad():
        res_global_chi = {}
        res_global_idxs_chi = {}
        for k in  dataloaders['train'].keys():
            session = k.replace('-', '_')
            with open(max_idx_dict_path, 'rb') as f:
                max_idxs_dict = pickle.load(f)
            max_idxs = max_idxs_dict[k]
            res_chi = {}
            res_idxs_chi = {}
            for i in tqdm(range(max_idxs.shape[0])):

                optimal_gabor_params = params_gabor.params_dict_from_idx(max_idxs[i])

                c = min_contrast * contrast_increment ** np.arange(num_contrasts)
                contrasts = np.concatenate([np.zeros(1), c], axis=0)
                tuning_curve_lst = []
                g_pref = GaborSet(
                        canvas_size=canvas_size,
                        center_range=(
                            optimal_gabor_params["location"][0],
                            optimal_gabor_params["location"][0] + 1,
                            optimal_gabor_params["location"][1],
                            optimal_gabor_params["location"][1] + 1,
                        ),
                        sizes=[optimal_gabor_params["size"]],
                        spatial_frequencies=[optimal_gabor_params["spatial_frequency"]],
                        contrasts=contrasts,
                        orientations=[optimal_gabor_params["orientation"]],
                        phases=[optimal_gabor_params["phase"]],
                        relative_sf=False,
                    )
                comps_pref = g_pref.images()

                for ph_shift in orth_phase_shifts:

                    g_orth = GaborSet(
                        canvas_size=canvas_size,
                        center_range=(
                            optimal_gabor_params["location"][0],
                            optimal_gabor_params["location"][0] + 1,
                            optimal_gabor_params["location"][1],
                            optimal_gabor_params["location"][1] + 1,
                        ),
                        sizes=[optimal_gabor_params["size"]],
                        spatial_frequencies=[optimal_gabor_params["spatial_frequency"]],
                        contrasts=contrasts,
                        orientations=[optimal_gabor_params["orientation"] + np.pi / 2],
                        phases=[optimal_gabor_params["phase"] + ph_shift],
                        relative_sf=False,
                    )
                    comps_orth = g_orth.images()
                    plaids = comps_pref[None, ...] + comps_orth[:, None, ...]

                    images_or = np.reshape(plaids, [-1] + [1] + canvas_size )
                    images = torch.tensor(images_or).to(device)

                    images = add_behaviour_as_channel(median_beh[k], images, device)
                    out = gaussian_forward(model, images, k)
                    output = out.detach().cpu().numpy()[:, i].reshape(len(contrasts), len(contrasts)) # this is 100 numbers, hence 10 x 10
                    tuning_curve_lst.append(output)

                tuning_curve = np.stack(tuning_curve_lst)
                phase_avg_tc = tuning_curve.mean(axis=0)
                index_curve = 1 - (phase_avg_tc / phase_avg_tc[:1, :])
                coi_index = index_curve.max()
                res_chi[i] = phase_avg_tc
                res_idxs_chi[i] = coi_index
            res_global_chi[k] = res_chi
            res_global_idxs_chi[k] = res_idxs_chi
    return res_global_chi, res_global_idxs_chi


def phase_invariance_tuning(model, dataloaders, max_idx_dict_path, params_gabor, device, median_beh, n_phases=12):
    phases = np.arange(n_phases) * (2 * np.pi) / n_phases
    with torch.no_grad():
        max_idxs_res_global = {}
        max_idxs_res_global_idxs = {}
        with open(max_idx_dict_path, 'rb') as f:
            max_idxs_dict = pickle.load(f)
        for k in dataloaders['train'].keys():
            session = k.replace('-', '_')
            max_idxs = max_idxs_dict[k]
            max_idxs_res = {}
            max_idxs_res_idxs = {}
            for i in tqdm(range(max_idxs.shape[0])):
                optimal_gabor_params = params_gabor.params_dict_from_idx(max_idxs[i])
                images = phase_invariance_images_batch(optimal_gabor_params, phases)
                images = torch.tensor(images).unsqueeze(1).to(device)
                images = add_behaviour_as_channel(median_beh[k], images, device)

                out = gaussian_forward(model, images, k)
                output = out.detach().cpu().numpy()[:, i]
                max_idxs_res[i] = output 
                max_idxs_res_idxs[i] = compute_ph_inv_index(output)
            max_idxs_res_global[k] = max_idxs_res
            max_idxs_res_global_idxs[k] = max_idxs_res_idxs
    return max_idxs_res_global, max_idxs_res_global_idxs


  def compute_ph_inv_index(tc):
    f0 = tc.mean()
    f1 = np.linalg.norm((tc * np.exp(phases * 1j)).mean())
    phase_tuning = f1 / f0
    phase_invariance = 1 - phase_tuning
    return phase_invariance          


if __main__ == 'main':
    ## TODO - add argparse with median_behaviour_path, data_path, device, model_chekpoint_path, factorized, regularizer_type, phase_inv_path, coi_path, orient_path, ss_path

    parser = argparse.ArgumentParser(description="Script for 3D Point Cloud Classification")
    
    parser.add_argument("--median_behaviour_path", type=str, required=True, help="Path to the median behaviour file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computation (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--factorized", type=bool, default=False,  help="Flag to indicate whether to use a factorized model")
    parser.add_argument("--regularizer_type", type=str, choices=["L1", "L2", "adaptive_log_norm"], default="L1", help="Type of regularizer to use, only for gaussian model")
    parser.add_argument("--phase_inv_path", type=str, required=True, help="Path to the phase invariance file")
    parser.add_argument("--coi_path", type=str, required=True, help="Path to the COI file")
    parser.add_argument("--orient_path", type=str, required=True, help="Path to the orientation file")
    parser.add_argument("--ss_path", type=str, required=True, help="Path to the SS file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the tuning dicts")
    
    args = parser.parse_args()

    # value from here https://github.com/ecker-lab/burg2021_learning_divisive_normalization/blob/main/divisivenormalization/insilico.py#L211C38-L211C58
    x_start = 0
    x_end = MAX_SIZE
    y_start = 0
    y_end = MAX_SIZE
    min_size = MIN_SIZE
    min_sf = (1.3 ** -1)
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
    model = load_model(args.model_path, dataloaders, args.device, args.factorized, args.regularizer_type,)
    phase_inv_curve, phase_inv_idx = phase_invariance_tuning(model, dataloaders, args.phase_inv_path, params_gabor, args.device, median_beh)
    coi_curve, coi_idx = cross_orientation_inhibition_tuning(model, dataloaders, args.coi_path, params_gabor, args.device, median_beh)
    orient_curve, orient_idx = orientation_tuning(model, dataloaders, args.orient_path, params_gabor, args.device, median_beh)
    ss_curve, ss_idx = surround_suppresion_tuning(model, dataloaders, args.ss_path, params_gabor, median_beh, args.device)

    save_lst = [phase_inv_curve, phase_inv_idx, coi_curve, coi_idx, orient_curve, orient_idx, ss_curve, ss_idx]
    for idx in range(len(save_lst)):
        name = get_variable_name(save_lst[idx])
        save_file(f'{args.save_path}/{name}.pkl', save_lst[idx])

 

    