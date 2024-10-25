import os

import torch
from nnfabrik.builder import get_data, get_model

DATASET_CONFIG = {
    "normalize": True,
    "include_behavior": True,
    "include_eye_position": True,
    "batch_size": 1,
    "scale": 0.25,
    "cuda": True,
}

GAUSSIAN_CONFIG = {
    "pad_input": False,
    "stack": -1,
    "layers": 4,
    "hidden_channels": 16,
    "num_rotations": 8,
    "input_kern": 13,
    "hidden_kern": 5,
    "gamma_input": 6.3831,
    "feature_reg_weight": 5,
    "gamma_readout": 5,
    "depth_separable": False,
    "grid_mean_predictor": None,
    "init_sigma": 0.1,
    "init_mu_range": 0.3,
    "gauss_type": "full",
    "shifter": True,
    "regularizer_type": "adaptive_log_norm",
}


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_variable_name(var):
    names = {id(v): k for k, v in globals().items()}
    return names.get(id(var), None)


def fake_forward(readout, inp):
    N = inp.shape[0]
    n_neurons = readout.features.shape[-1]
    x = torch.stack([inp.squeeze()] * n_neurons, axis=-1)
    c = x.shape[1]
    c_in, w_in, h_in = readout.in_shape
    feat = readout.features.view(1, c, readout.outdims)
    bias = readout.bias
    outdims = readout.outdims
    y = (x * feat).sum(1).view(N, outdims)
    if readout.bias is not None:
        y = y + bias
    return y


def load_model(
    model_path,
    dataloaders,
    device,
    factorized,
    regularizer_type,
    use_default_config=True,
    model_config=GAUSSIAN_CONFIG,
    seed=42,
):
    model_fn = "sensorium.models.stacked_core_full_gauss_readout"
    if use_default_config:
        if factorized:
            model_config = GAUSSIAN_CONFIG.copy()
            model_config["shifter"] = False
            model_config["factorized"] = True
        else:
            model_config = GAUSSIAN_CONFIG.copy()
            model_config["regularizer_type"] = regularizer_type
    model = get_model(
        model_fn=model_fn, model_config=model_config, dataloaders=dataloaders, seed=seed
    )
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = model.to(device)
    return model


def load_dataloader(data_path, device, dataset_config=DATASET_CONFIG):
    dataset_config["cuda"] = device
    dataset_fn = "sensorium.datasets.static_loaders"
    filenames = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    dataset_config["paths"] = filenames
    dataloaders = get_data(dataset_fn, dataset_config)
    return dataloaders
