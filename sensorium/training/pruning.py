import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sensorium
from sensorium.models import stacked_core_full_gauss_readout
from sensorium.datasets import static_loaders
from sensorium.utility import *
import random
from neuralpredictors.layers.hermite import RotationEquivariantScale2DLayer
from torch import nn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


CHANNELS = 16
LAST_AFFINE = False


BASEPATH = "../sensorium/sensorium/notebooks/data/"
SAVE_START = '../sensorium/sensorium/notebooks/model_checkpoints/'

model_config = {
  'pad_input': False,
  'stack': -1,
  'layers': 4,
  'hidden_channels': CHANNELS,
  'num_rotations': 8,
  'input_kern': 13,
  'hidden_kern': 5,    
  'gamma_input': 6.3831,
  'feature_reg_weight':5,    
  'gamma_readout': 5,    
  'depth_separable': False,       
  'grid_mean_predictor': {
       'type': 'cortex',
       'input_dimensions': 2,
       'hidden_layers': 1,
       'hidden_features': 30,
       'final_tanh': True
  },
  'init_sigma': 0.1,
  'init_mu_range': 0.3,
  'gauss_type': 'full',
  'shifter': True,
  'final_batchnorm_scale': False, 
  'batch_norm' : True,
  'independent_bn_bias' :False,  # This I'd always do, this arg does not make sense IMHO
  'core_bias' : True,
  'batch_norm_scale': True,
}

filenames = [os.path.join(BASEPATH, file) for file in os.listdir(BASEPATH) if ".zip" in file ]
dataset_config = {
    'paths': filenames,
    'normalize': True,
    'include_behavior': True,
    'include_eye_position': True,
    'batch_size': 256,
    'scale':.25,
}

trainer_config = {
    'max_iter': 450,
    'verbose': True,
    'lr_decay_steps': 4,
    'avg_loss': False,
    'lr_init': 0.005,
    'track_training': True,
    'save_checkpoints' : True,
    'freeze_core' : False,
    'detach_core' : False,
}


def calculate_next_index_to_mask(model, dataloaders, path_to_save, device,
                                 model_config=model_config, prev_masked=[], range_lim=CHANNELS):
    name_prev_masked = '_'
    if len(prev_masked) > 1:
        name_prev_masked = '_' + '_'.join([str(i) for i in prev_masked]) + '_'
        
    scores = []
    m = stacked_core_full_gauss_readout(
                dataloaders=dataloaders,
                seed=seed, 
                **model_config
            )
    
    scale = RotationEquivariantScale2DLayer(num_rotations=model_config['num_rotations'], channels=CHANNELS)
    m.core.features.layer3.add_module('scale', scale)    
    for i in range(range_lim):
        if i not in prev_masked:
            directory = f'{path_to_save}masked{name_prev_masked}{i}/final.pth'
            m.load_state_dict(torch.load(directory))
            m.eval()
            validation_correlation = get_correlations(
                m, dataloaders["validation"], device=device, as_dict=False, per_neuron=False
            )
            scores.append((i, validation_correlation))
        else:
            scores.append((i, -1))
    res = scores[np.argmax(np.asarray([i[1] for i in scores]))]
    return res[0]
    

def train_iteration(model, dataloaders, path_to_save, device,
                    model_config=model_config, prev_masked=[], range_lim=CHANNELS, iterate_over=None):
    name_prev_masked = ''
    if len(prev_masked) > 0:
        name_prev_masked = '_'.join([str(i) for i in prev_masked])
        
    if iterate_over is None:
        iterate_over = list(range(range_lim))
    for hidden_chan in iterate_over:
        if hidden_chan not in prev_masked:
            channels_to_hide = prev_masked + [hidden_chan]

            model = stacked_core_full_gauss_readout(
                dataloaders=dataloaders,
                seed=seed, 
                **model_config
            )
                
            model.core.features.layer3.norm.batch_norm.affine=False
            for param in model.core.features.layer3.norm.batch_norm.parameters():
                param.requires_grad = False
            
            scale = RotationEquivariantScale2DLayer(num_rotations=model_config['num_rotations'], channels=CHANNELS)
            if len(prev_masked) == 0:
                model.load_state_dict(torch.load(f'{path_to_save}final.pth'))                
                model.core.features.layer3.add_module('scale', scale)
            else:
                model.core.features.layer3.add_module('scale', scale)                
                model.load_state_dict(torch.load(f'{path_to_save}masked_{name_prev_masked}/final.pth'))

            with torch.no_grad():
                for param in model.core.features.layer3.scale.parameters():
                    for c in channels_to_hide:
                        param[:,:, c, :, :] = 0.0
                    param.requires_grad = False
            if len(prev_masked) > 0:
                directory = f'{path_to_save}/masked_{name_prev_masked}_{hidden_chan}/'
            else:
                directory = f'{path_to_save}/masked_{hidden_chan}/'
                
            if not os.path.exists(directory):
                os.makedirs(directory)    
            trainer_config['device'] = f"cuda:{dev}"
            trainer_config['checkpoint_save_path'] = directory
            model.train()
            validation_score, trainer_output, state_dict = standard_trainer(model, 
                                                                            dataloaders, 
                                                                            seed=seed,
                                                                            **trainer_config
                                                                           )

def run_pruning(seed, dev, path_to_save=SAVE_START, pruning_iterations=CHANNELS):
    seed = 7607
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.set_device(dev)
    ## TODO - add first training iteration

    dataloaders = static_loaders(**dataset_config)
    model = stacked_core_full_gauss_readout(
            dataloaders=dataloaders,
            seed=seed, 
            **model_config
        )
    device = f'cuda:{DEV}'
    trainer_config['device'] = device
    trainer_config['checkpoint_save_path'] = path_to_save
    model.train()
    validation_score, trainer_output, state_dict = standard_trainer(model, 
                                                                    dataloaders, 
                                                                    seed=seed,
                                                                    **trainer_config
                                                                   )                                   
    hid = []
    for i in range(pruning_iterations):
        train_iteration(model, dataloaders, path_to_save, device, prev_masked=hid, range_lim=pruning_iterations)
        loc_hid = calculate_next_index_to_mask(model, dataloaders, path_to_save, device, hid, range_lim=pruning_iterations)
        hid = hid + [loc_hid]                       
    return hid
    
if __name__ == "__main__":
    hid = run_pruning(seed, dev)
    print(f'The order of channels pruning : {hid}')

