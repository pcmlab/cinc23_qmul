import os
import shutil
import tempfile
import time
import h5py
#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pandas as pd
import json
import glob
import torch.nn as nn
# from monai.config import print_config
# from monai.utils import first, set_determinism
# from monai.utils import set_determinism
from monai.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
# from sklearn.metrics import roc_auc_score,accuracy_score
# from monai.metrics import ROCAUCMetric, compute_roc_auc
# from torchmetrics.classification import BinaryAccuracy
from clearml import Task
import fire

from defs import get_config, read_array_igb, select_closest_nodes, dom_freq, last_peak_time, Net, h5pyDataset_init

def main(config_file: str, name: str) -> None:
    task = Task.init(project_name="fibrosis_classification", task_name=name)
    task.set_resource_monitor_iteration_timeout(180)
    config = get_config(config_file)
    
    if config['data']['preprocessing']:
        cases = config['data']['cases']
        shape = config['data']['shape']

        # h5 with dom freq maps
        total_start = time.time()
        final_df = np.zeros(((shape,shape,cases)))
        peaks=[]
        for ind in tqdm(range(cases)):
            print(ind+1)
            la_coor_file_2d = config['data']['2d_coord_file']
            try:
                data_AF = np.array(read_array_igb(config['data']['simulaions_path']+str(ind+1)+'/vm_atria_regular.igb'))
            except FileNotFoundError:
                data_AF = np.array(read_array_igb(config['data']['simulaions_path']+str(ind+1)+'/AF/vm_atria_regular.igb'))
            print('data uploaded')
            indexes = select_closest_nodes(la_coor_file_2d, shape, 3)
            df_map = dom_freq(data_AF, indexes)
            final_df[:,:,ind] = df_map

            last_peaks = last_peak_time(data_AF, indexes)
            last_peak = np.nanmax(last_peaks)
            peaks.append(last_peak) 

        labels = pd.DataFrame(peaks)        
        labels_peaks = (labels > config['data']['threshold_last_peak']).astype(int)
        labels = np.array(labels_peaks[0]) #save as json!
        
        with open(config['data']['labels_file'], 'w') as f:
            json.dump(labels.tolist(), f, indent=2) 
        

        total_time = time.time() - total_start
        print(total_time)

        h5f = h5py.File(config['data']['df_file'], 'w')
        h5f.create_dataset('DF', data=final_df)
        h5f.close()
    
    # initialise the LightningModule
    net = Net(config)

    # set up loggers and checkpoints
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger("lightning_logs", name="multi_input")

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
    #     accelerator="cpu",
        devices=config['training']['gpus'],
        strategy='ddp',
        max_epochs=config['training']['epochs'],
        logger=tb_logger,
        enable_checkpointing=True,
        check_val_every_n_epoch=config['training']['val_interval'],
    )

    trainer.fit(net)

if __name__ == '__main__':
    fire.Fire(main)