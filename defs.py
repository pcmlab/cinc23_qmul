import numpy as np
import pandas as pd
import torch
import pytorch_lightning
import torch.nn as nn
import yaml
import os, re, struct
from scipy.signal import find_peaks
from torchmetrics.classification import BinaryAccuracy
import h5py
from monai.data import Dataset, list_data_collate, decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric, compute_roc_auc
from sklearn.metrics import roc_auc_score,accuracy_score
import json

def get_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class h5pyDataset_init(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, index):
        return np.array(self.tensor[index][None, :])
    
def read_array_igb(igbfile):
    """
    Purpose: Function to read a .igb file
    """
    data = []
    file = open(igbfile, mode="rb")
    header = file.read(1024)
    words = header.split()
    word = []
    for i in range(4):
        word.append(int([re.split(r"(\d+)", s.decode("utf-8")) for s in [words[i]]][0][1]))

    nnode = word[0] * word[1] * word[2]

    for _ in range(os.path.getsize(igbfile) // 4 // nnode):
        data.append(struct.unpack("f" * nnode, file.read(4 * nnode)))

    file.close()
    return data


def select_closest_nodes(filename_coor, bins,step):
    ind_2d = np.zeros((bins, bins))
    coor = pd.read_csv(filename_coor, sep=' ', names=['x','y','z'])[1:]    
#     coor = np.loadtxt(root_dir+filename_coor, delimiter=' ')
    array= pd.DataFrame(columns = ['x_coor', 'y_coor'])
    array['x_coor'] = coor['x'].values
    array['y_coor'] = coor['y'].values
    
    for x in range(bins):
        for y in range(bins):
            vect_1 = np.array(((2*x+1)/(2*bins), (2*y+1)/(2*bins)))
            part_array = array[(array['x_coor'].between((x/bins), (x+step)/bins, inclusive=False)) \
                  & (array['y_coor'].between((y/bins), (y+step)/bins, inclusive=False))]
            dist=[]
            for i in range(part_array.shape[0]):
                vect_2 = np.array((part_array['x_coor'].iloc[i],part_array['y_coor'].iloc[i]))
                dist.append(np.linalg.norm(vect_1-vect_2)) 
            if np.size(dist) == 0:
                ind_2d[x,y] = float("nan")
                continue
            index =  np.argmin(dist)
            ind_2d[x,y] = part_array.iloc[index].name
    return ind_2d

def dom_freq(data,indexes):
    df_map = np.zeros((indexes.shape[0],indexes.shape[1]))

    for x in range(indexes.shape[0]):
        for y in range(indexes.shape[1]):
            if not np.isnan(indexes[x,y]):
                signal = data[:,int(indexes[x,y])]
                N = int(signal.shape[0])
                fs = 200
                fft_yf = np.fft.fft(signal)
                fft_xf = np.fft.fftfreq(N, 1/fs)

                fft_20_index = np.argwhere((fft_xf<20) & (fft_xf>2))        
                fft_yf_20 = fft_yf[fft_20_index] #cutting on 20Hz
                fft_xf_20 = fft_xf[fft_20_index] #cutting on 20Hz

                spectrum = []
                for i in range(len(np.abs(fft_yf_20))):
                    spectrum.append(int(np.abs(fft_yf_20)[i]))

                peaks, properties = find_peaks(spectrum, height=0)
                try:
                    df_index = np.argsort(properties['peak_heights'])[-1]
                    df = float(fft_xf_20[peaks[df_index]])
                except:
                    df = 0
                df_map[x,y] = df
    return df_map

def last_peak_time(data_AF, indexes):
    last_peak = np.zeros((indexes.shape[0],indexes.shape[1]))
    for x in range(indexes.shape[0]):
        for y in range(indexes.shape[1]):
            try:
                signal = data_AF.T[int(indexes[x,y])]
                peaks, properties = find_peaks(signal)
                last_peak[x,y] = peaks[-1]
            except IndexError: last_peak[x,y] = float('nan')
            except ValueError: last_peak[x,y] = float('nan')
    return last_peak

class h5pyDataset_classification(torch.utils.data.Dataset):
    def __init__(self, labels, tensor_1, tensor_2):
        self.tensor_1 = tensor_1
        self.tensor_2 = tensor_2
        self.labels = labels

    def __len__(self):
        return len(self.tensor_1)

    def __getitem__(self, index):
        return {'fibrosis': np.array(self.tensor_1[index][None, :]),
                'psd': np.array(self.tensor_2[index][None, :]),
                'label': self.labels[index]}

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), 
        nn.ReLU(), 
        nn.BatchNorm2d(output_size), 
        nn.MaxPool2d((2, 2)),
    )

    return block


class Net(pytorch_lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        self.ln1 = nn.Linear(64*4*4, 16) ## change to 64*10*10 here if 96 by 96 pixels or 64*4*4 if 48 by 48
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout2d(0.5)
        self.ln2 = nn.Linear(16, 2)
        
        
        self.conv4 = conv_block(1, 16)
        self.conv5 = conv_block(16, 32)
        self.conv6 = conv_block(32, 64)

        self.ln3 = nn.Linear(64*4*4, 16) ## change to 64*10*10 here if 96 by 96 pixels 
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout2d(0.5)
        self.ln4 = nn.Linear(16, 2)

        self.ln5 = nn.Linear(4, 1)       
        self.loss_function = torch.nn.BCELoss()
        
        self.accuracy = BinaryAccuracy()
        self.rocauc = []
        self.acc = []
        self.best_val_roc = 0
        self.best_val_epoch = 0

    def forward(self, tensor_1, tensor_2):
        tensor_1 = self.conv1(tensor_1)
        tensor_1 = self.conv2(tensor_1)
        tensor_1 = self.conv3(tensor_1)
        tensor_1 = tensor_1.reshape(tensor_1.shape[0], -1)
        tensor_1 = self.ln1(tensor_1)
        tensor_1 = self.relu(tensor_1)
        tensor_1 = self.batchnorm1(tensor_1)
        tensor_1 = self.dropout1(tensor_1)
        tensor_1 = self.ln2(tensor_1)
        tensor_1 = self.relu(tensor_1)
        
        tensor_2 = self.conv4(tensor_2)
        tensor_2 = self.conv5(tensor_2)
        tensor_2 = self.conv6(tensor_2)
        tensor_2 = tensor_2.reshape(tensor_2.shape[0], -1)
        tensor_2 = self.ln3(tensor_2)
        tensor_2 = self.relu(tensor_2)
        tensor_2 = self.batchnorm2(tensor_2)
        tensor_2 = self.dropout2(tensor_2)
        tensor_2 = self.ln4(tensor_2)
        tensor_2 = self.relu(tensor_2)                      
        
        x = torch.cat((tensor_1, tensor_2), dim=1)
        x = self.relu(x)
        return torch.sigmoid(self.ln5(x))

    def prepare_data(self):
        # set up the correct data path
        
        with open("labels.json", 'r') as f: labels = json.load(f) 
            
        cases = h5py.File('fibr_gauss_48.h5','r')
        fibrosis = [cases['fibrosis'][:,:,i] for i in range(self.config['data']['cases'])]
        cases.close() 
        
        cases = h5py.File(self.config['data']['df_file'],'r')
        psd = [cases['DF'][:,:,i] for i in range(self.config['data']['cases'])]
        cases.close()        
        
        threshold = self.config['training']['threshold']
        train_fibrosis, val_fibrosis = fibrosis[:-threshold], fibrosis[-threshold:]
        train_psd, val_psd = psd[:-threshold], psd[-threshold:]
        train_labels, val_labels = labels[:-threshold], labels[-threshold:]

        #set_determinism(seed=0)

        self.train_ds = h5pyDataset_classification(train_labels,train_fibrosis, train_psd)
        self.val_ds = h5pyDataset_classification(val_labels,val_fibrosis, val_psd)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config['training']['batch'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, 
            batch_size=10, 
            num_workers=self.config['training']['num_workers'])
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config['training']['optimizer']['params']['lr'])
        return optimizer   
    
    def training_step(self, batch):
        tensor_1, tensor_2, labels = batch['fibrosis'].float(), batch['psd'].float(), batch['label']
        output = self.forward(tensor_1, tensor_2)
        loss = self.loss_function(output.squeeze(), labels.float())
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        tensor_1, tensor_2, labels = batch['fibrosis'].float(), batch['psd'].float(), batch['label']
        outputs = self.forward(tensor_1, tensor_2)
        loss = self.loss_function(outputs.squeeze(), labels.float())
        tensorboard_logs = {"val_loss": loss.item()}
        print("val_loss", loss.item())
        print('auc = ',roc_auc_score(labels.cpu(),outputs.cpu(),  average="weighted"))
        print('roc_auc = ',compute_roc_auc(y_pred=outputs, y=labels))
        print('acc = ',self.accuracy(outputs.squeeze(), labels.float()))
        self.acc.append(self.accuracy(outputs.squeeze(), labels.float()))
        self.rocauc.append(compute_roc_auc(y_pred=outputs, y=labels))
        print(labels.cpu(),outputs.cpu())
        return {"val_loss": loss, "log": tensorboard_logs,"val_number": len(outputs)}
    
    def on_validation_epoch_end(self):        

        mean_val_acc = torch.mean(torch.stack(self.acc))
        mean_val_roc = np.mean(self.rocauc)
        self.rocauc = []
        self.acc = []

        tensorboard_logs = {
            "val_roc": mean_val_roc,
        }

        if mean_val_roc > self.best_val_roc:
            self.best_val_roc = mean_val_roc
            self.best_val_epoch = self.current_epoch 

        print(
            f"current epoch: {self.current_epoch} "
            f"current mean roc: {mean_val_roc:.4f}"
            f"current mean acc: {mean_val_acc:.4f}"
            f"\nbest mean roc: {self.best_val_roc:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        return {"log": tensorboard_logs}