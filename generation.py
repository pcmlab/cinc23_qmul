# import os
# import shutil
import tempfile
import time
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from clearml import Task
import fire

from defs import get_config, h5pyDataset_init

def main(config_file: str, name: str) -> None:
    task = Task.init(project_name=name, task_name=name)
    task.set_resource_monitor_iteration_timeout(180)
    config = get_config(config_file)
    
    cases = h5py.File(config['data']['training_file'],'r')
    fibrosis = [cases['fibrosis'][:,:,i] for i in range(config['data']['training_length'])]
    cases.close()         

    train_fibrosis, val_fibrosis = fibrosis[:-config['data']['threshold']], fibrosis[-config['data']['threshold']:]

    train_ds = h5pyDataset_init(train_fibrosis)
    val_ds = h5pyDataset_init(val_fibrosis)

    train_loader = DataLoader(train_ds, 
                              batch_size=config['training']['batch'], 
                              num_workers=config['training']['num_workers'], 
                              shuffle=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, 
                              batch_size=config['training']['batch'], 
                              num_workers=config['training']['num_workers'], 
                              persistent_workers=True)

    device = torch.device(config['training']['gpus'])

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(32, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )
    model.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['training']['optimizer']['params']['lr'])

    inferer = DiffusionInferer(scheduler)

    n_epochs = config['training']['epochs']
    val_interval = config['training']['val_interval']
    epoch_loss_list = []
    val_epoch_loss_list = []
    # device = devices[1]
    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = (batch/config['training']['image_coef']).to(device).float()
    #         print(images)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = (batch/config['training']['image_coef']).to(device).float()
                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")


    length = config['testing']['length'] 
    shape = config['testing']['shape'] 
    results = np.zeros(((length, shape, shape)))
    total_start = time.time()
    for i in tqdm(range(length)):
        noise = torch.randn((1, 1, shape, shape))
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=config['training']['num_inference_steps'])
        with autocast(enabled=True):
            image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)
            results[i] = image[0, 0].cpu()

    h5f = h5py.File(config['testing']['testing_file'], 'w')
    h5f.create_dataset('fibrosis', data=results)
    h5f.close()
    
    total_time = time.time() - total_start
    print(f"generation completed, total time: {total_time}.")    

if __name__ == '__main__':
    fire.Fire(main)