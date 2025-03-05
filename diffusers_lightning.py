import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers import DDIMScheduler, UNet2DModel
from datetime import datetime
import torchvision
from torchvision import transforms
import numpy as np
import nrrd
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger


torchvision.disable_beta_transforms_warning()

neptune_key = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZWUyNGYzZi05ZDE0LTQwYjAtYTQzOS04M2QxZmQ5MTQ0MjcifQ=="

neptune_logger = NeptuneLogger(
    api_key=neptune_key,
    project="alexandrejarry/data-synthesis",
)

class DatasetFromDataFrame(Dataset):
    def __init__(self, root_dir, dataframe, transform=None):
        self.root_dir = root_dir
        self.df = dataframe
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for i in range(self.df.shape[0]):
            img_name = self.df.iloc[i,0]
            self.image_paths.append(os.path.join(self.root_dir,img_name))
            self.labels.append(0)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        data, header = nrrd.read(img_path)
        data = np.squeeze(data)
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-5)
        scaled_data = (normalized * 255)
        scaled_data = scaled_data[0].astype(np.uint8)
        image = Image.fromarray(scaled_data).convert('RGB')
        image = image.rotate(270)
    
        if self.transform:
            image = self.transform(image)
        
        label = int(self.labels[idx])

        return image, label

transform = transforms.Compose([
    transforms.Resize((256,256)),  
    transforms.ToTensor(),  
])

class DiffusionModel(pl.LightningModule):
    def __init__(self, lr=1e-5, num_train_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel()
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")
        self.loss_fn = nn.MSELoss()

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Ignore labels if they exist
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        noise = torch.rand_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        outputs = self(x, timesteps)
        loss = self.loss_fn(outputs, x)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    

parquet = "/mnt/raid/C1_ML_Analysis/CSV_files/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_noflyto_1e-4.parquet"
df = pd.read_parquet(parquet, engine="pyarrow")
num_samples = 50000
df_sampled = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

train_root_dir = '/mnt/raid/C1_ML_Analysis'
full_dataset = DatasetFromDataFrame(root_dir=train_root_dir,dataframe=df_sampled,transform=transform)
train_ratio = 0.9
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
# Dataset & DataLoader
batch_size = 8
num_workers = 4

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Checkpoint directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"outputs_lightning/output_{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

# PyTorch Lightning Trainer
trainer = pl.Trainer(
    accelerator="gpu", 
    logger=neptune_logger,
    devices=4,  
    strategy=DDPStrategy(),
    precision=16, 
    max_epochs=1,
    log_every_n_steps=10,
    enable_progress_bar=True,
)

# Train the model
model = DiffusionModel()
trainer.fit(model, train_dataloader, val_dataloader)


final_model_path = os.path.join(output_dir, "model.pth")
torch.save(model.state_dict(), final_model_path)