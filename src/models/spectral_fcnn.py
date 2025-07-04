import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SpectralClassifierFCNN(nn.Module):
    def __init__(self, 
                 in_dim=147, 
                 hidden_dims=[256, 128, 64], 
                 out_dim=1, 
                 activation="leaky_relu", 
                 dropout_rate=0.3, 
                 random_seed=None):
        super().__init__()
        
        activations = {
            "leaky_relu": nn.LeakyReLU(0.2),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "silu": nn.SiLU()
        }
        if random_seed is not None:
            self.set_seed(random_seed) 
        
        self.first_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Dropout(dropout_rate),
            activations[activation]
        )
        
        self.hidden_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.Dropout(dropout_rate),
                activations[activation]
            )
            self.hidden_blocks.append(block)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_dim),
        )

        self._init_weights()

    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def predict(self, val_dataloader):
        if isinstance(val_dataloader, DataLoader):
            pass
        elif isinstance(val_dataloader, TensorDataset):
            val_dataloader = DataLoader(val_dataloader, 
                                        batch_size=64, 
                                        shuffle=False)
                
        device = "cuda" \
            if next(self.parameters()).is_cuda \
            else "cpu"
        self.to(device)
        self.eval()
        preds_all = np.array([])
        proba_all = np.array([])
        labels_all = np.array([])
        for batch, labels in val_dataloader:
            with torch.no_grad():
                batch = batch.to(torch.float)
                logits = self(batch.to(device)).squeeze().to(torch.float)
                proba = torch.sigmoid(logits)
                preds = (proba > 0.5).to(torch.int)

                preds_all = np.concatenate((preds_all, 
                                            preds.detach().cpu().numpy()))
                labels_all = np.concatenate((labels_all, 
                                             labels.detach().cpu().numpy()))
                proba_all = np.concatenate((proba_all, 
                                            proba.detach().cpu().numpy()))

        return preds_all, proba_all, labels_all
    
    def create_default_dataloader(self, 
                                  X: pd.DataFrame, 
                                  y: pd.Series, 
                                  batch_size: int=32, 
                                  shuffle: bool=True, 
                                  worker_seed=42) -> DataLoader:

        labels = torch.tensor(y.to_numpy())
        data = torch.tensor(X.to_numpy(), dtype=torch.float)

        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                worker_init_fn=worker_seed)
        return dataloader
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, 
                                        mode='fan_out', 
                                        nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.first_block(x)
        for block in self.hidden_blocks:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x + residual
        return self.output(x)