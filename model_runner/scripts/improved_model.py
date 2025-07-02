import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from torch import Tensor
from torch import sigmoid, int


class ImprovedDiscriminator(nn.Module):
    def __init__(self, in_dim=147, hidden_dims=[256, 128, 64], out_dim=1, 
                 activation="leaky_relu", dropout_rate=0.3, calibrator=None):
        super().__init__()
        if calibrator is not None:
            self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        else:
            self.calibrator = None
        # Расширенный набор активаций
        activations = {
            "leaky_relu": nn.LeakyReLU(0.2),  # Добавлен отрицательный slope
            "relu": nn.ReLU(),
            "selu": nn.SELU(),  # Новый вариант
            "gelu": nn.GELU(),   # Современная активация
            "sigmoid": nn.Sigmoid()
        }
        
        # Проверка входных параметров
        assert activation in activations, f"Активация {activation} не поддерживается"
        
        # Улучшенный первый слой
        self.first_block = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(dropout_rate),
            activations[activation]
        )
        
        # Скрытые блоки с residual-связями
        self.hidden_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.Dropout(dropout_rate),
                activations[activation]
            )
            self.hidden_blocks.append(block)
        
        # Улучшенный выходной слой
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_dim),
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.first_block(x)
        for block in self.hidden_blocks:
            residual = x  # Residual connection
            x = block(x)
            if x.shape == residual.shape:  # Добавляем residual, если размерности совпадают
                x = x + residual
        return self.output(x)
    
    def predict(self, data: Tensor) -> Tensor:
        logits = self.forward(data)
        if self.calibrator is not None:
            proba = self.calibrator.transform(sigmoid(logits).detach().cpu().numpy())
            preds = (proba > 0.5).astype(np.int16)
        else:
            preds = (sigmoid(logits) > 0.5).to(int)
        return preds.detach().cpu().numpy()
    
    def predict_proba(self, data: Tensor) -> Tensor:
        logits = self.forward(data)
        if self.calibrator is not None:
            proba = self.calibrator.transform(sigmoid(logits).detach().cpu().numpy())
            return proba
        else:
            return sigmoid(logits)