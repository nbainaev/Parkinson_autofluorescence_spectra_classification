import torch
import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y, check_array

class NeuralNetCalibrator:
    def __init__(self, 
                 model, 
                 methods=['isotonic'], 
                 n_bins=15, 
                 fcnn_hidden_size=32, 
                 fcnn_lr=1e-3, 
                 fcnn_epochs=100, 
                 batch_size=64):

        self.model = model
        self.methods = methods
        self.n_bins = n_bins
        self.calibrator = None
        self.fcnn_hidden_size = fcnn_hidden_size
        self.fcnn_lr = fcnn_lr
        self.fcnn_epochs = fcnn_epochs
        self.batch_size = batch_size
        
        self.fcnn_ = None
        
    def fit(self, 
            X_val, 
            y_val,
            batch_size=128):

        val_probs = self._predict_proba(X_val, 
                                        batch_size)
        val_probs = val_probs.cpu().numpy()
        y_val = y_val.cpu().numpy() if \
            isinstance(y_val, torch.Tensor) else y_val
        
        if 'isotonic' in self.methods:
            self.calibrator_iso = IsotonicRegression(
                out_of_bounds='clip'
                )
            self.calibrator_iso.fit(
                val_probs, y_val
                )
            
        if 'platt' in self.methods:

            self.calibrator_platt = LogisticRegression()
            self.calibrator_platt.fit(
                val_probs.reshape(-1, 1), 
                y_val
            )
            
        if 'temperature' in self.methods:
            self.temperature = torch.nn.Parameter(
                torch.ones(1)
                )
            optimizer = torch.optim.AdamW(
                [self.temperature], lr=0.01
                )
        
            def eval():
                optimizer.zero_grad()
                loss = torch.nn.BCELoss()(
                    torch.sigmoid(
                        self._logits(
                            X_val
                        ).squeeze() / self.temperature
                    ),
                    torch.FloatTensor(y_val)
                )
                loss.backward()
                return loss
                
            optimizer.step(eval)
        if 'isotemp' in self.methods:
            self.temperature = torch.nn.Parameter(
                torch.ones(1)
                )
            optimizer = torch.optim.LBFGS(
                [self.temperature], lr=0.01
                )
        
            def eval():
                optimizer.zero_grad()
                loss = torch.nn.BCELoss()(
                    torch.sigmoid(
                        self._logits(
                            X_val
                        ).squeeze() / self.temperature
                    ),
                    torch.FloatTensor(y_val)
                )
                loss.backward()
                return loss
                
            optimizer.step(eval)

            self.calibrator_isotemp = IsotonicRegression(
                out_of_bounds='clip'
                )
            self.calibrator_isotemp.fit(
                torch.sigmoid(
                    self._logits(
                        X_val
                    ).squeeze() / self.temperature
                ).detach().cpu().numpy(), 
                y_val
            )
        
        if 'fcnn' in self.methods:
            val_probs = self._predict_proba(
                X_val, batch_size
                )
            self._train_fcnn(
                val_probs, y_val
                )
    
    def _train_fcnn(self, svc_output, y):
        n_features = svc_output.shape[1] if\
            len(svc_output.shape) > 1 else 1
        
        self.fcnn_ = nn.Sequential(
            nn.Linear(n_features, self.fcnn_hidden_size),
            nn.GELU(),
            nn.Linear(self.fcnn_hidden_size, 1),
        )
        weight = len(y[y == 0]) / len(y[y == 1])
        optimizer = torch.optim.AdamW(
            self.fcnn_.parameters(), lr=self.fcnn_lr
            )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([weight])
            )
        
        X_tensor = torch.FloatTensor(
            svc_output
            ).unsqueeze(1) if n_features == 1 else\
                  torch.FloatTensor(
                        svc_output
                        )
        y_tensor = torch.FloatTensor(
            (y == 1).astype(np.float32)
            )
        
        dataset = torch.utils.data.TensorDataset(
            X_tensor, y_tensor
            )
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
            )
        
        for epoch in range(self.fcnn_epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.fcnn_(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
    def calibrate_proba(self, X, batch_size=128):
        probs = self._predict_proba(
            X, batch_size
            ).cpu().numpy()
        proba_calib = []
        if 'isotonic' in self.methods:
            proba_calib.append(
                self.calibrator_iso.predict(probs)
                )
        if 'platt' in self.methods:
            proba_calib.append(
                self.calibrator_platt.predict_proba(
                    probs.reshape(-1, 1)
                    )[:, 1])
        if 'temperature' in self.methods:
            logits = self._logits(X)
            proba_calib.append(
                torch.sigmoid(
                    logits.squeeze() / self.temperature
                    ).detach().cpu().numpy()
                )
        
        if 'isotemp' in self.methods:
            logits = self._logits(X)
            proba_calib .append(
                self.calibrator_isotemp.predict(
                    torch.sigmoid(
                        logits.squeeze() / self.temperature
                        ).detach().cpu().numpy()
                    )
                )
        
        if 'fcnn' in self.methods:
            logits = self._logits(X)
            proba_raw = torch.sigmoid(logits.squeeze())
            with torch.no_grad():
                svc_tensor = torch.FloatTensor(
                    proba_raw
                    ).unsqueeze(1) if\
                        len(proba_raw.shape) == 1 else\
                        torch.FloatTensor(proba_raw)
                proba_positive = torch.sigmoid(
                    self.fcnn_(svc_tensor).flatten()
                ).numpy()
            
            proba = np.zeros((len(X), 2))
            proba[:, 0] = 1 - proba_positive
            proba[:, 1] = proba_positive
            proba_calib.append(proba_positive)

        if len(proba_calib) > 1:
            proba_mean = proba_calib[0]
            for proba in proba_calib[1:]:
                proba_mean += proba
            return proba / len(proba_calib)
        else:
            return proba_calib[0]
        
    def _predict_proba(self, X, batch_size):

        self.model.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
            
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch in loader:
                outputs = self.model(
                    batch[0].to(next(
                        self.model.parameters()
                        ).device)
                    )
                probs.append(torch.sigmoid(outputs))
                
        return torch.cat(probs)
    
    def _logits(self, X):
        self.model.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
            
        with torch.no_grad():
            logits = self.model(
                X.to(next(
                    self.model.parameters()
                    ).device)
                )
        return logits

class SVCWithCalibration(BaseEstimator, ClassifierMixin):
    def __init__(self, model=None, model_config=None, fcnn_hidden_size=32, 
                 fcnn_lr=0.01, fcnn_epochs=1000, batch_size=64):
        """
        SVC + FCNN калибратор вероятностей
        
        Параметры:
        -----------
        svc_params : dict, optional
            Параметры для SVC (по умолчанию {'kernel':'rbf', 'probability':False})
        fcnn_hidden_size : int, optional
            Размер скрытого слоя FCNN (по умолчанию 32)
        fcnn_lr : float, optional
            Learning rate для FCNN (по умолчанию 0.01)
        fcnn_epochs : int, optional
            Количество эпох обучения FCNN (по умолчанию 1000)
        batch_size : int, optional
            Размер батча для обучения FCNN (по умолчанию 64)
        """
        self.model_config = model_config if model_config else {'kernel':'rbf', 'probability':False}
        self.fcnn_hidden_size = fcnn_hidden_size
        self.fcnn_lr = fcnn_lr
        self.fcnn_epochs = fcnn_epochs
        self.batch_size = batch_size
        
        self.is_fitted_ = False
        self.svc_ = model
        self.fcnn_ = None
        self.classes_ = None

    
    def fit(self, X=None, y=None, X_calib=None, y_calib=None):
        """
        Обучение SVC и калибрующей FCNN
        
        Параметры:
        -----------
        X : array-like, shape (n_samples, n_features)
            Обучающие данные для SVC
        y : array-like, shape (n_samples,)
            Целевые значения для SVC
        X_calib : array-like, optional
            Данные для калибровки (если None, используется X)
        y_calib : array-like, optional
            Целевые значения для калибровки (если None, используется y)
        """
        if X is not None and y is not None:
            X, y = check_X_y(X, y)
            self.classes_ = np.unique(y)
        
        # 1. Обучаем SVC
        self.svc_ = SVC(**self.model_config)
        self.svc_.fit(X, y)
        
        # 2. Получаем решения SVC для калибровочного набора
        if X_calib is None:
            X_calib, y_calib = X, y
        else:
            X_calib, y_calib = check_X_y(X_calib, y_calib)
            
        svc_output = self.svc_.decision_function(X_calib)
        
        # 3. Обучаем FCNN для калибровки
        self._train_fcnn(svc_output, y_calib)
        
        return self

    def _train_fcnn(self, svc_output, y):
        """Обучение калибрующей FCNN"""
        n_features = svc_output.shape[1] if len(svc_output.shape) > 1 else 1
        
        self.fcnn_ = nn.Sequential(
            nn.Linear(n_features, self.fcnn_hidden_size),
            nn.GELU(),
            nn.Linear(self.fcnn_hidden_size, 1),
        )
        weight = len(y[y == 0]) / len(y[y == 1])
        optimizer = torch.optim.Adam(self.fcnn_.parameters(), lr=self.fcnn_lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        
        X_tensor = torch.FloatTensor(svc_output).unsqueeze(1) if n_features == 1 else torch.FloatTensor(svc_output)
        y_tensor = torch.FloatTensor((y == self.classes_[1]).astype(np.float32))
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.fcnn_epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.fcnn_(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        check_array(X)
        with torch.no_grad():
            svc_output = self.svc_.decision_function(X)
            svc_tensor = torch.FloatTensor(svc_output).unsqueeze(1) if len(svc_output.shape) == 1 else torch.FloatTensor(svc_output)
            proba_positive = torch.sigmoid(self.fcnn_(svc_tensor).flatten()).numpy()
            
        proba = np.zeros((len(X), 2))
        proba[:, 0] = 1 - proba_positive
        proba[:, 1] = proba_positive
        return proba

    def predict(self, X):
        """Предсказание классов"""
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] > 0.5).astype(int)]

    def is_fitted(self):
        return self.is_fitted_
