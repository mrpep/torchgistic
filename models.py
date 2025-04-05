import torch
from dataclasses import dataclass
import numpy as np

@dataclass
class TorchLogisticRegressor:
    lr: float = 0.1
    l1_w: float = 0.0
    l2_w: float = 1.0
    max_iterations: int = 1000
    tolerance: float = 1e-4
    intercept: bool = True
    device: str = 'cuda:0'

    def fit(self, x, y):
        if y.ndim == 1:
            num_classes = len(np.unique(y))
        else:
            num_classes = y.shape[-1]
        #x = (x - np.mean(x, axis=0, keepdims=True))/(np.std(x, axis=0, keepdims=True)+1e-6)
        self.model = torch.nn.Linear(x.shape[-1], num_classes, bias=self.intercept, dtype=torch.float64, device=self.device)
        optimizer = torch.optim.LBFGS(self.model.parameters(), 
                                      lr=self.lr, 
                                      history_size=100, 
                                      max_iter=20, 
                                      line_search_fn='strong_wolfe')
        criterion = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            x_ = torch.from_numpy(x).to(dtype=torch.float64, device=self.device)
            y_ = torch.from_numpy(y).to(device=self.device)
            outputs = self.model(x_)
            loss = criterion(outputs, y_)

            l1_penalty = sum(p.abs().sum() for p in self.model.parameters())/(x_.shape[-1]*num_classes)
            l2_penalty = sum(p.pow(2.0).sum() for p in self.model.parameters())/(x_.shape[-1]*num_classes)
            loss += self.l1_w * l1_penalty + self.l2_w * l2_penalty  # Apply regularization
            
            loss.backward()
            return loss

        last_loss = np.inf
        i = 0
        while i < self.max_iterations:
            loss = optimizer.step(closure)
            if (last_loss - loss) < self.tolerance:
                break
            i+=1
            last_loss = loss
        if i == self.max_iterations:
            print(f'Warning: model did not converge after running {self.max_iterations} iterations. Consider increasing max_iterations.')

    def predict(self, x):
        return self.model(torch.from_numpy(x).to(dtype=torch.float64, device=self.device)).detach().cpu().numpy()