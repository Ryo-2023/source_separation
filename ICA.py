import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

class ICA():
    def __init__(self,X,n=2,learning_rate=1e-3,max_iter=100):
        self.X = X      # [D,N]
        
        self.F = X.shape[0]
        self.T = X.shape[1]
        
        self.N = n  # 分離信号数
        
        self.W = torch.rand(self.F,self.N)
        self.S = torch.rand(self.F,self.T)  # 分離信号
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def centering(self):
        # 中心化
        X_centered = self.X - torch.mean(self.X,dim=1,keepdim=True)
        return X_centered
    
    def update_W(self):
        # 損失勾配の計算
        
    
x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float32)
ica = ICA(x)
x2 = ica.centering()
print(x2)
        