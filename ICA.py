import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

class ICA():
    def __init__(self,X,learning_rate=1e-3,threshold = 1e-8,max_iter=100):
        self.X = X      # [D,N]
        
        self.D = X.shape[0]
        self.N = X.shape[1] 
        
        self.W = torch.eye(self.D,self.D)
        self.S = torch.rand(self.D,self.N)  # 分離信号
        
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.threshold = threshold
        self.max_iter = max_iter
        
    def centering(self):
        # 中心化
        X_centered = self.X - torch.mean(self.X,dim=0,keepdim=True)
        print("X_centered:",X_centered)
        return X_centered
    
    def whiten(self,X_centered):
        # 白色化
        # 特徴間の相関をなくし、
        cov = X_centered @ X_centered.H / self.N + self.eps
        A,P = torch.linalg.eigh(cov)
        A += self.eps  # 0割防止
        A_inv_sqrt = torch.diag(1.0 / torch.sqrt(A))
        print("A_inv_sqrt:",A_inv_sqrt)
        self.W = A_inv_sqrt * P.H  # [D,D]
        X_whiten = self.W @ X_centered  #[D,D]@[D,N] -> [D,N]
        print("X_whiten:",X_whiten)
        return X_whiten
    
    def update_W(self):
        # 損失勾配の計算
        S = self.W @ self.X
        G = torch.tanh(S)  # [D,N]
        G_prime = 1-G**2
        
        # E[φ(s)s^H] の計算
        E = G @ S.H / self.N  # [D,D]
        
        # Wの更新
        new_W = self.W - self.learning_rate * (E - torch.eye(self.D)) @ self.W
        self.W = new_W / torch.linalg.norm(new_W, axis=1, keepdim=True)  # 正規化
        print("W:",self.W)
        self.W = new_W
        
        # Sの更新
        new_S = self.W @ self.X
        self.S = new_S
    
    def run_pca(self):
        X_centered = self.centering()
        X_whiten = self.whiten(X_centered)
        
        with tqdm(total=self.max_iter,desc="ICA",leave=False) as pbar:
            for i in range(self.max_iter):
                old_S = self.S
                self.update_W()
                new_S = self.S
                diff = torch.sum((new_S-old_S)**2)
                print("diff:",diff)
                if diff < self.threshold:
                    break
                pbar.update(1)
        
        self.S = self.W @ self.X
        return self.S
        
def main():
    x = torch.tensor([[2.5,7,3],[6,2,3],[4,5,8]],dtype=torch.float32)
    torch.manual_seed(0)
    x2 = torch.randn(3,10)
    ica = ICA(x2)
    S = ica.run_pca()
    print("S:",S)
    
if __name__ == "__main__":
    main()
        