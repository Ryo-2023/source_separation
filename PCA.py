import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class PCA():
    def __init__(self,X,N,dim=1):
        self.X = X
        self.dim = dim
        self.mean = torch.mean(self.X,dim=self.dim,keepdim=True)
        self.X_standardized = self.standarize()
        self.N = N
        
        self.F = X.shape[0]
        self.T = X.shape[1]
        
        self.W = torch.rand(self.T,self.N)  # W = (w1,w2,...,wN)
        self.S = torch.rand(self.F,self.N)  # S = W*X
        
        self.contribution_ratio = 0
        
        self.eps = 1e-10
        self.threshold = 1e-5
        self.max_iter = 100
        
    def centralize(self):  # 中心化
        X_centered = self.X - self.mean
        return X_centered
        
    def standarize(self):  # 標準化
        x_centered = self.centralize()
        std = torch.std(x_centered, dim=self.dim,keepdim=True)
        X_standardized = x_centered / std
        return X_standardized
    
    def pca(self):
        # 共分散行列を求める
        if self.dim == 1:
            conv = torch.cov(self.X_standardized.T)  # (T,F) -> (T,T) 転地してから共分散行列を求める
        else:
            conv = torch.cov(self.X_standardized)
        
        # 固有値分解
        eighvals, eigvectors = torch.linalg.eigh(conv)
        eighvals = eighvals.real  # 実数部だけ取得
        
        # 固有値の大きい順にソート
        sorted_idx = torch.argsort(eighvals,descending=True)
        sorted_vals = eighvals[sorted_idx]
        sorted_vectors = eigvectors[:,sorted_idx]
        
        # 主成分を取得（固有値の大きい方からN個）
        lamda = sorted_vals[:self.N]
        W = sorted_vectors[:,:self.N]
        self.W = W
        
        # 射影
        S = torch.einsum("tn,ft->fn",self.W,self.X_standardized)
        self.S = S
        
        # 寄与率
        ratio = torch.sum(lamda) / torch.sum(eighvals)
        self.contribution_ratio = ratio
        print(f"Contribution ratio: {ratio}")
        
        return self.S, self.contribution_ratio
    
def generate_test_data():
    # 2次元のデータを生成
    np.random.seed(1)
    torch.manual_seed(1)
    
    # まずは大きめの分散を持つ分布
    mean1 = [1.2, 1.5]
    cov1 = [[0.65, 0.355], [0.3, 0.7]]
    data1 = np.random.multivariate_normal(mean1, cov1, 50)

    # 異なるオフセット＆分散を持つ別の分布
    mean2 = [-2, -0.6]
    cov2 = [[1, 0.2], [0.2, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, 50)
    
    mean3 = [1.1,-1.5]
    cov3 = [[0.1,0.9],[0.2,0.8]]
    data3 = np.random.multivariate_normal(mean3,cov3,50)

    # 結合
    data = np.vstack((data1, data2,data3))  # shape: (100, 2)
    X_noisy = torch.tensor(data, dtype=torch.float32).T  # (2, 100)
    
    # 中心化
    X_noisy = X_noisy - torch.mean(X_noisy,dim=1,keepdim=True)
    
    # 標準化
    X_noisy = X_noisy / torch.std(X_noisy,dim=1,keepdim=True)
    
    # ノイズの追加（分散を大きめに設定）
    noise = torch.randn(X_noisy.shape) * 0.3
    X_noisy =  X_noisy + noise
    
    return X_noisy
        
def main():
    X = generate_test_data()
    N = 3
    dim = 1
    
    pca = PCA(X,N,dim)
    s,r = pca.pca()
    print("s:",s)
    print("r:",r)
    
    # 元のデータと次元削減後のデータをプロット
    plt.scatter(X[0, :], X[1, :], label='Original Data')
    plt.scatter(s[0, :], s[1,:],np.zeros(s.shape[1]), label='PCA Reduced Data')
    
    # 固有ベクトルを矢印としてプロット
    origin = torch.mean(X, dim=1).numpy()
    for i in range(pca.W.shape[1]):
        plt.quiver(origin[0], origin[1], pca.W[0, i], pca.W[1, i], scale=5, scale_units='xy', angles='xy', color='r')
        
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
        
        
    