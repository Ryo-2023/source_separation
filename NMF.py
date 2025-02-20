import torch
import torch.nn.functional as NN_F
from scipy.io import wavfile
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

from mix_data import STFT

"""
N : 音源数
F : 周波数ビン数
T : フレーム数
K : 基底数
"""

class NMF():
    def __init__(self,N,n_fft,K,T,X,threshold=1e-5,max_iter=100):
        self.N = N                 # 音源数
        self.n_fft = n_fft         # 窓サイズ
        self.F_bin = n_fft//2 + 1  # 周波数ビン数
        self.K = K                 # 基底数
        self.T = T                 # フレーム数
        
        self.X = X                           # 入力スペクトログラム  [N,F,T]
        self.W = torch.rand(N,self.F_bin,K)  # 分離フィルタ
        self.H = torch.rand(N,T,K)           # アクティベーション
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # 推定スペクトログラム, W*H, [N,F,T]
        
        self.eps = 1e-10            # 0割防止のための微小値
        self.threshold = 1e-5       # 収束判定の閾値
        self.max_iter = max_iter    # 最大イテレーション数
        
    @staticmethod
    def KL_divergence(X,Y):
        KL = torch.einsum("nft->n",X * torch.log(X) - X*torch.log(Y) - X + Y)
        return KL
    
    @staticmethod
    def IS_divergence(X,Y):
        IS = torch.einsum("nft->n",X/Y - torch.log(X/Y) -1)
        return IS
    
    def update_W(self):
        # 分子
        X_div_Y2 = self.X / (self.Y + self.eps)**2
        num = torch.einsum("nft,ntk->nfk",X_div_Y2,self.H)
        
        # 分母
        ones = torch.ones_like(self.Y)
        den = torch.einsum("nft,ntk->nfk",ones/(self.Y+self.eps),self.H)
        
        new_W = torch.sqrt(num/den) * self.W
        
        self.W = new_W  # Wの更新
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # Yの更新
        
    def update_H(self):
        # 分子
        X_div_Y2 = self.X / (self.Y + self.eps)**2
        num = torch.einsum("nft,nfk->ntk",X_div_Y2,self.W)
        
        # 分子
        ones = torch.ones_like(self.Y)
        den = torch.einsum("nft,nfk->ntk",ones/(self.Y+self.eps),self.W)
        
        new_H = torch.sqrt(num/den) * self.H
        
        self.H = new_H  # Hの更新
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # Yの更新
        
    """
    # KLダイバージェンスの場合
    def update_W(self):
        new_W = torch.einsum("nft,ntk -> nfk",self.X/self.Y,self.H)
        ones = torch.ones(self.N,self.F_bin,self.T)
        reshaped_H = torch.einsum("nft,ntk->nfk",ones,self.H)
        new_W = torch.einsum("nfk,nfk->nfk",new_W / reshaped_H,self.W)
        self.W = new_W  # Wの更新
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # Yの更新
    
    def update_H(self):
        new_H = torch.einsum("nft,nfk->ntk",self.X/self.Y,self.W)
        ones = torch.ones(self.N,self.F_bin,self.T)
        reshaped_W = torch.einsum("nft,nfk->ntk",ones,self.W)
        new_H = torch.einsum("ntk,ntk->ntk",new_H / reshaped_W,self.H)
        self.H = new_H  # Hの更新
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # Yの更新
    """
    
    def run_NMF(self):
        with tqdm(total=self.max_iter, desc="NMF",leave=False) as pbar:
            IS_list = []
            for _ in range(self.max_iter):
                self.update_W()
                self.update_H()
                IS = self.IS_divergence(self.X,self.Y)
                IS_list.append(IS)
                if IS < self.threshold:
                    break
                pbar.update(1)
            return self.Y, IS_list
    
def add_noise(data,threshold=1e-3):
    noisy_data = data.clone()
    noise = torch.rand(data.shape)*2*threshold - threshold
    return noisy_data + noise

def plot_spectrogram(X,sr, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.imshow(20 * torch.log10(torch.abs(X) + 1e-10).numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    # 縦軸のラベルを周波数に設定
    num_freq_bins = X.shape[0]
    freq_bins = np.linspace(0, int(sr / 2), num_freq_bins)
    tick_freqs = np.array(np.arange(0, int(sr / 2) + 1, 2000),dtype=int)
    tick_positions = np.round(tick_freqs / (float(sr) / 2) * (num_freq_bins - 1)).astype(int)
    plt.yticks(tick_positions, tick_freqs)
    
    plt.tight_layout()
    plt.show()
    
def plot_divergence(KL_values):
    plt.figure()
    plt.plot(KL_values)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Divergence')
    plt.title('Divergence over Iterations')
    plt.grid(True)
    plt.show()

    
def main():
    # torchの設定
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device
    
    data_path = ["wav1.wav","wav2.wav"]
    
    # ハイパラの設定
    n_fft = 1024
    F_bin = n_fft // 2 + 1
    K = 100
    
    stft = STFT(data_path,n_fft=n_fft)
    source_amp, source_phase, X_amp, X_phase = stft.run_stft()
    sr = stft.sr.item()  # サンプリング周波数

    # X_amp を (N,F,T) に変換
    if X_amp.dim() == 2:
        X_amp = X_amp.unsqueeze(0)
    
    # 分離前のスペクトログラムの表示
    """
    for i in range(len(source_amp)):
        plot_spectrogram(source_amp[i], sr, title="Source Spectrogram " + str(i+1))
    """
    plot_spectrogram(X_amp[0], sr, title="Mixed Spectrogram")
    
    # NMFの実行
    N = X_amp.shape[0]
    T = X_amp.shape[2]
    max_iter = 10000

    nmf = NMF(N,n_fft,K,T,X_amp,max_iter=max_iter)
    Y,IS_data = nmf.run_NMF()
    
    # 分離されたスペクトログラムの表示
    plot_spectrogram(Y[0], sr, title="Separated Spectrogram 1")
    
    plot_divergence(IS_data)
    
if __name__ == "__main__":
    main()
