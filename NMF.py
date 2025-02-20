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
"""

class NMF():
    def __init__(self,N,n_fft,K,T,X,threshold=1e-5):
        self.N = N                 # 音源数
        self.n_fft = n_fft         # 窓サイズ
        self.F_bin = n_fft//2 + 1  # 周波数ビン数
        self.K = K                 # 基底数
        self.T = T                 # フレーム数
        
        self.X = X                      # 入力スペクトログラム
        self.W = torch.rand(N,self.F_bin,K)  # 分離フィルタ
        self.H = torch.rand(N,T,K)      # アクティベーション
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # 推定スペクトログラム, W*H
        
        self.eps = 1e-10       # 0割防止のための微小値
        self.threshold = 1e-5  # 収束判定の閾値
        
    @staticmethod
    def KL_divergence(X,Y):
        KL = torch.einsum("ft ->",X * torch.log(X) - X*torch.log(Y) - X + Y)
        return KL
    
    def update_W(self):
        new_W = torch.einsum("nft,ntk -> nfk",self.X/self.Y,self.H)
        ones = torch.ones_like(new_W)
        reshaped_H = torch.einsum("nfk,ntk->nfk",ones,self.H)
        new_W = torch.einsum("nfk,nfk->nfk",new_W / reshaped_H,self.W)
        self.W = new_W  # Wの更新
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # Yの更新
    
    def update_H(self):
        new_H = torch.einsum("nft,nfk->ntk",self.X/self.Y,self.W)
        ones = torch.ones_like(new_H)
        reshaped_W = torch.einsum("nft,nfk->ntk",self.X/self.Y,ones)
        new_H = torch.einsum("ntk,ntk->ntk",new_H / reshaped_W,self.H)
        self.H = new_H  # Hの更新
        self.Y = torch.einsum("nfk,ntk->nft",self.W,self.H)  # Yの更新
    
    def run_NMF(self):
        while True:
            self.update_W()
            self.update_H()
            KL = self.KL_divergence(self.X,self.Y)
            if KL < self.threshold:
                break
        return self.Y
    
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
    plt.yticks(np.arange(0, num_freq_bins, step=int(2000 / (sr / 2) * num_freq_bins)), np.round(freq_bins[::int(2000 / (sr / 2) * num_freq_bins)]).astype(int))    
    plt.tight_layout()
    plt.show()
    
def main():
    # torchの設定
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device
    
    data_path = ["wav1.wav","wav2.wav"]
    
    # ハイパラの設定
    N = 1
    n_fft = 1024
    F_bin = n_fft // 2 + 1
    K = 100
    
    stft = STFT(data_path,n_fft=n_fft)
    X,phase = stft.stft()
    print("X.shape:",X.shape)
    sr = stft.sr
    
    # NMFの実行
    T = X.shape[1]
    print("N,F_bin,K,T:",N,F_bin,K,T)

    nmf = NMF(N,n_fft,K,T,X)
    Y = nmf.run_NMF()
    
    # 分離されたスペクトログラムの表示
    plot_spectrogram(Y[0], sr, title="Separated Spectrogram 1")
    plot_spectrogram(Y[1], sr, title="Separated Spectrogram 2")
    
if __name__ == "__main__":
    main()
