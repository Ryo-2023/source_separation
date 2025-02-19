import torch
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

"""
M : マイク数
N : フレームサイズ
L_shift : シフトサイズ
"""

class STFT():
    def __init__(self,source,N):
        self.source = source
        self.M = source.shape[1]
        self.N = N
        self.overlap = self.N // 4
        self.shift = self.N - self.overlap
        
    def hanning(self,n):
        return 0.5 - 0.5 * torch.cos(2*torch.tensor(np.pi)*n/(self.N-1))
    
    def hamming(self,n):
        return 0.54 - 0.46 * torch.cos(2*torch.tensor(np.pi)*n/(self.N-1))
    
    def stft(self):
        window = self.hanning(torch.arange(self.N))
        
        l = self.source.shape[0]  # 音声信号の長さ
        
        F = (l - self.N) // self.shift + 1  # 周波数ビン数
        
        # 結果を格納するテンソル
        print("M,F,N",self.M,F,self.N)
        X = torch.zeros((self.M,F,self.N))
        
        for m in range(self.M):
            # マイクごとにSTFTを計算
            s = self.source[m]
            for f in range(F):
                start_idx = f * self.shift
                end_idx = start_idx + self.N
                frame = s[start_idx:end_idx] * window
                X[m,f] = torch.fft.fft(frame)
                
        return X
    
def plot_spectrogram(X, N, shift, sample_rate):
    # Xの形状は (M, F, N) なので、まずは1つのマイクのデータを取り出します
    X_mic = X[0]  # ここでは最初のマイクのデータを使用
    
    # 振幅スペクトルを計算
    magnitude = torch.abs(X_mic).numpy()
    
    # 時間軸と周波数軸を計算
    time_bins = np.arange(magnitude.shape[0]) * shift / sample_rate
    freq_bins = np.fft.fftfreq(N, d=1/sample_rate)
    
    # 正の周波数成分のみを使用
    magnitude = magnitude[:, :N//2]
    freq_bins = freq_bins[:N//2]
    
    # スペクトログラムをプロット
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_bins, freq_bins, 20 * np.log10(magnitude.T), shading='gouraud')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.show()
    
wav1_sample_rate,wav1_data = wavfile.read("C:/Users/bfe-l/sound_separation/wav1.wav")
print(wav1_sample_rate)
X = STFT(torch.tensor(wav1_data),256).stft()
plot_spectrogram(X, 256, 64, 16000)

