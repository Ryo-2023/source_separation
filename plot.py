import torch
import matplotlib.pyplot as plt
import numpy as np

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
    
def plot_divergence(values):
    plt.figure()
    plt.plot(values)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Divergence')
    plt.title('Divergence over Iterations')
    plt.grid(True)
    plt.show()