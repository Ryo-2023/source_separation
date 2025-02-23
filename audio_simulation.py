import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import soundfile as sf
import os
import librosa
from plot import plot_spectrogram

def simulation(path,fs=16000):
    N = 2  # 音源数
    M = 4  # マイク数
    
    # ルームサイズ
    room_dim = [5,4,3]
    
    # 部屋の作成
    room = pra.ShoeBox(room_dim, fs=16000, max_order=3, absorption=0.3)
    
    # マイクの配置, shape: (3,M)
    mic_positions = np.c_[
        [0.1,0.1,1.5],
        [0.1,3.9,1.5],
        [4.9,0.1,1.5],
        [4.9,3.9,1.5]
    ]
    print("mic_positions.shape:",mic_positions.shape)
    
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    
    # 音源の配置
    source_positions = [
        [2.5,1,1.5],
        [2.5,1,1.5],
    ]
        
    min_length = float("inf")
    source_signals = []
    for i,pos in enumerate(source_positions):
        data, sr = sf.read(path[i])
        # サンプリングレートが異なる場合はリサンプリング
        if sr != fs:
            data = librosa.resample(data,orig_sr=sr,target_sr=fs)
        source_signals.append(data)
        min_length = min(min_length,len(data))
    
    # 音源の長さを揃える
    for i, data in enumerate(source_signals):
        source_signals[i] = data[:min_length]
        room.add_source(source_positions[i],signal=source_signals[i])
        
    # 三つ目の音源に白色雑音を追加
    noise_amp = 1e-2
    white_noise = np.random.normal(0,noise_amp,min_length)
    room.add_source([2.5,2,1.5],signal=white_noise)
        
    # シミュレーション実行
    room.compute_rir()
    room.simulate()
    
    # 出力データ
    simulated_data = room.mic_array.signals  # shape: (M,T)
    
    # wavファイルとして保存
    is_Save = True
    if is_Save:
        output_path = "/Users/onoryousuke/sound_separation/data/simulated_data.wav"
        sf.write(output_path,simulated_data.T,fs)
        print("Saved simulated data to ",output_path)
        
    return simulated_data, fs
    
def main():
    # パスの設定
    path = ["/Users/onoryousuke/sound_separation/data/wav1.wav","/Users/onoryousuke/sound_separation/data/wav2.wav"]
    
    # シミュレーション
    simulated_data, fs = simulation(path)
    
    # プロット
    for i in range(simulated_data.shape[0]):
        X = librosa.stft(simulated_data[i],n_fft=1024,hop_length=256)
        plot_spectrogram(torch.tensor(X),fs,title="Simulated Data " + str(i+1))
        
if __name__ == "__main__":
    main()
        
    
    