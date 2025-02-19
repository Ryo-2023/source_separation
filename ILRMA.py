import torch
from scipy.io import wavfile
from tqdm import tqdm

"""
N : 音源数
F : 周波数ビン数
T : フレーム数
"""

class ILRMA():
    def __init__(self,N,F,T)
        self.N = N
        self.F = F
        self.T = T