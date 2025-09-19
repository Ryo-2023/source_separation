import numpy as np
import torch

class ILRMA():
    def __init__(self,X):
        self.X = X  # [F,T,M]
        
        self.F = X.shape[0]  # 周波数ビン数
        self.T = X.shape[1]  # フレーム数
        self.M = X.shape[2]  # マイク数
        self.N = self.M      # 音源数 = マイク数
        
        self.K = 4  # 基底数
        self.method = "ILRMA"
    
    def initialize(self):
        self.W_

    
        
