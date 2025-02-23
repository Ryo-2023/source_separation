import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ICA():
    def __init__(self,X):
        self.X = X      # [N,F,T]
        
        self.N = X.shape[0]
        self.F = X.shape[1]
        self.T = X.shape[2]
        
        self.W = 
