import numpy as np

class ILRMA():
    def __init__(self,X):
        self.X = X  # [F,T,M]
        
        self.F = X.shape[0]  # 周波数ビン数
        self.T = X.shape[1]  # フレーム数
        self.M = X.shape[2]  # マイク数
        self.N = self.M      # 音源数 = マイク数
        
        self.K = 4  # 基底数
        self.eps = 1e-8
        self.method = "ILRMA"
    
    def initialize(self):
        self.V_FMM = np.random.randn(self.F,self.M,self.M)
        self.W_FNM = np.random.randn(self.F,self.M,self.N)
        self.U_NFK = np.random.randn(self.N,self.F,self.K)
        self.H_NKT = np.random.randn(self.N,self.K,self.T)
        
        self.lambda_NFT = np.einsum("nfk,nkt->nft",self.U_NFK,self.H_NKT)
        
    def update_W(self):
        # IVA part
        XX = np.einsum("ftm,ftp->ftmp",self.X,np.conj(self.X))
        V_FNMM = np.einsum("ftmp,nft->fnmp",XX,1/(self.lambda_NFT+ self.eps)) / self.T
        
        WV_FNN = np.einsum("fij,fnjk->fnik",self.W_FNM,V_FNMM)
        I_FNN = np.broadcast_to(np.eye(self.N,dtype=np.complex128),(self.F,self.N,self.N))
        inv_WV = np.linalg.solve(WV_FNN,I_FNN)
        temp_W_FNM = np.swapaxes(inv_WV,1,2)
        
        norm = np.einsum("fnm,fnmk,fnk->fn",np.conj(temp_W_FNM),V_FNMM,temp_W_FNM)
        self.W_FNM = np.conj(temp_W_FNM / np.sqrt(norm + self.eps)[...,None])
        
        
        
        
        
        
        
        

    
        
