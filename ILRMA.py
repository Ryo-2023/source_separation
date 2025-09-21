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
        self.W_FNM = np.tile(np.eye(self.M,dtype=np.complex128)[None,:,:],(self.F,1,1))
        
        rng = np.random.default_rng()
        self.U_NFK = rng.random((self.N,self.F,self.K)) + self.eps
        self.H_NKT = rng.random((self.N,self.K,self.T)) + self.eps
        
        self.lambda_NFT = np.einsum("nfk,nkt->nft",self.U_NFK,self.H_NKT)
        
    def update_W(self):
        XX_FTMM  = np.einsum('ftm,ftp->ftmp', self.X, np.conj(self.X))
        V_FNMM   = np.einsum('ftmp,nft->fnmp', XX_FTMM, 1.0/(self.lambda_NFT + self.eps)) / self.T
        for n in range(self.N):
            A_FNMM = np.einsum('fij,fjk->fik', self.W_FNM, V_FNMM[:, n])
            e_FM   = np.zeros((self.F, self.M), dtype=np.complex128)
            e_FM[:, n] = 1.0
            w_FM   = np.linalg.solve(A_FNMM, e_FM)
            denom  = np.einsum('fm,fmn,fn->f', np.conj(w_FM), V_FNMM[:, n], w_FM).real + self.eps
            w_FM   = w_FM / np.sqrt(denom)[:, None]
            self.W_FNM[:, n, :] = np.conj(w_FM)

                                          

            
    
    
    
        
        
        
        
        
        
        
        

    
        
