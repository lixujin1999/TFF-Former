import os

class Config(object):
    def __init__(self):
        #all
        self.N = 1
        self.p = 0.45
        self.d_model = 128 #128
        self.hidden = self.d_model * 4
        self.n_heads= 4 #feature % n_heads**2 ==0
        self.time = 2
        
        #raw
        self.C = 64
        self.T = 500
        self.patchsize = 16
        
        self.H = self.C // self.patchsize
        self.W = self.T // self.patchsize
        
        #frequence
        self.Cf = 64
        self.Tf = self.T
        self.patchsizefh = 64
        self.patchsizefw = 4
        self.fftn = 500
        
        self.Hf = self.Cf // self.patchsizefh
        self.Wf = self.Tf // self.patchsizefw   

        #temporal
        self.Ct = 64
        self.Tt = self.T
        self.patchsizeth = 64
        self.patchsizetw = 4 
        self.Ht = self.Ct // self.patchsizeth
        self.Wt = self.Tt // self.patchsizetw   
        
        self.batchsize = 64
        self.epoch = 80
        self.patience = 300
        self.lr = 1e-3
        self.val = 500
        self.smooth = 0.01
        self.num_class = 40
        self.window = 250 
        self.stride = 250 
        
        
        self.save = os.path.abspath(os.curdir)+'/TFF-Former-SSVEP/bestpath.pkl'
        self.sample = 1    
        
config = Config()
