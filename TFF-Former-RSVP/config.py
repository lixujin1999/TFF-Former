import os

class Config(object):
    def __init__(self):
        #all
        self.N = 1
        self.p = 0.2
        self.d_model = 128 #128
        self.hidden = self.d_model * 4
        self.n_heads= 4 #feature % n_heads**2 ==0
        
        #raw
        self.C = 64
        self.T = 248
        self.patchsize = 16
        
        self.H = self.C // self.patchsize
        self.W = self.T // self.patchsize
        
        #frequence
        self.Cf = 64
        self.Tf = 248
        self.patchsizefh = 64
        self.patchsizefw = 4
        self.fftn = 248
        
        self.Hf = self.Cf // self.patchsizefh
        self.Wf = self.Tf // self.patchsizefw   

        #temporal
        self.Ct = 64
        self.Tt = 248
        self.patchsizeth = 64
        self.patchsizetw = 4
        self.Ht = self.Ct // self.patchsizeth
        self.Wt = self.Tt // self.patchsizetw   
        
        self.batchsize = 64
        self.epoch = 110
        self.patience = 100
        self.lr = 5e-4
        self.val = 2240
        self.smooth = 0.05
        self.num_class = 2
        self.kl = 0.001
        
        self.save = os.path.abspath(os.curdir)+'/TFF-Former-RSVP/bestpath.pkl'
        self.sample = 1    
        
config = Config()
