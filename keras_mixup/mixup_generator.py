import numpy as np
from keras.utils import to_categorical
import math

class TensorGenerator():
    
    def __init__(self, X_train, y_train, batch_size=32, shuffle=True, num_classes=-1):
        self.X_train = X_train
        self.y_train = y_train if num_classes==-1 else to_categorical(y_train, num_classes)
        self.batch_size = batch_size
        self.shuffle = shuffle       
        
    def __call__(self):
        bs = self.batch_size
        indexes1 = np.arange(len(self.X_train))
        while True:
            if self.shuffle: np.random.shuffle(indexes1)
            for i in range(0, len(self.X_train), bs):
                yield self.X_train[indexes1[i:i+bs]], self.y_train[indexes1[i:i+bs]]

    def __len__(self):
        return math.ceil(len(self.X_train)/self.batch_size)
                

class VAEGenerator():
    
    def __init__(self, filepath='models/vae.{}.{}.npz.npy', batch_size=32):
        self.filepath = filepath
        self.batch_size = batch_size
        X, Y = np.load(filepath.format(0,'X')), np.load(filepath.format(0,'Y'))
        self.length = math.ceil(len(X)/batch_size)
        
    def __call__(self):
        bs = self.batch_size
        while True:
            for idx in range(40):
                X, Y = np.load(self.filepath.format(idx,'X')), np.load(self.filepath.format(idx,'Y'))
                for i in range(0, len(X), bs):
                    yield np.expand_dims(X[i:i+bs], -1), Y[i:i+bs]
                    
    def __len__(self):
        return self.length
                
class MixTensorGenerator():
    
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, num_classes=-1):      
        self.gen1 = TensorGenerator(X_train, y_train, batch_size, shuffle, num_classes)
        self.gen2 = TensorGenerator(X_train, y_train, batch_size, shuffle, num_classes)
        self.batch_size = batch_size
        self.alpha = alpha
        
        
    def __call__(self):
        for (X1,y1),(X2,y2) in zip(self.gen1(), self.gen2()):
            if self.alpha==0: yield X1,y1
            else:
                l = np.random.beta(self.alpha, self.alpha, len(X1))
                X_l = l.reshape(len(X1), 1, 1, 1)
                y_l = l.reshape(len(X1), 1)
                yield X1 * X_l + X2 * (1 - X_l), y1 * y_l + y2 * (1 - y_l)

    def __len__(self):
        return len(self.gen1)
            
         
                
class VAEMixGenerator():
    
    def __init__(self,  X_train, y_train, enc, dec, batch_size=32, alpha=0.2, shuffle=True, shuffle2=True, num_classes=-1):   
        self.enc = enc
        self.dec = dec   
        self.gen1 = TensorGenerator(X_train, y_train, batch_size, shuffle, num_classes)
        self.gen2 = TensorGenerator(X_train, y_train, batch_size, shuffle2, num_classes)
        self.batch_size = batch_size
        self.alpha = alpha
        
        
    def __call__(self):
        for (X1,y1),(X2,y2) in zip(self.gen1(), self.gen2()):
            if self.alpha==0: yield X1,y1
            else:
                l = np.random.beta(self.alpha, self.alpha, len(X1))
                X_l = l.reshape(len(X1), 1)
                y_l = l.reshape(len(X1), 1)
                E1 = self.enc.predict(X1.reshape(len(X1),-1))
                E2 = self.enc.predict(X2.reshape(len(X2),-1))
                #print(E1.shape, X_l.shape)
                E12 = E1 * X_l + E2 * (1 - X_l)
                out = self.dec.predict(E12)
                #print("out", out.shape)
                yield  out.reshape(len(X1), 28, 28), y1 * y_l + y2 * (1 - y_l)

    def __len__(self):
        return len(self.gen1)
            
         
