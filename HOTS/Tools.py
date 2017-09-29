import numpy as np
import pickle

def EuclidianNorm(Hist, Histo_proto):
    return np.sqrt(np.sum((Hist - Histo_proto)**2,axis=1))

def NormalizedNorm(Hist,Histo_proto):
    summation = np.sum(Histo_proto,axis=1)
    return np.sqrt(np.sum((Hist/np.sum(Hist) - Histo_proto/summation[:,None])**2,axis=1))

def BattachaNorm(Hist,Histo_proto):
    summation = np.sum(Histo_proto,axis = 1)
    return -np.log(np.sum(np.sqrt(np.multiply(Histo_proto/summation[:,None],Hist/np.sum(Hist))),axis=1))

def SaveObject(obj,filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def LoadObject(filename):
    with open(filename, 'rb') as file:
        Clust = pickle.load(file)
    return Clust
#def Load(filename):

def GenerateHistogram(event):
    #if len(self.output.ChangeIdx) == 1:
    last_change=0
    for idx, each_change in enumerate(event.ChangeIdx):
        freq, pola = np.histogram(event.polarity[last_change:each_change+1],bins=len(event.ListPolarities))
        if idx != 0:
            freq_mat = np.vstack((freq_mat,freq))
        else :
            freq_mat = freq
        last_change=each_change
    return freq_mat, pola
