import numpy as np

def EuclidianNorm(Hist, Histo_proto):
    return np.sqrt(np.sum((Hist - Histo_proto)**2,axis=1))

def NormalizedNorm(Hist,Histo_proto):
    summation = np.sum(Histo_proto,axis=1)
    return np.sqrt(np.sum((Hist/np.sum(Hist) - Histo_proto/summation[:,None])**2,axis=1))

def BattachaNorm(Hist,Histo_proto):
    summation = np.sum(Histo_proto,axis = 1)
    return -np.log(np.sum(np.sqrt(np.multiply(Histo_proto/summation[:,None],Hist/np.sum(Hist))),axis=1))
