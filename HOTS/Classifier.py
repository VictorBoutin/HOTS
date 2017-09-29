import numpy as np

from HOTS.Tools import EuclidianNorm, NormalizedNorm, BattachaNorm
from HOTS.Tools import GenerateHistogram

class Classifier(object):
    def __init__(self,event_train,event_test,training_label):
        self.event_train = event_train
        self.event_test = event_test
        self.training_label = training_label
    def HistogramDistance(self):#(histo_to_classify,proto_histo,proto_label_list=None):
        histo_train, pola_train = GenerateHistogram(self.event_train)
        histo_test, pola_test = GenerateHistogram(self.event_test)
        print(histo_test.shape)
        print(histo_train.shape)

        for idx,each_histo in enumerate(histo_test):
            Euclidian_distance = EuclidianNorm(each_histo,histo_train)
            Normalize_distance = NormalizedNorm(each_histo,histo_train)
            Battacha_distance = BattachaNorm(each_histo,histo_train)
            #KL_distance = KL(each_histo,proto_histo)

            min_dist_eucli = np.argmin(Euclidian_distance)
            min_dist_norm = np.argmin(Normalize_distance)
            min_dist_battacha = np.argmin(Battacha_distance)
            #min_dist_KL= np.argmin(KL_distance)

            to_return = (min_dist_eucli, min_dist_norm, min_dist_battacha)#, min_dist_KL)

            min_label_eucli = self.training_label[min_dist_eucli][0]
            min_label_norm = self.training_label[min_dist_norm][0]
            min_label_battacha = self.training_label[min_dist_battacha][0]
            #min_label_KL = proto_label_list[min_dist_KL][0]
            to_return = (min_label_eucli, min_label_norm, min_label_battacha)#, min_label_KL)

            if idx != 0:
                output_eucli = np.vstack((output_eucli,to_return[0]))
                output_norm = np.vstack((output_norm,to_return[1]))
                output_battacha = np.vstack((output_battacha,to_return[2]))
            else :
                output_eucli = to_return[0]
                output_norm = to_return[1]
                output_battacha = to_return[2]
                #output_KL = to_return[3]

        return (output_eucli, output_norm, output_battacha)#, output_KL)
