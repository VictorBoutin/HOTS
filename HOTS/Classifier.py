import numpy as np

from HOTS.Tools import EuclidianNorm, NormalizedNorm, BattachaNorm
from HOTS.Tools import GenerateHistogram

class Classifier(object):
    def __init__(self, event_train, event_test, TrainingLabel, GroundTruth):
        self.event_train = event_train
        self.event_test = event_test
        self.TrainingLabel = TrainingLabel
        self.GroundTruth = GroundTruth
        #   print(training_label)
    def HistogramDistance(self):#(histo_to_classify,proto_histo,proto_label_list=None):
        histo_train, pola_train = GenerateHistogram(self.event_train)
        histo_test, pola_test = GenerateHistogram(self.event_test)

        for idx,each_histo in enumerate(histo_test):
            Euclidian_distance = EuclidianNorm(each_histo,histo_train)
            Normalize_distance = NormalizedNorm(each_histo,histo_train)
            Battacha_distance = BattachaNorm(each_histo,histo_train)
            #KL_distance = KL(each_histo,proto_histo)

            min_dist_eucli = np.argmin(Euclidian_distance)
            min_dist_norm = np.argmin(Normalize_distance)
            min_dist_battacha = np.argmin(Battacha_distance)
            #min_dist_KL= np.argmin(KL_distance)


            min_label_eucli = self.TrainingLabel[min_dist_eucli][0]
            min_label_norm = self.TrainingLabel[min_dist_norm][0]
            min_label_battacha = self.TrainingLabel[min_dist_battacha][0]
            to_return = (min_label_eucli, min_label_norm, min_label_battacha)#, min_label_KL)

            if idx != 0:
                output_eucli = np.vstack((output_eucli,to_return[0]))
                output_norm = np.vstack((output_norm,to_return[1]))
                output_battacha = np.vstack((output_battacha,to_return[2]))
            else :
                output_eucli = to_return[0]
                output_norm = to_return[1]
                output_battacha = to_return[2]


        accu = self.Accuracy([output_eucli, output_norm, output_battacha], self.GroundTruth)

        print('Classification Accuracy : \n Euclidian Norm {0:.2f}% \n Normalized Norm {1:.2f}% \
            \n BattachaNorm {2:.2f}%'.format(accu[0]*100,accu[1]*100, accu[2]*100))

        return (output_eucli, output_norm, output_battacha,accu)#, output_KL)

    def Accuracy(self,list_of_classified,GroundTruth):
        if type(list_of_classified) is not list:
            list_of_classified = [list_of_classified]

        nb_sample = list_of_classified[0].shape[0]
        list_of_accuracy = list()
        for idx,each_norm in enumerate(list_of_classified):
            list_of_accuracy.append((np.sum(list_of_classified[idx]==GroundTruth)/nb_sample))

        return list_of_accuracy
