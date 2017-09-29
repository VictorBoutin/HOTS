import time
import numpy as np


from HOTS.Tools import EuclidianNorm


class Cluster(object):
    def __init__(self, nb_cluster, verbose=0):
        self.nb_cluster = nb_cluster
        self.verbose = verbose
        self.prototype = np.zeros(0)
        #self.area = area

    def test(self,to_print):
        print(to_print)

class CustomKmeans(Cluster):
    def __init__(self,nb_cluster, verbose=0):
        Cluster.__init__(self, nb_cluster, verbose)

    def fit (self,STS, init=None, nb_cycle=1):
        tic = time.time()

        surface = STS.Surface.copy()

        if init is None :
            prototype=surface[:self.nb_cluster,:]
        elif init == 'rdn' :
            idx = np.random.permutation(np.arange(surface.shape[0]))[:self.nb_cluster]
            prototype = surface[idx, :]
        else :
            raise NameError('argument '+str(init)+' is not valid. Only None or rdn are valid')

        for each_cycle in range(nb_cycle):
            nb_proto = np.zeros((self.nb_cluster)).astype(int)
            for idx, Si in enumerate(surface):
                Distance_to_proto = EuclidianNorm(Si, prototype)
                closest_proto_idx = np.argmin(Distance_to_proto)
                pk = nb_proto[closest_proto_idx]
                Ck = prototype[closest_proto_idx,:]
                alpha = 0.01/(1+pk/20000)
                beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))*np.sqrt(np.dot(Ck, Ck)))
                #Ck_t = Ck + alpha*(Si - beta*Ck)
                Ck_t = (1 - alpha*beta) * Ck + alpha*beta*Si
                nb_proto[closest_proto_idx] += 1
                #Ck_t /= np.amax(Ck_t)
                prototype[closest_proto_idx, :] = Ck_t
            tac = time.time()
        self.prototype = prototype
        self.nb_proto = nb_proto
        if self.verbose > 0:
            print('Clustering SpatioTemporal Surface in ------ {0:.2f} s'.format(tac-tic))

        return prototype

    def predict(self, STS, event):
        to_predict = STS.Surface
        if self.prototype is None :
            raise ValueError('Train the Cluster before doing prediction')
        polarity,output_distance = np.zeros(to_predict.shape[0]).astype(int),np.zeros(to_predict.shape[0])

        for idx,surface in enumerate(to_predict):
            Euclidian_distance = EuclidianNorm(surface,self.prototype)
            polarity[idx] = np.argmin(Euclidian_distance)
            output_distance[idx] = np.amin(Euclidian_distance)
        event_output = event.copy()
        event_output.polarity = polarity
        event_output.ListPolarities= list(np.arange(self.nb_cluster))
        return event_output, output_distance
