import time
import numpy as np
import pandas as pd

from HOTS.Tools import EuclidianNorm


class Cluster(object):
    def __init__(self, nb_cluster, record_each=0 ,verbose=0):
        self.nb_cluster = nb_cluster
        self.verbose = verbose
        self.prototype = np.zeros(0)
        self.record_each = record_each
        if self.record_each>0:
            self.record = pd.DataFrame()
        #self.area = area

    def test(self,to_print):
        print(to_print)

class CustomKmeans(Cluster):
    def __init__(self,nb_cluster, record_each=0, verbose=0):
        Cluster.__init__(self, nb_cluster,record_each, verbose)

    def fit (self,STS, init=None, NbCycle=1):
        tic = time.time()

        surface = STS.Surface.copy()

        if init is None :
            self.prototype=surface[:self.nb_cluster,:]
        elif init == 'rdn' :
            idx = np.random.permutation(np.arange(surface.shape[0]))[:self.nb_cluster]
            self.prototype = surface[idx, :]
        else :
            raise NameError('argument '+str(init)+' is not valid. Only None or rdn are valid')
        idx_global=0
        nb_proto = np.zeros((self.nb_cluster)).astype(int)
        for each_cycle in range(NbCycle):
            nb_proto = np.zeros((self.nb_cluster)).astype(int)
            for idx, Si in enumerate(surface):
                Distance_to_proto = EuclidianNorm(Si, self.prototype)
                closest_proto_idx = np.argmin(Distance_to_proto)
                pk = nb_proto[closest_proto_idx]
                Ck = self.prototype[closest_proto_idx,:]
                alpha = 0.01/(1+pk/20000)
                beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))*np.sqrt(np.dot(Ck, Ck)))
                Ck_t = Ck + alpha*(Si - beta*Ck)
                #Ck_t = (1 - alpha*beta) * Ck + alpha*beta*Si
                nb_proto[closest_proto_idx] += 1
                #Ck_t /= np.amax(Ck_t)
                self.prototype[closest_proto_idx, :] = Ck_t

                if self.record_each != 0 :
                    if idx_global % int(self.record_each) == 0 :
                        output_distance = self.predict(STS,SurfaceFilter=1000)
                        error = np.mean(output_distance)
                        record_one = pd.DataFrame([{'error':error}],
                                            index=[idx_global])
                        self.record = pd.concat([self.record, record_one])

                idx_global += 1
            tac = time.time()
        #self.prototype = prototype
        self.nb_proto = nb_proto
        if self.verbose > 0:
            print('Clustering SpatioTemporal Surface in ------ {0:.2f} s'.format(tac-tic))

        return self.prototype

    def predict(self, STS, event=None, SurfaceFilter=None):
        if SurfaceFilter == None:
            to_predict = STS.Surface
        else :
            random_selection = np.random.permutation(np.arange(STS.Surface.shape[0]))[:SurfaceFilter]
            to_predict = STS.Surface[random_selection]

        if self.prototype is None :
            raise ValueError('Train the Cluster before doing prediction')
        polarity,output_distance = np.zeros(to_predict.shape[0]).astype(int),np.zeros(to_predict.shape[0])

        for idx,surface in enumerate(to_predict):
            Euclidian_distance = EuclidianNorm(surface,self.prototype)
            polarity[idx] = np.argmin(Euclidian_distance)
            output_distance[idx] = np.amin(Euclidian_distance)
        if event is not None :
            event_output = event.copy()
            event_output.polarity = polarity
            event_output.ListPolarities= list(np.arange(self.nb_cluster))
            return event_output, output_distance
        else :
            return output_distance


class KmeansMaro(Cluster):
    def __init__(self,nb_cluster, record_each=0, verbose=0):
        Cluster.__init__(self, nb_cluster, record_each, verbose)

    def fit (self,STS, init=None, NbCycle=1):
        tic = time.time()

        surface = STS.Surface.copy()
        #print(surface.shape)
        self.prototype=surface[:self.nb_cluster,:]



        nb_proto = np.zeros((self.nb_cluster))
        last_time_activated = np.zeros((self.nb_cluster)).astype(int)
        idx_global = 0
        for each_cycle in range(NbCycle):
            for idx, Si in enumerate(surface):
                # find the closest prototype
                Distance_to_proto = EuclidianNorm(Si, self.prototype)
                closest_proto_idx = np.argmin(Distance_to_proto)
                Ck = self.prototype[closest_proto_idx,:]
                last_time_activated[closest_proto_idx] = idx
                ## Updating the prototype
                pk = nb_proto[closest_proto_idx]

                alpha = 1/(1+pk)
                beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))*np.sqrt(np.dot(Ck, Ck)))
                Ck_t = Ck + alpha*beta*(Si-Ck)

                # Updating the number of selection
                nb_proto[closest_proto_idx] += 1
                self.prototype[closest_proto_idx, :] = Ck_t

                critere = (idx-last_time_activated)>10000
                critere2 = nb_proto<25000
                if np.any(critere2*critere):
                    cri = nb_proto[critere]<25000
                    idx_critere = np.arange(0,self.nb_cluster)[critere][cri]
                    for idx_c in idx_critere:
                        Ck = self.prototype[idx_c,:]
                        beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))*np.sqrt(np.dot(Ck, Ck)))
                        Ck_t = Ck + 0.2*beta*(Si-Ck)
                        self.prototype[idx_c,:]=Ck_t
                    #print('critere atteint')
                    #print(critere)
                    #print(nb_proto)
                    #print(nb_proto[critere])
                    #break
                    #print(idx-last_time_activated)
                    #print(nb_proto[critere])
                #    print('ouahhh')
                if self.record_each != 0 :
                    if idx_global % int(self.record_each) == 0 :
                        output_distance = self.predict(STS,SurfaceFilter=1000)
                        error = np.mean(output_distance)
                        record_one = pd.DataFrame([{'error':error}],
                                            index=[idx_global])
                        self.record = pd.concat([self.record, record_one])
                idx_global += 1

        tac = time.time()
        #self.prototype = prototype
        self.nb_proto = nb_proto
        if self.verbose > 0:
            print('Clustering SpatioTemporal Surface in ------ {0:.2f} s'.format(tac-tic))

        return self.prototype#,nb_proto,last_time_activated

    def predict(self, STS, event=None, SurfaceFilter=None):
        if SurfaceFilter == None:
            to_predict = STS.Surface
        else :
            random_selection = np.random.permutation(np.arange(STS.Surface.shape[0]))[:SurfaceFilter]
            to_predict = STS.Surface[random_selection]

        if self.prototype is None :
            raise ValueError('Train the Cluster before doing prediction')
        polarity,output_distance = np.zeros(to_predict.shape[0]).astype(int),np.zeros(to_predict.shape[0])

        for idx,surface in enumerate(to_predict):
            Euclidian_distance = EuclidianNorm(surface,self.prototype)
            polarity[idx] = np.argmin(Euclidian_distance)
            output_distance[idx] = np.amin(Euclidian_distance)
        if event is not None :
            event_output = event.copy()
            event_output.polarity = polarity
            event_output.ListPolarities= list(np.arange(self.nb_cluster))
            return event_output, output_distance
        else :
            return output_distance
