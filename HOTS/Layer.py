import numpy as np
from HOTS.STS import STS
from HOTS.Cluster import CustomKmeans, KmeansMaro, KmeansHomeo

class Layer(object):
    '''
    Layer is a mother class. A Layer is considered as an object with 2 main attributes :
        self.input : the event in input
        self.output : the event in output
    '''

    def __init__(self, verbose=0):
        #self.input = Event()
        #self.output = event.copy()
        self.verbose = verbose
        self.type = 'void'

    def GenerateAM(self):
        pass

    def Train(self):
        if self.type == 'ClusteringLayer':
            pass

class FilterNHBD(Layer):
    '''
    General Class for the Filters. This inherit all the methods and attribute of the Layer Class
    The method of this class correspond to different type of filters
    '''
    def __init__(self, threshold, neighbourhood, verbose=0):
        Layer.__init__(self, verbose)
        self.type = 'Filter'
        self.threshold = threshold
        self.neighbourhood = neighbourhood

    def RunFilter(self, event):
        '''
        Filter that keep the event if the number of event in a neighbour of size [2*neighbourhood+1,2*neighbourhood+1]
        is over the threshold value
        INPUT
            + threshold : (int), specify the minimum number of neighbour
            + neighbourhood : (int), specify the size of the neighbourhood to take into account
        OUTPUT
            + event : (event object) the filtered event
        '''
        self.input = event
        filt = np.zeros(self.input.address.shape[0]).astype(bool)
        idx_old = 0
        accumulated_image = np.zeros((self.input.ImageSize[0]+2*self.neighbourhood,self.input.ImageSize[1]+2*self.neighbourhood))
        X_p, Y_p = np.meshgrid(np.arange(-self.neighbourhood,self.neighbourhood+1),np.arange(-self.neighbourhood,self.neighbourhood+1),indexing='ij')
        for idx, (each_address, each_pola,each_time) in enumerate(zip(self.input.address, self.input.polarity,self.input.time)):
            #if self.input.event_nb[idx_old]>self.input.event_nb[idx] :
            if self.input.time[idx_old]>self.input.time[idx] :
                accumulated_image = np.zeros((self.input.ImageSize[0]+2*self.neighbourhood,self.input.ImageSize[1]+2*self.neighbourhood))
            x_translated = each_address[0] + self.neighbourhood
            y_translated = each_address[1] + self.neighbourhood

            accumulated_image[x_translated,y_translated]= 1
            nb_voisin = np.sum(accumulated_image[x_translated+X_p, y_translated+Y_p])
            if nb_voisin>self.threshold:
                filt[idx]=True
            idx_old = idx
        self.output = self.input.filter(filt)
        return self.output


class ClusteringLayer(Layer):
    '''
    Class of a layer associating SpatioTemporal surface from input event to a Cluster, and then outputing another event
    INPUT :
        + tau : (int), the time constant of the spatiotemporal surface
        + R : (int), the size of the neighbourhood taken into consideration in the time surface
    '''
    def __init__(self, tau, R,  ThrFilter=0, verbose=0, LearningAlgo='standard', kernel='exponential',eta=None,eta_homeo=None):
        Layer.__init__(self, verbose)
        self.type = 'Layer'
        self.tau = tau
        self.R = R
        self.ThrFilter = ThrFilter
        self.LearningAlgo = LearningAlgo
        if self.LearningAlgo not in ['homeo','maro','standard']:
            raise KeyError('LearningAlgo should be in [homeo,maro,standard]')
        self.kernel = kernel
        if self.kernel not in ['linear','exponential']:
            raise KeyError('[linear,exponential]')
        self.eta = eta
        #print(eta)
        self.eta_homeo = eta_homeo
        #print(eta_homeo)

    def RunLayer(self, event, Cluster):
        '''
        Associate each polarity of the event input to the prototype of the cluster
        INPUT :
            + event (<object Event>) : input event
            + Cluster (<object Cluster>) : Cluster, previously trained,
        OUTPUT :
            + self.output : (<object Event>) : Event with new polarities corresponding to the closest cluster center
        '''
        self.input = event
        self.SpTe_Layer = STS(tau=self.tau, R=self.R, verbose=self.verbose)
        Surface_Layer = self.SpTe_Layer.create(event=self.input, kernel=self.kernel)
        event_filtered, filt = self.SpTe_Layer.FilterRecent(event = self.input, threshold=self.ThrFilter) ## Check that THRFilter=0 is equivalent to no Filter
        self.output,_ = Cluster.predict(STS=self.SpTe_Layer,event = event_filtered)
        return self.output

    def TrainLayer(self, event, nb_cluster, record_each=0, NbCycle=1):
        '''
        Learn the Cluster
        INPUT :
            + event (<object event>) : input event
            + nb_cluster(<int>) : nb of centers
            + record_each (<int>) : record the convergence error each reach_each
            + NbCycle (<int>) : number of time we repeat the learning process. Need to be used when not enought training data to reach the convergence

        OUTPUT :
            + output (<object event>) : Event with new polarities corresponding to the closest cluster center
            + ClusterLayer (<object Cluster) : Learnt cluster
        '''
        self.input=event
        self.SpTe_Layer = STS(tau=self.tau, R=self.R, verbose=self.verbose)
        Surface_Layer = self.SpTe_Layer.create(event = self.input, kernel=self.kernel)
        event_filtered, filt = self.SpTe_Layer.FilterRecent(event = self.input, threshold=self.ThrFilter)
        if self.LearningAlgo == 'standard' :
            self.ClusterLayer = CustomKmeans(nb_cluster = nb_cluster,verbose=self.verbose, record_each=record_each)
        elif self.LearningAlgo == 'maro' :
            self.ClusterLayer = KmeansMaro(nb_cluster = nb_cluster,verbose=self.verbose, record_each=record_each,
                                        eta=self.eta)
        elif self.LearningAlgo == 'homeo' :
            self.ClusterLayer = KmeansHomeo(nb_cluster = nb_cluster,verbose=self.verbose, record_each=record_each,
                                        eta=self.eta, eta_homeo=self.eta_homeo)
        Prototype = self.ClusterLayer.fit(self.SpTe_Layer, NbCycle=NbCycle)
        self.output,_ = self.ClusterLayer.predict(STS=self.SpTe_Layer,event = event_filtered)

        return self.output, self.ClusterLayer
