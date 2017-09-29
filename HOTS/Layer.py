import numpy as np
from HOTS.STS import STS
from HOTS.Cluster import CustomKmeans


class Layer(object):
    '''
    Layer is a mother class. A Layer is considered as an object with 2 main attributes :
        self.input : the event in input
        self.output : the event in output
    '''

    def __init__(self,event,verbose=0):
        self.input = event
        self.output = event.copy()
        self.verbose = verbose

    def GenerateAM(self):
        pass

    def GenerateHistogram(self):
        #if len(self.output.ChangeIdx) == 1:
        last_change=0
        for idx, each_change in enumerate(self.output.ChangeIdx):
            freq, pola = np.histogram(self.output.polarity[last_change:each_change+1],bins=len(self.output.ListPolarities))
            if idx != 0:
                freq_mat = np.vstack((freq_mat,freq))
            else :
                freq_mat = freq
            last_change=each_change
        return freq_mat, pola

class Filter(Layer):
    '''
    General Class for the Filters. This inherit all the methods and attribute of the Layer Class
    The method of this class correspond to different type of filters
    '''
    def __init__(self,event,verbose=0):
        Layer.__init__(self,event,verbose)

    def FilterNHBD(self,threshold, neighbourhood):
        '''
        Filter that keep the event if the number of event in a neighbour of size [2*neighbourhood+1,2*neighbourhood+1]
        is over the threshold value
        INPUT
            + threshold : (int), specify the minimum number of neighbour
            + neighbourhood : (int), specify the size of the neighbourhood to take into account
        OUTPUT
            + event : (event object) the filtered event
        '''
        filt = np.zeros(self.input.address.shape[0]).astype(bool)
        idx_old = 0
        accumulated_image = np.zeros((self.input.ImageSize[0]+2*neighbourhood,self.input.ImageSize[1]+2*neighbourhood))
        X_p, Y_p = np.meshgrid(np.arange(-neighbourhood,neighbourhood+1),np.arange(-neighbourhood,neighbourhood+1),indexing='ij')
        for idx, (each_address, each_pola,each_time) in enumerate(zip(self.input.address, self.input.polarity,self.input.time)):
            #if self.input.event_nb[idx_old]>self.input.event_nb[idx] :
            if self.input.time[idx_old]>self.input.time[idx] :
                accumulated_image = np.zeros((self.input.ImageSize[0]+2*neighbourhood,self.input.ImageSize[1]+2*neighbourhood))
            x_translated = each_address[0] + neighbourhood
            y_translated = each_address[1] + neighbourhood

            accumulated_image[x_translated,y_translated]= 1
            nb_voisin = np.sum(accumulated_image[x_translated+X_p, y_translated+Y_p])
            if nb_voisin>threshold:
                filt[idx]=True
            idx_old = idx
        self.output = self.input.filter(filt)
        return self.output

class ClusteringLayer(Layer):
    '''
    Class of a layer associating SpatioTemporal surface from input event to a Cluster, and then outputing another event
    INPUT :
        + event : and event object serving as input
    '''
    def __init__(self, event, verbose=0):
        Layer.__init__(self, event, verbose)

    def RunLayer(self, tau, R, Cluster):
        '''
        Run the layer
        INPUT :
            + tau : (int), the time constant of the spatiotemporal surface
            + R : (int), the size of the neighbourhood taken into consideration in the time surface
        OUTPUT :
            + self.output : (event object)
        '''
        self.SpTe_Layer = STS(tau=tau, R=R, verbose=self.verbose)
        Surface_Layer = self.SpTe_Layer.create(event=self.input)
        self.output,_ = Cluster.predict(STS=self.SpTe_Layer,event = self.input)
        return self.output

    def TrainLayer(self, tau, R, nb_cluster):
        '''
        Comment to do
        '''
        self.SpTe_Layer= STS(tau=tau, R=R, verbose=self.verbose)
        Surface_Layer = self.SpTe_Layer.create(event = self.input)
        ClusterLayer = CustomKmeans(nb_cluster = nb_cluster,verbose=self.verbose)
        Prototype = ClusterLayer.fit(self.SpTe_Layer)
        self.output,_ = ClusterLayer.predict(STS=self.SpTe_Layer,event = self.input)
        return self.output, ClusterLayer
