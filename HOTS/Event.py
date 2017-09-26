import scipy.io
import numpy as np

class Event(object):
    #def __init__(self):

    def LoadFromMat(self,path, image_number=None):
        '''
        Load Events from a .npy file:
        INPUT
            + path : a string which is the path of the ;npy file (ex : './data_cache/ExtractedStabilized.mat')
            + image_number : list with all the numbers of image to load
        OUTPUT
            + address_matrix : np array of shape [nb_event, 2] with the x and y of each event
            + time_matrix : np.array of shape [nb_event] with the time stamp of each event
            + polarity_matrix : np.array of shape [nb_event] with the polarity number of each event
            + event_matrix : np.array of shape [nb_event] with the number of the event
        '''
        obj = scipy.io.loadmat(path)
        ROI = obj['ROI'][0]

        if type(image_number) is int:
            image_number = [image_number]
        elif type(image_number) is not list:
            raise TypeError('the type of argument image_number should be int or list')

        print("chargement des images {0}".format(image_number))
        for idx, each_image in enumerate(image_number):
            image = ROI[each_image][0,0]
            each_address = np.hstack((image[1].T - 1, image[0].T - 1)).astype(int)
            each_time = image[3].transpose()*1e-6
            each_polarity = image[2].transpose().astype(int)
            each_event_nb = np.arange(each_address.shape[0])
            if idx!=0 :
                self.address = np.vstack((self.address, each_address))
                self.polarity = np.concatenate((self.polarity, each_polarity))
                self.time = np.concatenate((self.time, each_time))
                self.event_nb = np.concatenate((self.event_nb, each_event_nb))
            else :
                self.address = each_address
                self.polarity = each_polarity
                self.time = each_time
                self.event_nb = each_event_nb


        return self.address, self.time[:,0], self.polarity[:,0], self.event_nb

    def SeparateEachImage(self):
        '''
        find the separation event index if more than one image is represented in self.event_nb
        INPUT
            ...
        OUTPUT
            + change_idx : list of shape [nb_of_change+1] indicating all the index of change
        '''

        self.change_idx = [0]
        old_event = 0
        for absolute_idx, event_number in enumerate(self.event_nb):
            if old_event > event_number:
                self.change_idx.append(absolute_idx-1)
            old_event = event_number
        self.change_idx.append(absolute_idx)
        return self.change_idx



class Filters(object):
    def __init__(self, events):
        self.events = events

    def Neighbour(self, threshold, neighbourhood, image_size):
        '''
        keep the event if the number of event in a neighbour of size [2*neighbourhood+1,2*neighbourhood+1]
        is over the threshold value
        INPUT
            threshold : (int), specify the minimum number of neighbour
            neighbourhood : (int), specify the size of the neighbourhood to take into account
            image_size : (tuple of shape 2), specify the size of the image
        OUTPUT
            address : the filtered address
            time : the filtered time
            polarity : the filtered polarity
            event_nb : the filtered event_nb
        '''
        filt = np.zeros(self.events.address.shape[0]).astype(bool)
        idx_old = 0
        accumulated_image = np.zeros((image_size[0]+2*neighbourhood,image_size[1]+2*neighbourhood))
        X_p, Y_p = np.meshgrid(np.arange(-neighbourhood,neighbourhood+1),np.arange(-neighbourhood,neighbourhood+1),indexing='ij')
        for idx, (each_address, each_pola,each_time) in enumerate(zip(self.events.address, self.events.polarity,self.events.time)):
            if self.events.event_nb[idx_old]>self.events.event_nb[idx] :
                accumulated_image = np.zeros((image_size[0]+2*neighbourhood,image_size[1]+2*neighbourhood))
            x_translated = each_address[0] + neighbourhood
            y_translated = each_address[1] + neighbourhood

            accumulated_image[x_translated,y_translated]= 1
            nb_voisin = np.sum(accumulated_image[x_translated+X_p, y_translated+Y_p])
            if nb_voisin>threshold:
                filt[idx]=True
            idx_old = idx
        self.events.address = self.events.address[filt]
        self.events.time = self.events.time[filt][:,0]
        self.events.polarity = self.events.polarity[filt][:,0]
        self.events.event_nb = self.events.event_nb[filt]
        return self.events.address, self.events.time, self.events.polarity, self.events.event_nb
