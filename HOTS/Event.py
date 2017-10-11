import scipy.io
import numpy as np
from HOTS.Tools import LoadObject

class Event(object):
    '''
    Events is a class representing an event with all his attribute
    ATTRIBUTE
        + polarity : np.array of shape [nb_event] with the polarity number of each event
        + address : np array of shape [nb_event, 2] with the x and y of each event
        + time : np.array of shape [nb_event] with the time stamp of each event
        + ImageSize : tuple of shape 2, representing the maximum window where an event could appear
        + ListPolarities : list of the polarity we want to keep
        + ChangeIdx : list composed by the last index of event of each event
    '''
    def __init__(self,ImageSize,ListPolarities,OutOnePolarity=False):
        self.polarity = np.zeros(1)
        self.address = np.zeros(1)
        self.time = np.zeros(1)
        self.ImageSize = ImageSize
        #self.event_nb = np.zeros(1)
        self.ListPolarities = ListPolarities
        self.ChangeIdx = list()
        self.type = 'event'
        self.OutOnePolarity=OutOnePolarity
        ## Idée, faire un mécanisme pour vérifier qu'il n'y a pas d'adresse en dehors de l'image
    def LoadFromMat(self,path, image_number=None, verbose=0):
        '''
        Load Events from a .mat file. Only the events contained in ListPolarities are kept:
        INPUT
            + path : a string which is the path of the ;npy file (ex : './data_cache/ExtractedStabilized.mat')
            + image_number : list with all the numbers of image to load
        '''
        obj = scipy.io.loadmat(path)
        ROI = obj['ROI'][0]

        if type(image_number) is int:
            image_number = [image_number]
        elif type(image_number) is not list:
            raise TypeError('the type of argument image_number should be int or list')
        if verbose>0:
            print("chargement des images {0}".format(image_number))
        for idx, each_image in enumerate(image_number):
            image = ROI[each_image][0,0]
            each_address = np.hstack((image[1].T - 1, image[0].T - 1)).astype(int)
            each_time = image[3].transpose()*1e-6
            each_polarity = image[2].transpose().astype(int)
            if idx!=0 :
                self.address = np.vstack((self.address, each_address))
                self.polarity = np.concatenate((self.polarity, each_polarity))
                self.time = np.concatenate((self.time, each_time))
            else :
                self.address = each_address
                self.polarity = each_polarity
                self.time = each_time
        self.polarity = self.polarity[:,0]
        self.time = self.time[:,0]

        ## Filter only the wanted polarity
        filt = np.in1d(self.polarity,np.array(self.ListPolarities))
        self.filter(filt,mode='itself')

        if self.OutOnePolarity == True :
            self.polarity = np.ones_like(self.polarity)
            self.ListPolarities = [1]

    def SeparateEachImage(self):
        '''
        find the separation event index if more than one image is represented, and store it into
        self.ChangeIDX

        '''
        old_time = 0
        self.ChangeIdx = list()
        for idx, each_time in enumerate(self.time):
            if each_time<old_time:
                self.ChangeIdx.append(idx-1)
            old_time = each_time
        self.ChangeIdx.append(idx)

    def copy(self):
        '''
        copy the address, polarity, timing, and event_nb to another event
        OUTPUT :
            + event_output = event object which is the copy of self
        '''
        event_output = Event(self.ImageSize,self.ListPolarities)
        event_output.address = self.address.copy()
        event_output.polarity = self.polarity.copy()
        event_output.time = self.time.copy()
        event_output.ChangeIdx = self.ChangeIdx
        event_output.type = self.type
        event_output.OutOnePolarity = self.OutOnePolarity

        return event_output

    def filter(self,filt,mode=None):
        '''
        filter the event is mode is 'itself', or output another event else
        INPUT :
            + filt : np.array of boolean having the same dimension than self.polarity
        OUTPUT :
            + event_output : return an event, which is the filter version of self, only if mode
                is not 'itself'
        '''
        if mode == 'itself':
            self.address = self.address[filt]
            self.time = self.time[filt]
            self.polarity = self.polarity[filt]
            self.SeparateEachImage()
        else :
            event_output = self.copy()
            event_output.address = self.address[filt]
            event_output.time = self.time[filt]
            event_output.polarity = self.polarity[filt]
            event_output.SeparateEachImage()
            return event_output



def SimpleAlphabet(NbTrainingData,NbTestingData,Path=None,LabelPath=None, ClusteringData=None, OutOnePolarity = False, ListPolarities=[-1,1], verbose=0):
    if Path is None :
        Path = '../Data/ExtractedStabilized.mat'

    if LabelPath is None :
        label_list=LoadObject('../Data/alphabet_label.pkl')
    else :
        label_list=LoadObject(LabelPath)

    if NbTrainingData+NbTestingData > 76:
        raise NameError('Overlaping between TrainingData and Testing Data')
    event_tr = Event(ImageSize=(32,32), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)
    event_te = Event(ImageSize=(32,32), ListPolarities=ListPolarities, OutOnePolarity=OutOnePolarity)
    event_cl = Event(ImageSize=(32,32), ListPolarities=ListPolarities ,OutOnePolarity=OutOnePolarity)
    event_tr.LoadFromMat(Path,image_number=list(np.arange(0,NbTrainingData)),verbose=verbose)
    event_te.LoadFromMat(Path,image_number=list(np.arange(NbTrainingData, NbTrainingData+NbTestingData)),verbose=verbose)

    if ClusteringData is None :
        event_cl = event_tr
    else     :
        event_cl.LoadFromMat(Path,image_number=ClusteringData,verbose=verbose)

    ## Generate Groud Truth Label
    for idx,img in enumerate(np.arange(0,NbTrainingData)):
        if idx!= 0:
            label_tr = np.vstack((label_tr,label_list[img][0]))
        else:
            label_tr = label_list[img][0]

    for idx,img in enumerate(np.arange(NbTrainingData,NbTrainingData+NbTestingData)):
        if idx!= 0:
            label_te = np.vstack((label_te,label_list[img][0]))
        else:
            label_te = label_list[img][0]



    return event_tr, event_te, event_cl, label_tr, label_te
