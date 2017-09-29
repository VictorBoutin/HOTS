import numpy as np
import time

class STS(object):
    def __init__(self, tau, R ,verbose=0, initial_time=100, sigma=None):
        self.verbose=verbose
        self.tau = tau
        self.R = R

        self.RF_diam = 2*self.R + 1
        self.area = self.RF_diam**2

        self.initial_time = initial_time

        self.X_p, self.Y_p = np.meshgrid(np.arange(-self.R, self.R+1),
                                         np.arange(-self.R, self.R+1), indexing='ij')
        self.radius = np.sqrt(self.X_p**2 + self.Y_p**2)
        if sigma is None:
            self.mask = np.ones_like(self.radius)
        else:
            self.mask = np.exp(- .5 * self.radius**2 / self.R **2 / sigma**2 )
        self.mask = self.mask.reshape((1, self.mask.shape[0]*self.mask.shape[1]))

    def create(self, event, stop=None):
        self.ListPolarities = event.ListPolarities
        self.nb_polarities = len(self.ListPolarities)
        
        self.width = event.ImageSize[0] + 2*self.R
        self.height = event.ImageSize[1] + 2*self.R
        self.ListOfTimeMatrix = np.zeros((self.nb_polarities, self.width,self.height))-self.initial_time


        if stop is not None :
            self.Surface = np.zeros((stop+1, self.nb_polarities * self.area))
        else :
            self.Surface = np.zeros((event.address.shape[0], self.nb_polarities * self.area))

        timer = time.time()
        idx=0
        t_previous=0
        for idx_event, [addr,t,pol] in enumerate(zip(event.address,event.time,event.polarity)):
            if t<t_previous:
                self.ListOfTimeMatrix = np.zeros((self.nb_polarities, self.width, self.height)) - self.initial_time

            x, y = addr + self.R
            idx_pola = self.ListPolarities.index(pol)
            self.ListOfTimeMatrix[idx_pola, x, y] = t
            self.LocalTimeMatrix = self.ListOfTimeMatrix[:,(x-self.R):(x+self.R+1),(y-self.R):(y+self.R+1)]
            SI = np.exp(-(t-self.LocalTimeMatrix)/self.tau).reshape((len(self.ListPolarities)* self.area))
            self.Surface[idx_event,:] = SI#SI*self.mask
            t_previous = t
            if idx_event == stop:
                break

        tac = time.time()
        if self.verbose != 0:
            print('Generation of SpatioTemporal Surface in ------ {0:.2f} s'.format((tac-timer)))
        return self.Surface
