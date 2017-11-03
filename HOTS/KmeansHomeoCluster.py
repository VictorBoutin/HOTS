__author__ = "(c) Victor Boutin & Laurent Perrinet INT - CNRS"

import time
import numpy as np
import pandas as pd

from HOTS.Tools import EuclidianNorm
from HOTS.KmeansCluster import Cluster
import itertools



class KmeansHomeo(Cluster):
    def __init__(self,nb_cluster, to_record=False, verbose=0, nb_quant=100,
                    C=6, Norm_Type='max',eta=0.000005, eta_homeo=0.0005):
        Cluster.__init__(self, nb_cluster, to_record, verbose)
        self.nb_quant = nb_quant
        self.C = C
        self.Norm_Type = Norm_Type
        self.do_sym = False

        if eta is None :
            self.eta = 0.000005
        else :
            self.eta = eta

        if eta_homeo is None:
            self.eta_homeo = 0.0005
        else :
            self.eta_homeo = eta_homeo

        self.verbose = verbose

    def fit(self, STS, batch_size=100, NbCycle=1, record_num_batches=10000):

        #if self.record_each>0:
        #   import pandas as pd
        #    record = pd.DataFrame()
        X = STS.Surface
        if self.to_record == True :
            self.record_each = X.shape[0]//100
        n_samples, n_pixels = X.shape
        X_train = X.copy()
        norm = self.norm(X_train, Norm_Type=self.Norm_Type)
        X_train /= norm[:, np.newaxis]
        prototype = X_train[:self.nb_cluster,:].copy()
        #prototype = np.random.randn(self.nb_cluster, n_pixels)
        #print('avant', np.amax(prototype,axis=1), self.norm(prototype, Norm_Type=self.Norm_Type))
        #norm = self.norm(prototype, Norm_Type=self.Norm_Type)
        #prototype /= norm[:, np.newaxis]

        #print('apres', np.amax(prototype,axis=1), self.norm(prototype, Norm_Type=self.Norm_Type))
        #norm = np.amax(dictionary, axis=1) #np.sqrt(np.sum(dictionary**2,axis=1))

        self.P_cum = np.linspace(0, 1, self.nb_quant, endpoint=True)[np.newaxis, :] * np.ones((self.nb_cluster, 1))

        # splits the whole dataset into batches
        #print(n_samples)
        n_batches = n_samples // batch_size

        np.random.shuffle(X_train)
        #print(X_train.shape)
        #print(n_batches)
        batches = np.array_split(X_train, n_batches)
        import itertools
        # Return elements from list of batches until it is exhausted. Then repeat the sequence indefinitely.
        #
        batches = itertools.cycle(batches)

        n_iter = int(NbCycle * n_samples)
        #print(n_iter)

        for ii, this_X in zip(range(n_iter), batches):
            if this_X.ndim == 1:
                this_X = this_X[:, np.newaxis]

            n_samples, n_pixels = this_X.shape
            #n_dictionary, n_pixels = dictionary.shape
            sparse_code = np.zeros((n_samples,self.nb_cluster))

            if not self.P_cum is None:
                #nb_quant = P_cum.shape[1]
                stick = np.arange(self.nb_cluster)*self.nb_quant

            corr = (this_X @ prototype.T)

            for i_sample in range(n_samples):
                c = corr[i_sample, :].copy()
                #ind = np.argmax(c)
                ind  = np.argmax(self.z_score(self.P_cum, self.prior(c), stick))
                sparse_code[i_sample, ind] = c[ind]
                Si = this_X[i_sample,:]
                Ck = prototype[ind,:]

                #alpha = 1/(1+pk)
                beta = np.dot(Ck,Si)/(np.sqrt(np.dot(Si,Si))*np.sqrt(np.dot(Ck,Ck)))
                #prototype[ind,:] = Ck + self.eta*(Si - beta * Ck)
                prototype[ind,:] = Ck + beta * self.eta * (Si - Ck)
                #prototype[ind,:] /= self.norm(prototype[ind,:],Norm_Type=self.Norm_Type)

            norm = self.norm(prototype, Norm_Type=self.Norm_Type)

            prototype /= norm[:, np.newaxis]

            if self.verbose > 0 and ii % 5000 == 0:
                print('{0} / {1}'.format(ii,n_iter))

            if self.to_record == True:
                if ii % int(self.record_each) == 0:
                    from scipy.stats import kurtosis
                    indx = np.random.permutation(X_train.shape[0])[:record_num_batches]
                    polarity = self.code(X_train[indx, :],prototype,self.P_cum,sparse=True)
                    error = np.linalg.norm(X_train[indx, :] - polarity @ prototype)/record_num_batches
                    active_probe = np.sum(polarity>0,axis=0)
                    #polarity, dist_to_cluster = Distance_Prediction(X_train[indx, :],dictionary)
                    #error = np.mean(dist_to_cluster)
                    record_one = pd.DataFrame([{'error':error,
                                                'histo':active_probe,
                                                'var': np.var(active_probe)}],
                                            index=[ii])
                    self.record = pd.concat([self.record, record_one])

            self.P_cum = self.update_Pcum(self.P_cum, sparse_code)
        self.prototype = prototype
        return prototype
        #if self.record_each==0:
        #    return prototype, P_cum
        #else:
        #    return prototype, P_cum, record

    '''
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
            Euclidian_distance = np.linalg.norm(surface-self.prototype,ord=2,axis=1)
            #Euclidian_distance = EuclidianNorm(surface,self.prototype)
            polarity[idx] = np.argmin(Euclidian_distance)
            output_distance[idx] = np.amin(Euclidian_distance)
        if event is not None :
            event_output = event.copy()
            event_output.polarity = polarity
            event_output.ListPolarities= list(np.arange(self.nb_cluster))
            return event_output, output_distance
        else :
            return output_distance
    '''
    def norm(self, to_normalize,Norm_Type):
        if Norm_Type == 'standard':
            if to_normalize.ndim > 1:
                norm = np.sqrt(np.sum(to_normalize**2,axis=1))
            else :
                norm = np.sqrt(np.sum(to_normalize**2))
        elif Norm_Type == 'max':
            if to_normalize.ndim > 1:
                norm = np.amax(to_normalize, axis=1)
            else :
                norm = np.amax(to_normalize)

        # elif Norm_Type == 'both':
        #     if to_normalize.ndim > 1:
        #         norm
        return norm

    def update_Pcum(self, P_cum, code):
        """Update the estimated modulation function in place.

        Parameters
        ----------
        P_cum: array of shape (n_samples, n_components)
        Value of the modulation function at the previous iteration.

        dictionary: array of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

        code: array of shape (n_samples, n_features)
        Data matrix.

        eta_homeo: float
        Gives the learning parameter for the mod.

        verbose:
        Degree of output the procedure will print.

        Returns
        -------
        P_cum: array of shape (n_samples, n_components)
        Updated value of the modulation function.

        """
        if self.eta_homeo>0.:
            P_cum_ = self.get_Pcum(code)
            P_cum = (1 - self.eta_homeo)*P_cum + self.eta_homeo * P_cum_
        return P_cum




    def code(self, X, dictionary, P_cum, sparse=False):
        n_samples = X.shape[0]
        corr = X @ dictionary.T
        stick = np.arange(self.nb_cluster)*self.nb_quant
        if sparse == False :
            polarity = np.zeros(n_samples).astype(int)
        else :
            polarity = np.zeros((n_samples,self.nb_cluster)).astype(int)

        for i_sample in range(n_samples):
            c = corr[i_sample, :].copy()
            ind  = np.argmax(self.z_score(P_cum, self.prior(c), stick))
            if sparse == False :
                polarity[i_sample] = ind
            else:
                polarity[i_sample,ind] = 1
        return polarity

    def get_Pcum(self,code):
        n_samples = code.shape[0]
        P_cum = np.zeros((self.nb_cluster, self.nb_quant))
        for i in range(self.nb_cluster):
            p, bins = np.histogram(self.prior(code[:, i]), bins=np.linspace(0., 1, self.nb_quant, endpoint=True), density=True)
            p /= p.sum()
            P_cum[i, :] = np.hstack((0, np.cumsum(p)))
        return P_cum

    def prior(self, code):
        if self.do_sym:
            return 1.-np.exp(-np.abs(code)/self.C)
        else:
            return (1.-np.exp(-code/self.C))*(code>0)

    def z_score(self, Pcum, p_c, stick):
        return Pcum.ravel()[(p_c*Pcum.shape[1] - (p_c==1)).astype(np.int) + stick]
