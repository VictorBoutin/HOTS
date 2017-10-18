import time
import numpy as np
import pandas as pd

from HOTS.Tools import EuclidianNorm
import itertools

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
                #Distance_to_proto = EuclidianNorm(Si, self.prototype)
                Distance_to_proto = np.linalg.norm(Si - self.prototype,ord=2,axis=1)
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
            #Euclidian_distance = EuclidianNorm(surface,self.prototype)
            Euclidian_distance = np.linalg.norm(surface-self.prototype,ord=2,axis=1)
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
    def __init__(self,nb_cluster, record_each=0, verbose=0, eta=None):
        Cluster.__init__(self, nb_cluster, record_each, verbose)
        if eta is None :
            self.eta = 1
        else :
            self.eta = eta

    def fit (self,STS, init=None, NbCycle=1,eta=None):
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
                #Distance_to_proto = EuclidianNorm(Si, self.prototype)
                Distance_to_proto = np.linalg.norm(Si - self.prototype,ord=2,axis=1)
                closest_proto_idx = np.argmin(Distance_to_proto)
                Ck = self.prototype[closest_proto_idx,:]
                last_time_activated[closest_proto_idx] = idx
                ## Updating the prototype
                pk = nb_proto[closest_proto_idx]
                alpha = 1/(1+pk)
                beta = np.dot(Ck, Si)/(np.sqrt(np.dot(Si, Si))*np.sqrt(np.dot(Ck, Ck)))
                Ck_t = Ck + self.eta*alpha*beta*(Si-Ck)

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

        #diff = to_predict[:,:,np.newaxis]-self.prototype.T
        #Euclidian_distance = np.linalg.norm(diff,axis=1)
        #polarity = np.argmin(Euclidian_distance, axis=1)
        #output_distance = np.amin(Euclidian_distance, axis=1)
        for idx,surface in enumerate(to_predict):
            #Euclidian_distance = EuclidianNorm(surface,self.prototype)
            Euclidian_distance = np.linalg.norm(surface-self.prototype,ord=2,axis=1)
            polarity[idx] = np.argmin(Euclidian_distance)
            output_distance[idx] = np.amin(Euclidian_distance)
        if event is not None :
            event_output = event.copy()
            event_output.polarity = polarity
            event_output.ListPolarities= list(np.arange(self.nb_cluster))
            return event_output, output_distance
        else :
            return output_distance

    def predict0(self, STS, event=None, SurfaceFilter=None, batch_size=20000):
        if SurfaceFilter == None:
            to_predict = STS.Surface
        else :
            random_selection = np.random.permutation(np.arange(STS.Surface.shape[0]))[:SurfaceFilter]
            to_predict = STS.Surface[random_selection]

        if self.prototype is None :
            raise ValueError('Train the Cluster before doing prediction')
        if to_predict.shape[0]<=batch_size :
            batch_size = to_predict.shape[0]
        n_batch = to_predict.shape[0] // batch_size
        batches = np.array_split(to_predict, n_batch)

        batches = itertools.cycle(batches)
        polarity = np.zeros(to_predict.shape[0])
        output_distance = np.zeros(to_predict.shape[0])
        init_idx = 0
        for ii, this_X in zip(range(n_batch), batches):
            diff = this_X[:,:,np.newaxis] - self.prototype.T
            end_idx = init_idx + this_X.shape[0]
            norm1 = np.linalg.norm(diff, axis=1)
            polarity[init_idx:end_idx] = np.argmin(norm1,axis=1)
            output_distance[init_idx:end_idx] = np.amin(norm1,axis=1)
            init_idx = end_idx
        if event is not None :
            event_output = event.copy()
            event_output.polarity = polarity
            event_output.ListPolarities= list(np.arange(self.nb_cluster))
            return event_output, output_distance
        else :
            return output_distance

class KmeansMaroFast(Cluster):
    def __init__(self,nb_cluster, record_each=0, verbose=0, batch_size=10000):
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
                #Distance_to_proto = EuclidianNorm(Si, self.prototype)
                Distance_to_proto = np.linalg.norm(Si - self.prototype,ord=2,axis=1)
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
        #polarity,output_distance = np.zeros(to_predict.shape[0]).astype(int),np.zeros(to_predict.shape[0])

        diff = to_predict[:,:,np.newaxis]-self.prototype.T
        Euclidian_distance = np.linalg.norm(diff,axis=1)
        polarity = np.argmin(Euclidian_distance, axis=1)
        output_distance = np.amin(Euclidian_distance, axis=1)
        #for idx,surface in enumerate(to_predict):

        #    #Euclidian_distance = EuclidianNorm(surface,self.prototype)
        #    Euclidian_distance = np.linalg.norm(surface-self.prototype,ord=2,axis=1)
        #    polarity[idx] = np.argmin(Euclidian_distance)
        #    output_distance[idx] = np.amin(Euclidian_distance)
        if event is not None :
            event_output = event.copy()
            event_output.polarity = polarity
            event_output.ListPolarities= list(np.arange(self.nb_cluster))
            return event_output, output_distance
        else :
            return output_distance


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
            #Euclidian_distance = EuclidianNorm(surface,self.prototype)
            Euclidian_distance = np.linalg.norm(surface-self.prototype,ord=2,axis=1)
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
class KmeansHomeo(Cluster):
    def __init__(self,nb_cluster, record_each=0, verbose=0, nb_quant=100,
                    C=6, Norm_Type='max',eta=0.000005, eta_homeo=0.0005):
        Cluster.__init__(self, nb_cluster, record_each, verbose)
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
        if self.record_each>0:
            import pandas as pd
            record = pd.DataFrame()
        X = STS.Surface
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
        n_batches = n_samples // batch_size

        np.random.shuffle(X_train)
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

            if self.record_each>0:
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
    '''
    def fitMP(self, X,l0_sparseness,
                batch_size=100,NbCycle=1,record_num_batches=100):
        if self.record_each>0:
            import pandas as pd
            record = pd.DataFrame()


        n_samples, n_pixels = X.shape

        #print (n_samples, n_pixels)
        dictionary = np.random.randn(self.nb_cluster, n_pixels)
        norm = self.norm(dictionary, Norm_Type=self.Norm_Type)
        dictionary /= norm[:, np.newaxis]
        #norm = np.amax(dictionary, axis=1) #np.sqrt(np.sum(dictionary**2,axis=1))

        P_cum = np.linspace(0, 1, self.nb_quant, endpoint=True)[np.newaxis, :] * np.ones((self.nb_cluster, 1))
        print('size_dico : ', n_pixels)
        print('n_dico : ' ,self.nb_cluster)

        # splits the whole dataset into batches
        n_batches = n_samples // batch_size
        X_train = X.copy()
        np.random.shuffle(X_train)
        batches = np.array_split(X_train, n_batches)
        import itertools
        # Return elements from list of batches until it is exhausted. Then repeat the sequence indefinitely.
        batches = itertools.cycle(batches)
        n_iter = int(NbCycle * n_samples)
        for ii, this_X in zip(range(n_iter), batches):

            if this_X.ndim == 1:
                this_X = this_X[:, np.newaxis]

            n_samples, n_pixels = this_X.shape
            #n_dictionary, n_pixels = dictionary.shape
            sparse_code = np.zeros((n_samples,self.nb_cluster))

            if not P_cum is None:
                #nb_quant = P_cum.shape[1]
                stick = np.arange(self.nb_cluster)*self.nb_quant

            corr = (this_X @ dictionary.T)
            Xcorr = (dictionary @ dictionary.T)

            if  ii == 0:
                print('taille du batch : ', this_X.shape)
                print('taille de la coerrelation : ', corr.shape)

            for i_sample in range(n_samples):
                c = corr[i_sample, :].copy()
                #if i_sample == 0 :

                for i_l0 in range(int(l0_sparseness)) :
                #ind = np.argmax(c)
                    ind  = np.argmax(self.z_score(P_cum, self.prior(c), stick))
                    c_ind = c[ind] / Xcorr[ind, ind]
                    sparse_code[i_sample, ind] += c_ind
                    c -= c_ind * Xcorr[ind, :]
                    i_l0 += 1

            residual = this_X - sparse_code @ dictionary
            residual /= self.nb_cluster # divide by the number of features
            dictionary += self.eta * sparse_code.T @ residual

            # homeostasis
            #norm = np.sqrt(np.sum(dictionary**2, axis=1)).T
            norm = self.norm(dictionary, Norm_Type=self.Norm_Type)
            dictionary /= norm[:, np.newaxis]

            if ii % 5000 == 0:
                print('{0} / {1}'.format(ii,n_iter))

            if self.record_each>0:
                if ii % int(self.record_each) == 0:
                    from scipy.stats import kurtosis
                    indx = np.random.permutation(X_train.shape[0])[:record_num_batches]
                    polarity = self.code(X_train[indx, :],dictionary,P_cum,sparse=True)
                    error = np.linalg.norm(X_train[indx, :] - polarity @ dictionary)/record_num_batches
                    active_probe = np.sum(polarity>0,axis=0)
                    #polarity, dist_to_cluster = Distance_Prediction(X_train[indx, :],dictionary)
                    #error = np.mean(dist_to_cluster)
                    record_one = pd.DataFrame([{'error':error,
                                                'histo':active_probe,
                                                'var': np.var(active_probe)}],
                                            index=[ii])
                    record = pd.concat([record, record_one])

            P_cum = self.update_Pcum(P_cum, sparse_code)

        if self.record_each==0:
            return dictionary, P_cum
        else:
            return dictionary, P_cum, record
    '''

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
