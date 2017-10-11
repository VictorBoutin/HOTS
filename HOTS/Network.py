

class Network(object):
    def __init__(self, Layers, verbose=0):
        self.Layers = Layers
        self.verbose = verbose

    def TrainCluster(self, event, nb_cluster, KN ,record_each=0, NbCycle=1):
        event_i = event
        idx_Layer = 0
        ClusterList = list()
        for idx, each_Layer in enumerate(self.Layers):
            if each_Layer.type == 'void':
                print('problem !!' )
            elif each_Layer.type == 'Filter':
                event_o = each_Layer.RunFilter(event_i)
            elif each_Layer.type == 'Layer':
                event_o, Cluster = each_Layer.TrainLayer(event_i, nb_cluster*(KN)**(idx_Layer), record_each=record_each, NbCycle=NbCycle)
                ClusterList.append(Cluster)
                idx_Layer = idx_Layer + 1
                #if self.verbose>0 :
                #    print('Training of Layer {0}'.format(idx_Layer))
            else :
                print(type(each_Layer))
            event_i = event_o
            #print(event_i.address.shape[0])
        return ClusterList, event_o

    def RunNetwork(self,event, ClusterList):
        event_i = event
        idx_Layer = 0
        for idx, each_Layer in enumerate(self.Layers):
            if each_Layer.type == 'void':
                print('problem !!' )
            elif each_Layer.type == 'Filter':
                event_o = each_Layer.RunFilter(event_i)
            elif each_Layer.type == 'Layer':
                event_o = each_Layer.RunLayer(event_i, Cluster = ClusterList[idx_Layer] )
                idx_Layer = idx_Layer + 1
                if self.verbose>0:
                    print('Layer')
            else :
                print(type(each_Layer))
            event_i = event_o
        return event_o
