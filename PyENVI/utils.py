import tensorflow as tf
import numpy as np
import sklearn.neighbors
import scipy.sparse
import anndata
import scanpy as sc
import scipy.special



class LinearLayer(tf.keras.layers.Layer):
    """
    Costume keras linear layer

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        kernel_init (keras initializer): initializer for neural weights
        bias_init (keras initializer): initializer of neural biases
    """
    def __init__(self, units, input_dim, kernel_init, bias_init):
        super(LinearLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer=kernel_init, trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer=bias_init, trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class ConstantLayer(tf.keras.layers.Layer):
    """
    Costume keras constant layer, biases only

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        bias_init (keras initializer): initializer of neural biases
        comm_disp (bool): if True, spatial_dist and sc_dist share dispersion parameter(s)
        const_disp (bool): if True, dispertion parameter(s) are only per gene, rather there per gene per sample
    """
    def __init__(self, units, input_dim, bias_init):
        super(ConstantLayer, self).__init__()
        self.b = self.add_weight(shape=(units,), initializer=bias_init, trainable=True)

    def call(self, inputs):
        return tf.tile(self.b[None, :], [inputs.shape[0], 1])
    
class ENVIOutputLayer(tf.keras.layers.Layer):
    """
    Costume keras layer for ENVI expression decoder output

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        kernel_init (keras initializer): initializer for neural weights
        bias_init (keras initializer): initializer of neural biases
        spatial_dist (str): distribution used to describe spatial data (default pois, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm') 
        sc_dist (str): distribution used to describe sinlge cell data (default nb, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
    """
    def __init__(self, 
                 input_dim, 
                 units, 
                 kernel_init,
                 bias_init,
                 spatial_dist = 'pois',
                 sc_dist = 'nb',
                 comm_disp = False,
                 const_disp = False):
        super(ENVIOutputLayer, self).__init__()
        
        self.input_dim = input_dim
        self.units = units
        
        self.spatial_dist = spatial_dist
        self.sc_dist = sc_dist
        self.comm_disp = comm_disp
        self.const_disp = const_disp
        
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        
        self.r = LinearLayer(units, input_dim, kernel_init, bias_init)  
        
        if(self.comm_disp):  
            
            if(self.spatial_dist == 'zinb'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                                    
                self.d_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))

            elif(self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                
            if(self.sc_dist == 'zinb'):
                self.p_sc = (self.p_spatial if (self.spatial_dist == 'zinb' or self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm') 
                            else (ConstantLayer(units, input_dim, bias_init) if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init)))
                                    
                self.d_sc = (self.d_spatial if (self.spatial_dist == 'zinb') 
                            else (ConstantLayer(units, input_dim, bias_init) if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init)))

            elif(self.sc_dist == 'nb' or self.sc_dist == 'full_norm'):
                
                self.p_sc = (self.p_spatial if (self.spatial_dist == 'zinb' or self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm') 
                            else (ConstantLayer(units, input_dim, bias_init) if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init)))
                
                
            if(self.spatial_dist == 'zinb' or self.sc_dist == 'zinb'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                                
                self.p_sc = self.p_spatial
                
                self.d_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                
                self.d_sc = self.d_spatial

            elif(self.spatial_dist == 'nb' or self.sc_dist == 'nb' or self.spatial_dist == 'full_norm' or self.sc_dist == 'full_norm'):
                
                self.p_spatial = (ConstantLayer(units, input_dim, kernel_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                
                self.p_sc = self.p_spatial
        
        else:  
            
            if(self.spatial_dist == 'zinb'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                                    
                self.d_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))

            elif(self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                
            if(self.sc_dist == 'zinb'):
                self.p_sc = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
                                    
                self.d_sc = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))

            elif(self.sc_dist == 'nb' or self.sc_dist == 'full_norm'):
                self.p_sc = (ConstantLayer(units, input_dim, bias_init)
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init))
            
    
    
    def call(self, inputs, mode = 'spatial'):
        r = self.r(inputs)
         
        if(getattr(self, mode + '_dist') == 'zinb'):
                p = getattr(self, 'p_' + mode)(inputs)
                d = getattr(self, 'd_' + mode)(inputs)
                return(r,p,d)
                
        if(getattr(self, mode + '_dist') == 'nb' or getattr(self, mode + '_dist') == 'full_norm'):
                p = getattr(self, 'p_' + mode)(inputs)
                return(r,p)
        
        return(r)
    
    
def NormalNorm(arr):
    """
    z-scores data

    Args:
        arr (array): array to z-score
    Return:
        zscore_arr: arr z-scored
    """
    arr = np.asarray(arr)
    arr = (arr - arr.mean(axis = 0, keepdims = True))/arr.std(axis = 0, keepdims = True)
    return(arr)
    
def MatSqrtTF(Mats):
    """
    Computes psuedo matrix square root with tensorfow linear algebra on cpu

    Args:
        Mats (array): Matrices to compute square root of 
    Return:
        SqrtMats (np.array): psuedo matrix square of Mats
    """
    with tf.device('/CPU:0'):
        e,v = tf.linalg.eigh(Mats)
        e = tf.where(e < 0, 0, e)
        e = tf.math.sqrt(e)
        return(tf.linalg.matmul(tf.linalg.matmul(v, tf.linalg.diag(e)), v, transpose_b = True).numpy())


def BatchKNN(data, batch, k):
    
    kNNGraphIndex = np.zeros(shape = (data.shape[0], k))
    WeightedIndex = np.zeros(shape = (data.shape[0] ,k))
    
    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]
        
        batch_knn = sklearn.neighbors.kneighbors_graph(data[val_ind], n_neighbors=k, mode='distance', n_jobs=-1).tocoo()
        batch_knn_ind = np.reshape(np.asarray(batch_knn.col), [data[val_ind].shape[0], k])
        
        batch_knn_weight = scipy.special.softmax(-np.reshape(batch_knn.data, [data[val_ind].shape[0], k]), axis = -1)
        
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]
        WeightedIndex[val_ind] = batch_knn_weight
    return(kNNGraphIndex, WeightedIndex)
    
    
def GetNeighExp(spatial_data, kNN, spatial_key = 'spatial', batch_key = -1, data_key = None):
    
    """
    Computing Niche mean expression based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        kNN (int): number of nearest neighbours to define niche
        spatial_key (str): obsm key name with physical location of spots/cells (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        data_key (str): obsm key to compute niche mean across (defualt None, uses gene expression .X)

    Return:
        NeighExp: Average geene expression in niche 
        kNNGraphIndex: indices of nearest spatial neighbours per cell
    """
    
    if(data_key is None):
        Data = spatial_data.X
    else:
        Data = spatial_data.obsm[data_key]
        
    if(batch_key == -1):        
        kNNGraph = sklearn.neighbors.kneighbors_graph(spatial_data.obsm[spatial_key], n_neighbors=kNN, mode='distance', n_jobs=-1).tocoo()
        kNNGraph = scipy.sparse.coo_matrix((np.ones_like(kNNGraph.data), (kNNGraph.row, kNNGraph.col)), shape=kNNGraph.shape)
        kNNGraphIndex = np.reshape(np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN])
    else:
        kNNGraphIndex, _ = BatchKNN(spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN)
    
    
    return(Data[kNNGraphIndex[np.arange(spatial_data.obsm[spatial_key].shape[0])]])



def GetCovMats(spatial_data, kNN, spatial_key = 'spatial', batch_key = -1, MeanExp = None, weighted = False):
    
        
    """
    Wrapper to compute niche covariance based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        kNN (int): number of nearest neighbours to define niche
        spatial_key (str): obsm key name with physical location of spots/cells (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        MeanExp (np.array): expression vector to shift niche covariance with
        weighted (bool): if True, weights covariance by spatial distance
    Return:
        CovMats: niche covariance matrices
        kNNGraphIndex: indices of nearest spatial neighbours per cell
    """
    ExpData = np.log(spatial_data[:, spatial_data.var.highly_variable].X + 1)
    
    if(batch_key == -1):        
        kNNGraph = sklearn.neighbors.kneighbors_graph(spatial_data.obsm[spatial_key], n_neighbors=kNN, mode='distance', n_jobs=-1).tocoo()
        kNNGraph = scipy.sparse.coo_matrix((np.ones_like(kNNGraph.data), (kNNGraph.row, kNNGraph.col)), shape=kNNGraph.shape)
        kNNGraphIndex = np.reshape(np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN])
        
        WeightedIndex = scipy.special.softmax(-np.reshape(kNNGraph.data, [spatial_data.obsm[spatial_key].shape[0], kNN]), axis = -1)
    else:
        kNNGraphIndex, WeightedIndex = BatchKNN(spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN)
    
    if not weighted:
        WeightedIndex = np.ones_like(WeightedIndex)/kNN
        
    if(MeanExp is None):
        DistanceMatWeighted = (ExpData.mean(axis = 0)[None, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]) * np.sqrt(WeightedIndex)[:, :, None] * np.sqrt(1 / (1 - np.sum(np.square(WeightedIndex), axis= -1)))[:, None, None]
    else:
        DistanceMatWeighted = (MeanExp[:, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]) * np.sqrt(WeightedIndex)[:, :, None] * np.sqrt(1 / (1 - np.sum(np.square(WeightedIndex), axis= -1)))[:, None, None]

    CovMats = np.matmul(DistanceMatWeighted.transpose([0,2,1]), DistanceMatWeighted)
    CovMats = CovMats + CovMats.mean() * 0.00001 * np.expand_dims(np.identity(CovMats.shape[-1]), axis=0) 
    return(CovMats, kNNGraphIndex)

def GetCov(spatial_data, k, g, genes, cov_dist, spatial_key = 'spatial', batch_key = -1):
    
    """
    Compte niche covariance matrices for spatial data

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbours to define niche
        g (int): number of HVG to compute niche covariance matricies
        genes (list of str): list of genes to keep for niche covariance
        cov_dist (str): distribution to transform niche covariance matrices to fit into

    Return:
        CovMats: raw, untransformed niche covariance matrices
        CovMatsTransformed: covariance matrices transformed into chosen cov_dist
        NeighExp: Average geene expression in niche 
        CovGenes: Genes used for niche covariance 
    """
        
        
    spatial_dataCopy = spatial_data.copy()
    sc.pp.normalize_total(spatial_dataCopy, target_sum = np.median(np.asarray(spatial_data.X).sum(axis=1))) 

    if(g == -1):
        CovGeneSet = np.arange(spatial_data.shape[-1])
        spatial_data.var.highly_variable = True
    else:
        sc.pp.log1p(spatial_dataCopy)
        sc.pp.highly_variable_genes(spatial_dataCopy, n_top_genes = g)
        
        if(g == 0):
            spatial_dataCopy.var.highly_variable = False
        if(len(genes) > 0):
            spatial_dataCopy.var['highly_variable'][genes] = True
        
        spatial_data.var.highly_variable = spatial_dataCopy.var.highly_variable
    
    CovGeneSet = np.where(np.asarray(spatial_data.var.highly_variable))[0]
    CovGenes = spatial_data.var_names[CovGeneSet]
    
    CovMats, kNNGraphIndex = GetCovMats(spatial_data, k, spatial_key = spatial_key, batch_key = batch_key, weighted = False)
    NeighExp = spatial_data.X[kNNGraphIndex[np.arange(spatial_data.shape[0])]].mean(axis = 1)
    
    if(cov_dist == 'norm'):
        CovMatsTransformed = CovMats.reshape([CovMats.shape[0], -1])
        CovMatsTransformed = (CovMatsTransformed - CovMatsTransformed.mean(axis = 0, keepdims = True)) / CovMatsTransformed.std(axis = 0, keepdims = True)
    if(cov_dist == 'OT'):
        CovMatsTransformed = MatSqrtTF(CovMats)
    else:
        CovMatsTransformed = np.copy(CovMats)
    
    return(CovMats.astype('float32'), CovMatsTransformed.astype('float32'), NeighExp.astype('float32'), CovGenes)


