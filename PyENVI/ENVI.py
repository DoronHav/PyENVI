import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sklearn.neighbors
import scipy.sparse
import time
import anndata
import sklearn.neural_network
import scanpy as sc
import pandas as pd
import scipy.special
import phenograph

from utils import *
from dists import *


    
    
class ENVI(tf.keras.Model):
    
    """
    ENVI Integrates spatial and single-cell data
    
    Parameters: 
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        sc_data (anndata): anndata with sinlge cell data
        spatial_key (str): obsm key name with physical location of spots/cells (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        num_layers (int): number of neural network for decoders and encoders (default 3)
        num_neurons (int): number of neurons in each layer (default 1024)
        latent_dim (int): size of ENVI latent dimention (size 512)
        k_nearest (int): number of physical neighbours to describe niche (default 8)
        num_cov_genes (int): number of HVGs to compute niche covariance with default (64), if -1 takes all genes
        cov_genes (list of str): manual genes to compute niche with (default [])
        num_HVG (int): number of HVGs to keep for sinlge cell data (default 2048), if -1 takes all genes
        sc_genes (list of str): manual genes to keep for sinlge cell data (default [])
        spatial_dist (str): distribution used to describe spatial data (default pois, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm') 
        sc_dist (str): distribution used to describe sinlge cell data (default nb, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
        cov_dist (str): distribution used to describe niche covariance from spatial data (default OT, could be 'OT', 'wish' or 'norm')
        prior_dist (str): prior distribution for latent (default normal)
        comm_disp (bool): if True, spatial_dist and sc_dist share dispersion parameter(s) (default False)
        const_disp (bool): if True, dispertion parameter(s) are only per gene, rather there per gene per sample (default False)
        spatial_coeff (float): coefficient for spatial expression loss in total ELBO (default 1.0)
        sc_coeff (float): coefficient for sinlge cell expression loss in total ELBO (default 1.0)
        cov_coeff (float): coefficient for spatial niche loss in total ELBO (default 1.0)
        kl_coeff (float): coefficient for latent prior loss in total ELBO (default 1.0)
        skip (bool): if True, neural network has skip connections (default True)
        log_input (float): if larger than zero, a log is applied to input with pseudocount of log_input (default 0.0)
        spatial_pc (float): if larger than zero and spatial_dist is 'norm' or 'full_norm', log is applied to spatial_data with pseudocount spatial_pc (default 0.0)
        sc_pc (float): if larger than zero and spatial_dist is 'norm' or 'full_norm', log is applied to spatial_data with pseudocount spatial_pc (default 0.0)
        z_score (float): if True and spatial/sc_dist are 'norm' or 'full_norm', spatial and sinlge cell data are z-scored (default 0.0)
        agg (str or np.array): aggregation function of loss factors, 
                               'mean' will average across neurons, 
                               'sum' will sum across neurons (makes a difference because different number of genes for spatial and sinlge cell data), 
                                np.array will take a per-gene average weighed by elements in np.array (default 'mean')
        init_scale_out (float): scale for VarianceScaling normalization for output layer
        init_scale_enc (float): scale for VarianceScaling normalization for encoding layer
        init_scale_layer (float): scale for VarianceScaling normalization for regular layers
    """ 

            
    def __init__(self,
                 spatial_data, 
                 sc_data, 
                 spatial_key = 'spatial',
                 batch_key = -1,
                 num_layers = 3, 
                 num_neurons = 1024, 
                 latent_dim = 512,
                 k_nearest = 8,
                 num_cov_genes = 64,
                 cov_genes = [],
                 num_HVG = 2048,
                 sc_genes = [],
                 spatial_dist = 'pois',
                 cov_dist = 'OT',
                 sc_dist = 'nb',
                 prior_dist = 'norm', 
                 comm_disp = False,
                 const_disp = False,
                 spatial_coeff = 1,
                 sc_coeff = 1,
                 cov_coeff = 1,
                 kl_coeff = 1, 
                 skip = True, 
                 log_input = 0.0,
                 spatial_pc = None,
                 sc_pc = None,
                 z_score  = False,
                 agg = 'mean', 
                 init_scale_out = 0.1, 
                 init_scale_enc = 0.1, 
                 init_scale_layer = 0.1,
                 **kwargs):
        

        super(ENVI, self).__init__()
        
        
        self.spatial_data = spatial_data
        self.sc_data = sc_data
        
        self.overlap_genes = np.asarray(np.intersect1d(self.spatial_data.var_names, self.sc_data.var_names))
        self.spatial_data = self.spatial_data[:, list(self.overlap_genes)]

    
        sc_data_log = self.sc_data.copy()
        sc.pp.log1p(sc_data_log)
        sc.pp.highly_variable_genes(sc_data_log, n_top_genes = min(num_HVG, sc_data_log.shape[-1]))
        
        if(num_HVG == 0):
            sc_data_log.var.highly_variable = False
        if(len(sc_genes) > 0):
            sc_data_log.var['highly_variable'][np.intersect1d(sc_genes, self.sc_data.var_names)] = True
        
        if(self.sc_data.raw is None):   
            self.sc_data.raw = self.sc_data
            
        self.sc_data = self.sc_data[:, np.union1d(sc_data_log.var_names[sc_data_log.var.highly_variable], self.spatial_data.var_names)]
        
        self.non_overlap_genes = np.asarray(list(set(self.sc_data.var_names) - set(self.spatial_data.var_names)))
        self.sc_data = self.sc_data[:, list(self.overlap_genes) + list(self.non_overlap_genes)]
        
        self.k_nearest = k_nearest
        self.spatial_key = spatial_key
        self.batch_key = batch_key
        
        print("Computing Niche Covariance Matrices")
        
        self.spatial_data.obsm['CovMats'], self.spatial_data.obsm['CovMatsTrans'], self.spatial_data.obsm['NeighExp'], self.CovGenes = GetCov(self.spatial_data, self.k_nearest, num_cov_genes, cov_genes, cov_dist, spatial_key = spatial_key, batch_key = batch_key)
    
    
        self.overlap_num = self.overlap_genes.shape[0]
        self.cov_gene_num = self.spatial_data.obsm['CovMatsTrans'].shape[-1]
        self.full_trans_gene_num = self.sc_data.shape[-1]
    
        self.num_layers = kwargs['NumLayers'] if 'NumLayers' in kwargs.keys() else num_layers 
        self.num_neurons = kwargs['NumNeurons'] if 'NumNeurons' in kwargs.keys() else num_neurons 
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs.keys() else latent_dim 
        
        self.spatial_dist = spatial_dist
        self.cov_dist = cov_dist
        self.sc_dist = sc_dist
        self.comm_disp = comm_disp
        self.const_disp = const_disp
            
        self.prior_dist = prior_dist
        
        self.spatial_coeff = spatial_coeff
        self.sc_coeff = sc_coeff
        self.cov_coeff = cov_coeff
        self.kl_coeff = kl_coeff 
        self.skip = skip
        self.agg = agg


        
        self.enc_layers = []
        self.dec_exp_layers = []
        self.dec_cov_layers = []

        
        if(spatial_pc is not None and (self.spatial_dist == 'norm' or self.spatial_dist == 'full_norm')):
            self.spatial_pc = spatial_pc
            self.spatial_data.uns['log_pc'] = spatial_pc
            self.spatial_data.X = np.log(self.spatial_data.X + self.spatial_data.uns['log_pc'])
            
        if(sc_pc is not None and (self.sc_dist == 'norm' or self.sc_dist == 'full_norm')):
            self.sc_pc = sc_pc
            self.sc_data.uns['log_pc'] = sc_pc
            self.sc_data.X = np.log(self.sc_data.X + self.sc_data.uns['log_pc'])
        
        if(z_score and (self.sc_dist == 'norm' or self.sc_dist == 'full_norm') and (self.spatial_dist == 'norm' or self.spatial_dist == 'full_norm')):
            self.z_score = True
            self.spatial_data.var['mean'] = self.spatial_data.X.mean(axis = 0)
            self.spatial_data.var['std'] = self.spatial_data.X.std(axis = 0)
            
            self.spatial_data.X = (self.spatial_data.X - self.spatial_data.var['mean'][None, :])/self.spatial_data.var['std'][None, :]
            
            self.sc_data.var['mean'] = self.sc_data.X.mean(axis = 0)
            self.sc_data.var['std'] = self.sc_data.X.std(axis = 0)
            
            self.sc_data.X = (self.sc_data.X - self.sc_data.var['mean'][None, :])/self.sc_data.var['std'][None, :]
            
        if(log_input > 0 and np.mean(self.sc_data.X > 0) == 1 and np.mean(self.spatial_data.X > 0) == 1):
            self.log_input = kwargs['LogInput'] if 'LogInput' in kwargs.keys() else log_input 
        else:
            self.log_input = 0
        
        self.initializer_layers = tf.keras.initializers.VarianceScaling(scale = init_scale_layer, 
                                                                       mode='fan_out', 
                                                                       distribution='truncated_normal')
       
        
        self.initializer_enc = tf.keras.initializers.VarianceScaling(scale = init_scale_enc, 
                                                                    mode = 'fan_out', 
                                                                    distribution='truncated_normal')
        
        self.initializer_output = tf.keras.initializers.VarianceScaling(scale = init_scale_out, 
                                                                       mode = 'fan_out', 
                                                                       distribution='truncated_normal')
        
        
        print("Initializing VAE")

        for i in range(self.num_layers - 1):

            self.enc_layers.append(tf.keras.layers.Dense(units = self.num_neurons, 
                                                        kernel_initializer = self.initializer_layers, 
                                                        bias_initializer = self.initializer_layers))
            
            self.dec_exp_layers.append(tf.keras.layers.Dense(units = self.num_neurons, 
                                                           kernel_initializer = self.initializer_layers, 
                                                           bias_initializer = self.initializer_layers))
            
            self.dec_cov_layers.append(tf.keras.layers.Dense(units = self.num_neurons, 
                                                           kernel_initializer = self.initializer_layers, 
                                                           bias_initializer = self.initializer_layers))
        
    
        self.enc_layers.append(tf.keras.layers.Dense(units = 2 * latent_dim, 
                                                    kernel_initializer = self.initializer_enc, 
                                                    bias_initializer = self.initializer_enc))
        
        self.DecOutputLayer = ENVIOutputLayer(input_dim = self.num_neurons, 
                                                 units = self.full_trans_gene_num, 
                                                 spatial_dist = self.spatial_dist, 
                                                 sc_dist = self.sc_dist, 
                                                 comm_disp = self.comm_disp, 
                                                 const_disp = self.const_disp, 
                                                 kernel_init = self.initializer_output, 
                                                 bias_init = self.initializer_output)
        
        self.dec_cov_layers.append(tf.keras.layers.Dense(units = int(self.cov_gene_num * (self.cov_gene_num + 1) / 2), 
                                                       kernel_initializer = self.initializer_output, 
                                                       bias_initializer = self.initializer_output))
        
        print("Finished Initializing ENVI")

    @tf.function
    def encode_nn(self, Input):
        
        """
        Encoder forward pass
        
        Args:
            Input (array): input to encoder NN (size of #genes in spatial data + confounder)
        Returns:
            Output (array): NN output
        """
        
        Output = Input
        for i in range(self.num_layers - 1):
            Output = self.enc_layers[i](Output) + (Output if (i > 0 and self.skip) else 0)
            Output = tf.nn.leaky_relu(Output)
        return(self.enc_layers[-1](Output))

    @tf.function
    def decode_exp_nn(self, Input):
        
        """
        Expression decoder forward pass
        
        Args:
            Input (array): input to expression decoder NN (size of latent dimension + confounder)
            
        Returns:
            Output (array): NN output
        """
        
        Output = Input
        for i in range(self.num_layers - 1):
            Output = self.dec_exp_layers[i](Output) + (Output if (i > 0 and self.skip) else 0)
            Output = tf.nn.leaky_relu(Output)
        return(Output)

    @tf.function
    def decode_cov_nn(self, Output):
        
        """
        Covariance (niche) decoder forward pass
        
        Args:
            Input (array): input to niche decoder NN (size of latent dimension + confounder)
            
        Returns:
            Output (array): NN output
        """
        

        for i in range(self.num_layers - 1):
            Output = self.dec_cov_layers[i](Output) + (Output if (i > 0 and self.skip) else 0)
            Output = tf.nn.leaky_relu(Output)
        return(self.dec_cov_layers[-1](Output))


    @tf.function
    def encode(self, x, mode = 'sc'):
        
        """
        Appends confounding variable to input and generates an encoding
        
        Args:
            x (array): input to encoder (size of #genes in spatial data)
            mode (str): 'sc' for sinlge cell, 'spatial' for spatial data 
            
        Return:
            mean (array): mean parameter for latent variable
            log_std (array): log of the standard deviation for latent variable
        """

        conf_const = 0 if mode == 'spatial' else 1 
        if(self.log_input > 0):
            x = tf.math.log(x + self.log_input)
        x_conf = tf.concat([x, tf.one_hot(conf_const * tf.ones(x.shape[0], dtype=tf.uint8), 2, dtype=tf.keras.backend.floatx())], axis = -1)
        return(tf.split(self.encode_nn(x_conf), num_or_size_splits = 2, axis = 1))

    @tf.function
    def exp_decode(self, x, mode = 'sc'):    
        """
        Appends confounding variable to latent and generates an output distribution
        
        Args:
            x (array): input to expression decoder (size of latent dimension)
            mode (str): 'sc' for sinlge cell, 'spatial' for spatial data 
            
        Return:
            Output paramterizations for chosen expression distributions
        """
        conf_const = 0 if mode == 'spatial' else 1
        x_conf = tf.concat([x, tf.one_hot(conf_const * tf.ones(x.shape[0], dtype=tf.uint8), 2, dtype=tf.keras.backend.floatx())], axis = -1)
        DecOut = self.decode_exp_nn(x_conf)
        
        if (getattr(self, mode + '_dist') == 'zinb'):
            output_r, output_p, output_d = self.DecOutputLayer(DecOut, mode)
            return tf.nn.softplus(output_r), output_p, tf.nn.sigmoid(0.01 * output_d - 2)
        if (getattr(self, mode + '_dist') == 'nb'):
            output_r, output_p = self.DecOutputLayer(DecOut, mode)
            return tf.nn.softplus(output_r), output_p
        if (getattr(self, mode + '_dist') == 'pois'):
            output_l = self.DecOutputLayer(DecOut, mode)
            return tf.nn.softplus(output_l)
        if (getattr(self, mode + '_dist') == 'full_norm'):
            output_mu, output_logstd = self.DecOutputLayer(DecOut, mode)
            return output_mu, output_logstd
        if (getattr(self, mode + '_dist') == 'norm'):
            output_mu =  self.DecOutputLayer(DecOut, mode)
            return output_mu
        
    @tf.function
    def cov_decode(self, x):
        """
        Generates an output distribution for niche data
        
        Args:
            x (array): input to covariance decoder (size of latent dimension)
            
        Return:
            Output paramterizations for chosen niche distributions
        """
        
        DecOut = self.decode_cov_nn(x)
        if(self.cov_dist == 'wish'):
            TriMat = tfp.math.fill_triangular(DecOut)
            TriMat = tf.linalg.set_diag(TriMat, tf.math.softplus(tf.linalg.diag_part(TriMat)))
            return(TriMat)
        elif(self.cov_dist == 'norm'):
            TriMat = tfp.math.fill_triangular(DecOut)
            return(0.5 * TriMat + 0.5 * tf.tranpose(TriMat, [0,2,1]))
        elif(self.cov_dist == 'OT'):
            TriMat = tfp.math.fill_triangular(DecOut)
            return(tf.matmul(TriMat, TriMat, transpose_b = True))
    
    @tf.function
    def enc_mean(self, mean, logstd):
        """
        Returns posterior mean given latent parametrization, which is not the mean varialbe for a log_normal prior
        
        Args:
            mean (array): latent mean parameter
            logstd (array): latent mean parameter
        Return:
            Posterior mean for latent
        """
        if(self.prior_dist == 'norm'):
            return mean
        elif(self.prior_dist == 'log_norm'):
            return tf.exp(mean + tf.square(tf.exp(logstd))/2)
        
    @tf.function
    def reparameterize(self, mean, logstd):
        """
        Samples from latent using te reparameterization trick
        
        Args:
            mean (array): latent mean parameter
            logstd (array): latent mean parameter
        Return:
            sample from latent
        """
        reparm = tf.random.normal(shape=mean.shape, dtype = tf.keras.backend.floatx()) * tf.exp(logstd) + mean
        if(self.prior_dist == 'norm'):
            return reparm
        elif(self.prior_dist == 'log_norm'):
            return tf.exp(reparm)


    @tf.function
    def compute_loss(self, spatial_sample, cov_sample, sc_sample):
        
        """
        Computes ENVI liklihoods
        
        Args:
            spatial_sample (np.array or tf.tensor): spatial expression data sample/batch
            cov_sample (np.array or tf.tensor): niche covariance data sample/batch
            sc_sample (np.array or tf.tensor): single cell data sample/batch subsetted to spatial genes
        Return:
            spatial_like: ENVI liklihood for spatial expression
            cov_like: ENVI liklihood for covariance data
            sc_like: ENVI liklihood for sinlge cell data
            kl: KL divergence between posterior latent and prior
        """
            
        mean_spatial, logstd_spatial = self.encode(spatial_sample[:, :self.overlap_num], mode = 'spatial')
        mean_sc, logstd_sc = self.encode(sc_sample[:, :self.overlap_num], mode = 'sc')
         
        z_spatial = self.reparameterize(mean_spatial, logstd_spatial)
        z_sc = self.reparameterize(mean_sc, logstd_sc)
            
               
        if (self.spatial_dist == 'zinb'):
            spatial_r, spatial_p, spatial_d = self.exp_decode(z_spatial, mode = 'spatial')
            spatial_like = tf.reduce_mean(log_zinb_pdf(spatial_sample, 
                                                       spatial_r[:, :spatial_sample.shape[-1]], 
                                                       spatial_p[:, :spatial_sample.shape[-1]], 
                                                       spatial_d[:, :spatial_sample.shape[-1]], agg = self.agg), axis = 0)
        if (self.spatial_dist == 'nb'):
            spatial_r, spatial_p = self.exp_decode(z_spatial, mode = 'spatial')                                 
            spatial_like = tf.reduce_mean(log_nb_pdf(spatial_sample, 
                                                     spatial_r[:, :spatial_sample.shape[-1]],
                                                     spatial_p[:, :spatial_sample.shape[-1]], agg = self.agg), axis = 0)
        if (self.spatial_dist == 'pois'):
            spatial_l = self.exp_decode(z_spatial, mode = 'spatial')
            spatial_like = tf.reduce_mean(log_pos_pdf(spatial_sample, 
                                                      spatial_l[:, :spatial_sample.shape[-1]], agg = self.agg), axis = 0)
        if (self.spatial_dist == 'full_norm'):
            spatial_mu, spatial_logstd = self.exp_decode(z_spatial, mode = 'spatial')                                         
            spatial_like = tf.reduce_mean(log_normal_pdf(spatial_sample, 
                                                         spatial_mu[:, :spatial_sample.shape[-1]], 
                                                         spatial_logstd[:, :spatial_sample.shape[-1]], agg = self.agg), axis = 0)
        if (self.spatial_dist == 'norm'):
            spatial_mu = self.exp_decode(z_spatial, mode = 'spatial')
            spatial_like = tf.reduce_mean(log_normal_pdf(spatial_sample, 
                                                         spatial_mu[:, :spatial_sample.shape[-1]], 
                                                         tf.zeros_like(spatial_sample), agg = self.agg), axis = 0)
        
        
        if (self.sc_dist == 'zinb'):
            sc_r, sc_p, sc_d = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_zinb_pdf(sc_sample, 
                                                  sc_r, 
                                                  sc_p, 
                                                  sc_d, agg = self.agg), axis = 0) 
        if (self.sc_dist == 'nb'):
            sc_r, sc_p = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_nb_pdf(sc_sample, 
                                                  sc_r, 
                                                  sc_p, agg = self.agg), axis = 0)
        if (self.sc_dist == 'pois'):
            sc_l = self.exp_decode(z_sc, mode = 'sc')
            sc_l = tf.reduce_mean(log_pos_pdf(sc_sample, 
                                              sc_l, agg = self.agg), axis = 0)
        if (self.sc_dist == 'full_norm'):
            sc_mu, sc_std = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_normal_pdf(sc_sample, 
                                                      sc_mu, 
                                                      sc_std, agg = self.agg), axis = 0)
        if (self.sc_dist == 'norm'):
            sc_mu = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_normal_pdf(sc_sample, 
                                                    sc_mu, 
                                                    tf.zeros_like(sc_sample), agg = self.agg), axis = 0)
        
        
        if(self.cov_dist == 'wish'):
            cov_mu = self.cov_decode(z_spatial)
            cov_like = tf.reduce_mean(log_wish_pdf(cov_sample, 
                                                   cov_mu, agg = self.agg), axis = 0)
        elif(self.cov_dist == 'norm'):
            cov_mu = tf.reshape(self.cov_decode(z_spatial), [spatial_sample.shape[0], -1])
            cov_like = tf.reduce_mean(log_normal_pdf(tf.reshape(cov_sample, [cov_sample.shape[0], -1]), 
                                                     cov_mu, 
                                                     tf.zeros_like(cov_mu), agg = self.agg), axis = 0)
        elif(self.cov_dist == 'OT'):
            cov_mu = self.cov_decode(z_spatial)
            cov_like = tf.reduce_mean(OTDistance(cov_sample, 
                                                 cov_mu, agg = self.agg), axis = 0)
    
    
        if(self.prior_dist == 'norm'):
                kl_spatial = tf.reduce_mean(NormalKL(mean_spatial, logstd_spatial, agg = self.agg), axis = 0)
                kl_sc = tf.reduce_mean(NormalKL(mean_sc, logstd_sc, agg = self.agg), axis = 0)
        elif(self.prior_dist == 'log_norm'):
                kl_spatial = tf.reduce_mean(LogNormalKL(logstd_spatial, logstd, agg = self.agg), axis = 0)
                kl_sc = tf.reduce_mean(LogNormalKL(mean_sc, logstd_sc, agg = self.agg), axis = 0)
                
        kl = 0.5 * kl_spatial + 0.5 * kl_sc
    

        return(spatial_like, cov_like, sc_like, kl)
 


    def GetCovMean(self, cov_mat):
        """
        Reconstructs true covarianace (untransformed)
        
        Args:
            cov_mat (array/tensor): transformed covariance matricies to untransform
        Return:
            untransform covariance matrices
        """
        if(self.cov_dist == 'wish'):
            return(cov_mat * tf.sqrt(cov_mat.shape[-1]))
        elif(self.cov_dist == 'OT'):
            return(tf.mamtul(cov_mat, cov_mat))
        else:
            return(cov_mat)
    
    def GetMeanSample(self, decode, mode = 'spatial'):
        """
        Computes mean of expression distribution 
        
        Args:
            decode (list or array/tensor): parameter of distribution 
            mode (str): modality of data to compute distribution mean of (default 'spatial')
        Return:
            distribution mean from parameterization
        """
        if (getattr(self, mode + '_dist') == 'zinb'):
            return(decode[0] * tf.exp(decode[1]) * (1 - decode[2]))
        elif (getattr(self, mode + '_dist') == 'nb'):
            return(decode[0] * tf.exp(decode[1]))
        elif (getattr(self, mode + '_dist') == 'pois'):
            return(decode)
        elif (getattr(self, mode + '_dist') == 'full_norm'):
            return(decode[0])
        elif (getattr(self, mode + '_dist') == 'norm'):
            return(decode)
    
    def cluster_rep(self):
        comm_emb = phenograph.cluster(np.concatenate((self.spatial_data.obsm['envi_latent'], self.sc_data.obsm['envi_latent']), axis = 0))[0]
        
        self.spatial_data.obs['latent_cluster'] = comm_emb[:self.spatial_data.shape[0]]
        self.sc_data.obs['latent_cluster'] = comm_emb[self.spatial_data.shape[0]:]
    
    def latent_rep(self, NumDiv = 16): 
        """
        Compute latent embeddings for spatial and single cell data
        
        Args:
            NumDiv (int): number of splits for forward pass to allow to fit in gpu
        Return:
            no return, adds 'envi_latent' ENVI.spatial_data.obsm and ENVI.spatial_data.obsm
        """
        self.spatial_data.obsm['envi_latent'] = np.concatenate([self.encode(np.array_split(self.spatial_data.X.astype(tf.keras.backend.floatx()), 
                                                               NumDiv, axis = 0)[_], mode = 'spatial')[0].numpy() for _ in range(NumDiv)], axis = 0)

        
        self.sc_data.obsm['envi_latent'] = np.concatenate([self.encode(np.array_split(self.sc_data[:, self.spatial_data.var_names].X.astype(tf.keras.backend.floatx()), 
                                                              NumDiv, axis = 0)[_], mode = 'sc')[0].numpy() for _ in range(NumDiv)], axis = 0)
        
        
        
    def pred_type(self, pred_on = 'sc', key_name = 'cell_type', ClassificationModel = sklearn.neural_network.MLPClassifier(alpha=0.0, max_iter = 100, verbose = False)):
        """
        Transfer labeling from one modality to the other using latent embeddings
        
        Args:
            pred_on (str): what modality to predict labeling for (default 'sc', i.e. transfer from spatial_data to single cell data)
            key_name (str): obsm key name for labeling (default 'cell_type')
            ClassificationModel (sklearn model): Classification model to learn cell labelings (defualt sklearn.neural_network.MLPClassifier)
        Return:
            no return, adds key_name with cell labelings to ENVI.spatial_data.obsm or ENVI.spatial_data.obsm, depending on pred_on
        """
                    
        if(pred_on == 'sc'):
            ClassificationModel.fit(self.spatial_data.obsm['envi_latent'], self.spatial_data.obs[key_name])
            self.sc_data.obs[key_name] = ClassificationModel.predict(self.sc_data.obsm['envi_latent']) 
            
            print("Finished Transfering labels to single cell data! See " +  key_name +" in obsm of ENVI.sc_data")
        else:
            ClassificationModel.fit(self.sc_data.obsm['envi_latent'], self.sc_data.obs[key_name])
            self.spatial_data.obs[key_name] = ClassificationModel.predict(self.spatial_data.obsm['envi_latent'])  
            
            print("Finished Transfering labels to spatial data! See " +  key_name +" in obsm of ENVI.spatial_data")

    
    def impute(self, NumDiv = 16, return_raw = True):
        """
        Imput full transcriptome for spatial data
        
        Args:
            NumDiv (int): number of splits for forward pass to allow to fit in gpu
            return_raw (bool): if True, un-logs and un-zcores imputation if either were chosen
        Return:
            no return, adds 'imputation' to ENVI.spatial_data.obsm
        """
            
        decode = np.concatenate([self.GetMeanSample(self.exp_decode(np.array_split(self.spatial_data.obsm['envi_latent'], NumDiv, axis = 0)[_], mode = 'sc'), mode = 'sc').numpy() for _ in range(NumDiv)], axis = 0)
        
        imputation = pd.DataFrame(decode, columns = self.sc_data.var_names, index = self.spatial_data.obs_names)
        
        if(return_raw):
            if(hasattr(self, 'z_score')):
                imputation = imputation * self.sc_data.var['std'] + self.sc_data.var['mean']
            if(hasattr(self, 'sc_pc')):
                imputation = np.exp(imputation) - self.sc_data.uns['log_pc']
                imputation[imputation < 0] = 0
                
        self.spatial_data.obsm['imputation'] = imputation
        
        print("Finished imputing missing gene for spatial data! See 'imputation' in obsm of ENVI.spatial_data")
     
    def infer_cov(self, NumDiv = 16, revert = False):
        """
        Infer covariance niche composition for single cell data
        
        Args:
            NumDiv (int): number of splits for forward pass to allow to fit in gpu
            revert (bool): if True, computes actual covariance, if False, computes transformed covariance (default False)
        Return:
            no return, adds 'CovMatsTrans' or 'CovMats' to ENVI.sc_data.obsm
        """
            
        if(revert):
            decode = np.concatenate([self.GetCovMean(self.cov_decode(np.array_split(self.sc_data.obsm['envi_latent'], NumDiv, axis = 0)[_])) 
                                     for _ in range(NumDiv)], axis = 0)
            self.sc_data.obsm['CovMats'] = decode
        else:
            decode = np.concatenate([self.cov_decode(np.array_split(self.sc_data.obsm['envi_latent'], NumDiv, axis = 0)[_]) 
                                     for _ in range(NumDiv)], axis = 0)
            self.sc_data.obsm['CovMatsTrans'] = decode
    
        
    
    def reconstruct_niche(self, k = 32, niche_key = 'cell_type', pred_key = 'cell_type', gpu = True):
        
        """
        Infer niche composition for single cell data
        
        Args:
            k (float): k for kNN regression on covariance matrices (defulat 32)
            key (str): spaital obsm key to reconstruct niche from (default 'cell_type')
            pred_key (str): spatial & single cell obsm key to split up kNN regression by (default 'cell_type')
            gpu (bool): if True, uses gpu for kNN regression
        Return:
            no return, adds 'niche_by_type' to ENVI.sc_data.obsm
        """
            
            
        if('CovMatsTrans' not in self.sc_data.obsm.keys()):
            self.infer_cov(16, False)
        
        import sklearn.preprocessing 
        
        LabelEnc = sklearn.preprocessing.LabelBinarizer().fit(self.spatial_data.obs[niche_key])
        spatialCellTypeEncoding = LabelEnc.transform(self.spatial_data.obs[niche_key])
        self.spatial_data.obsm[niche_key + '_enc'] = spatialCellTypeEncoding
        CellTypeName = LabelEnc.classes_
          
        NeighCellType = GetNeighExp(self.spatial_data, self.k_nearest, 
                                    data_key = (niche_key + '_enc'), spatial_key = self.spatial_key, batch_key = self.batch_key)
        
        NeighCellType = NeighCellType.sum(axis = 1).astype('float32')
        
        self.spatial_data.obsm['niche_by_type'] = pd.DataFrame(NeighCellType, columns = CellTypeName, index = self.spatial_data.obs_names)

        if(gpu):
            import cuml.neighbors
            regressor = cuml.neighbors.KNeighborsRegressor(n_neighbors = k)
        else:
            import sklearn.neighbors
            regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = k)
        
        
        if(pred_key is None):
            CovFit = self.spatial_data.obsm['CovMatsTrans']
            CovPred = self.sc_data.obsm['CovMatsTrans']
            
            NeighFit = self.spatial_data.obsm['niche_by_type']
            
            regressor.fit(CovFit.reshape([CovFit.shape[0], -1]), NeighFit)
            NeighPred = regressor.predict(CovPred.reshape([CovPred.shape[0], -1]))
            self.sc_data.obsm['niche_by_type'] = pd.DataFrame(NeighPred, columns = CellTypeName, index = self.sc_data.obs_names)
        else:
        
            NeighPred = np.zeros(shape = (self.sc_data.shape[0], NeighCellType.shape[-1]))
            for val in np.unique(self.sc_data.obs[pred_key]):
                CovFit = self.spatial_data.obsm['CovMatsTrans'][self.spatial_data.obs[pred_key] == val]
                CovPred = self.sc_data.obsm['CovMatsTrans'][self.sc_data.obs[pred_key] == val]

                NeighFit = self.spatial_data.obsm['niche_by_type'][self.spatial_data.obs[pred_key] == val]

                regressor.fit(CovFit.reshape([CovFit.shape[0], -1]), NeighFit)
                NeighPred[self.sc_data.obs[pred_key] == val] = regressor.predict(CovPred.reshape([CovPred.shape[0], -1]))
            
            self.sc_data.obsm['niche_by_type'] = pd.DataFrame(NeighPred, columns = CellTypeName, index = self.sc_data.obs_names)
    
        print("Finished Niche Reconstruction! See 'niche_by_type' in obsm of ENVI.sc_data")
    
    @tf.function
    def compute_apply_gradients(self, spatial_sample, cov_sample, sc_sample):
                
        """
        Applies gradient descent step given training batch
        
        Args:
            spatial_sample (np.array or tf.tensor): spatial expression data sample/batch
            cov_sample (np.array or tf.tensor): niche covariance data sample/batch
            sc_sample (np.array or tf.tensor): single cell data sample/batch subsetted to spatial genes
        Return:
            spatial_like: ENVI liklihood for spatial expression
            cov_like: ENVI liklihood for covariance data
            sc_like: ENVI liklihood for sinlge cell data
            kl: KL divergence between posterior latent and prior
            nan: True if any factor in loss was nan and doesn't apply gradients
        """
            
        with tf.GradientTape() as tape:
            spatial_like,  cov_like, sc_like, kl = self.compute_loss(spatial_sample, cov_sample, sc_sample)
            loss = - self.spatial_coeff * spatial_like - self.sc_coeff * sc_like - self.cov_coeff * cov_like + 2 * self.kl_coeff * kl
        gradients = tape.gradient(loss, self.trainable_variables)
        t3 = time.time()
        if(tf.math.is_nan(loss)):
            return(spatial_like,  cov_like, sc_like, kl, True)
        else:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return (spatial_like,  cov_like, sc_like, kl, False)

    def Train(self, LR = 0.0001, batch_size = 512, epochs = np.power(2,14), split = 1.0, verbose = 128, LR_Sched = True):
                        
        """
        ENVI training loop and computing of latent representation at end
        
        Args:
            LR (float): learning rate for training (default 0.0001)
            batch_size (int): total batch size for traning, to sample from single cell and spatial data (default 512)
            epochs (int): number of training steps (default 16384)
            split (float): train/test split of data (default 1.0)
            verbose (int): how many training step between each print statement, if -1 than no printing (default 128)
            LR_Sched (bool): if True, decreases LR by factor of 10 after 0.75 * epochs steps (default True)
        Return:
            no return, trains ENVI and adds 'envi_latent' to obsm of ENVI.sc_data and ENVI.spatial_data
        """
            
        print("Training ENVI for {} steps".format(epochs))

        self.LR = LR
        self.optimizer = tf.keras.optimizers.Adam(self.LR)
        
        self.train_spatial = np.sort(np.random.choice(np.arange(self.spatial_data.shape[0]),  int(split * self.spatial_data.shape[0]), replace = False))
        self.train_sc = np.sort(np.random.choice(np.arange(self.sc_data.shape[0]), int(split * self.sc_data.shape[0]), replace = False))
        
        spatial_data_train = self.spatial_data.X[self.train_spatial].astype(tf.keras.backend.floatx())
        cov_data_train = self.spatial_data.obsm['CovMatsTrans'][self.train_spatial].astype(tf.keras.backend.floatx())
        sc_data_train = self.sc_data.X[self.train_sc].astype(tf.keras.backend.floatx())
            
        if(split < 1.0):
            
            self.test_spatial = np.delete(np.arange(self.spatial_data.shape[0]), self.train_spatial)
            
            self.test_sc = np.delete(np.arange(self.sc_data.shape[0]), self.train_sc)
            
            spatial_data_test = self.spatial_data.X[self.train_spatial].astype(tf.keras.backend.floatx())
            cov_data_test = self.spatial_data.obsm['CovMatsTrans'][self.train_spatial].astype(tf.keras.backend.floatx())
            sc_data_test = self.sc_data.X[self.train_sc].astype(tf.keras.backend.floatx())
        

        
        
        start_time = time.time()
        
        for epoch in range(0, epochs + 1):

            
            
            if((epoch == int(epochs * 0.75)) and LR_Sched):
                self.LR = LR * 0.1
                self.optimizer.lr.assign(self.LR)

            self.batch_spatial = np.random.choice(np.arange(self.train_spatial.shape[0]),
                                        min(batch_size, self.train_spatial.shape[0]), replace = False)
            
            self.batch_sc = np.random.choice(np.arange(self.train_sc.shape[0]),
                                   min(batch_size, self.train_sc.shape[0]), replace = False)

            if ((epoch % verbose == 0) and (verbose > 0)):

                
                end_time = time.time()

                loss_spatial, loss_cov, loss_sc, loss_kl = self.compute_loss(spatial_data_train[self.batch_spatial], 
                                                                             cov_data_train[self.batch_spatial], 
                                                                             sc_data_train[self.batch_sc])



                
                print('------- Epoch: {}, time elapse {:.5f} --------'.format(epoch, end_time - start_time))
                print('Trn: spatial Loss: {:.5f}, SC Loss: {:.5f}, Cov Loss: {:.5f}, KL Loss: {:.5f}'.format(loss_spatial.numpy(), 
                                                                                                          loss_sc.numpy(), 
                                                                                                          loss_cov.numpy(), 
                                                                                                          loss_kl.numpy()))
            
                if(split < 1.0):
                
                    self.test_batch_spatial = np.random.choice(np.arange(test_spatial.shape[0]),
                                                     min(batch_size, test_spatial.shape[0]), replace = False)
                    self.test_batch_sc = np.random.choice(np.arange(test_sc.shape[0]),
                                                min(batch_size, test_sc.shape[0]), replace = False)

                    test_loss_spatial, test_loss_cov, test_loss_sc, test_loss_kl = self.compute_loss(spatial_data_test[self.test_batch_spatial], 
                                                                                                     cov_data_test[self.test_batch_spatial], 
                                                                                                     sc_data_test[self.test_batch_sc])

                    print('Test: spatial Loss: {:.5f}, SC Loss: {:.5f}, Cov Loss: {:.5f}, KL Loss: {:.5f}'.format(test_loss_spatial, 
                                                                                                               test_loss_sc, 
                                                                                                               test_loss_cov, 
                                                                                                               test_loss_kl))


                start_time = time.time()

            loss_spatial,  loss_cov, loss_sc, loss_kl, nan = self.compute_apply_gradients(spatial_data_train[self.batch_spatial], 
                                         cov_data_train[self.batch_spatial], 
                                         sc_data_train[self.batch_sc])
            
        print("Finished Training ENVI! - calculating latent embedding, see 'envi_latent' obsm of ENVI.sc_data and ENVI.spatial_data")
    
        self.latent_rep()
            