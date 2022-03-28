import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

@tf.function
def LogNormalKL(mean, log_std, agg = None):
    KL = 0.5 * (tf.square(mean) + tf.square(tf.exp(log_std)) - 2 * log_std)
    if(agg is None):
        return(KL)
    if(isinstance(agg, (np.ndarray, np.generic))):
        return(tf.reduce_mean(KL, axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(KL, axis = -1))
    return(tf.reduce_mean(KL, axis = -1))
     
    
@tf.function        
def NormalKL(mean, log_std, agg = None):
    KL = 0.5 * (tf.square(mean) + tf.square(tf.exp(log_std)) - 2 * log_std)
    if(agg is None):
        return(KL)
    if(isinstance(agg, (np.ndarray, np.generic))):
        return(tf.reduce_mean(KL, axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(KL, axis = -1))
    
    return(tf.reduce_mean(KL, axis = -1))

@tf.function    
def log_pos_pdf(sample, l, agg = None):
    log_prob = tfp.distributions.Poisson(rate=l).log_prob(sample)
    if(agg is None):
        return(log_prob)
    if(isinstance(agg, (np.ndarray, np.generic))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function
def log_nb_pdf(sample, r, p, agg = None):
    log_prob = tfp.distributions.NegativeBinomial(total_count=r, logits=p).log_prob(sample)
    if(agg is None):
        return(log_prob)
    if(isinstance(agg, (np.ndarray, np.generic))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function
def log_zinb_pdf(sample, r, p, d, agg = None):
    log_prob = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(probs=tf.stack([d, 1-d], -1)),
        components=[tfp.distributions.Deterministic(loc = tf.zeros_like(d)), tfp.distributions.NegativeBinomial(total_count = r, logits = p)]).log_prob(sample)

    
    if(agg is None):
        return(log_prob)
    if(isinstance(agg, (np.ndarray, np.generic))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function   
def OTDistance(sample, mean, agg = None):
    sample = tf.reshape(sample, [sample.shape[0], -1])
    mean = tf.reshape(mean, [mean.shape[0], -1])
    log_prob = - tf.square(sample - mean)
    if(agg is None):
        return(log_prob)
    if(isinstance(agg, (np.ndarray, np.generic))):
        return(tf.reduce_mean(log_prob, axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function    
def log_normal_pdf(sample, mean, scale, agg = None):
    log_prob = tfp.distributions.Normal(loc = mean, scale = tf.exp(scale)).log_prob(sample)
    if(agg is None):
        return(log_prob)
    if(isinstance(agg, (np.ndarray, np.generic))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function
def trace_log(Mat):
    return(tf.reduce_mean(tf.math.log(tf.linalg.diag_part(Mat)), axis = -1))

@tf.function
def log_wish_pdf(sample, scale, agg = 'mean'):
    if(agg == 'mean'):
        return(tfp.distributions.WishartTriL(df = sample.shape[-1], scale_tril = scale, input_output_cholesky = True).log_prob(sample)/(sample.shape[-1] ** 2))
    elif(agg == 'mean'):
        return(tfp.distributions.WishartTriL(df = sample.shape[-1], scale_tril = scale, input_output_cholesky = True).log_prob(sample))
