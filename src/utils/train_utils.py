'''
Copyright (c) 2021-2022 Continental AG.

@author: Yue Yao
'''
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import gc

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import utils.utils as utils


def nll(dist, samples):
    """Calculates the negative log-likelihood for a given distribution
    and a data set."""
    ll = dist.log_prob(samples)
    mask_ll = tf.boolean_mask(ll, tf.math.is_finite(ll))
    ll = tf.where(tf.math.is_finite(ll), ll, [-1000])
    if mask_ll.shape[0] / ll.shape[0] < 0.7:
        print('Too much nan in one batch', mask_ll.shape[0], ll.shape[0] )
    return -tf.reduce_mean(ll)


#@tf.function
def get_loss_and_grads(dist, samples):
    with tf.GradientTape() as tape:
        tape.watch(dist.trainable_variables)
        loss = nll(dist, samples)
    grads = tape.gradient(loss, dist.trainable_variables)

    return loss, grads

def fit_distribution(dist, samples, opti, epoch):
    loss, grads = get_loss_and_grads(dist, samples)

    if tf.math.is_finite(loss):
        opti.apply_gradients(zip(grads, dist.trainable_variables))

    return loss


def build_ego_mvn(alpha, beta_diag, beta_by_diag, phi_t, num_points):
    '''
    build multivariate normal distribution for ego trajectory batch
    Inputs:
        alpha: trainable variable for model parameter covariance
        beta_diag: trainable variable for diagonal entities of observation covariance
        beta_by_diag: trainable variable for non-diagonal entities of observation covariance
        phi_t: batch-wise basis function
        num_points: number of sample points in one trajectory
    Outputs:
        tf mvn distribution
    '''
    def mvn_ego(alpha, beta_diag, beta_by_diag, phi_t):      
        b_by_diag = tf.eye(num_points_in_one_traj, dtype = tf.float64) * tf.math.softplus(beta_diag) * tf.math.tanh(beta_by_diag)
        by_eye = tf.convert_to_tensor([[0,1],[1,0]], dtype=tf.float64)
        b_diag = tf.eye(2*num_points, dtype=tf.float64) * tf.math.softplus(beta_diag)
        b_kron = b_diag  + tf.experimental.numpy.kron(by_eye, b_by_diag)

        cov =   b_kron + (phi_t @ alpha )  @ (tf.transpose(phi_t @ alpha, perm=[0, 2, 1]))
        
        return tfd.MultivariateNormalTriL(loc=tf.zeros((2* num_points), dtype = tf.float64), scale_tril=tf.linalg.cholesky(cov))
    
    return tfp.experimental.util.DeferredModule(build_fn=mvn_ego, alpha=alpha, beta_diag=beta_diag, beta_by_diag=beta_by_diag, phi_t = phi_t)

def train_ego(alpha, beta_diag, beta_by_diag, t_scale_factor, degree, opti, data_loader, epochs = 100, tf_summary_writer = None, verbose = False, early_stop = True):
    model_losses = []
    best_alpha = None
    best_beta_diag, best_beta_by_diag = None, None
    best_epoch_loss = np.inf
    best_epoch = 0
    
    # Start training
    for epoch in tqdm(range(epochs)):
        batch_losses = []
        for timestamp_samples, trajectories_samples in data_loader:
            # phi_t is scaled for better numerical stability
            phi_t_batch = utils.expand(timestamp_samples/t_scale_factor, bf=utils.polynomial_basis_function, bf_args=range(1, degree+1)).transpose((1, 0, 2))
            
            if intercept:
                phi_t_kron = np.kron(np.eye(2), phi_t_batch)
            else:
                phi_t_kron = np.kron(np.eye(2), phi_t_batch[:, :, 1:])
            
            # Cast to float64 for better numerical stability
            phi_t_kron = tf.cast(phi_t_kron, dtype = tf.float64)
            trajectories_samples = tf.cast(trajectories_samples, dtype = tf.float64)
            
            # build multivariate normal distribution with learnable variables.
            mvn_test = build_ego_mvn(alpha=alpha, beta_diag=beta_diag, beta_by_diag=beta_by_diag, phi_t = phi_t_kron, num_points = timestamp_samples.shape[1])

            batch_loss = fit_distribution(mvn_test, trajectories_samples, opti,epoch)
            batch_losses.append(batch_loss)
            
            tf.keras.backend.clear_session() # clear the initiated model in this loop
        gc.collect()
            
        assert not tf.math.is_nan(np.mean(batch_losses))
        
        epoch_loss = np.mean(batch_losses)
        
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch = epoch
            best_alpha, best_beta_diag, best_beta_by_diag = deepcopy(alpha), deepcopy(beta_diag), deepcopy(beta_by_diag)
        
        model_losses.append(epoch_loss)
        
        if tf_summary_writer:
            with tf_summary_writer.as_default():
                tf.summary.scalar('loss', np.mean(batch_losses), step=epoch)
        
        # Early stop if epoch loss doesn't decrease for more then 20 epochs 
        if early_stop and epoch - best_epoch >=20:
            print('Early Stop at ' + str(epoch) + '(' + str(best_epoch) + ')' + ' epoch')
            break
        
        if(epoch %10 == 0 and verbose):
            print('Epoch ', epoch, ', Loss: ', model_losses[-1])

        
    return model_losses, best_epoch_loss, best_epoch, best_alpha, best_beta_diag, best_beta_by_diag


def build_agt_mvn(alpha, beta_d, beta_theta, beta_const, phi_t, phi_d, d_norm, theta, R, num_points):
    '''
    build multivariate normal distribution for agt trajectory batch. 
    Covariance from polar to cartesian according to p77 in "N. Kämpchen, “Feature-level fusion of laser scannerand video data for advanced driver assistance systems, 
    Ph.D. dissertation, Universität Ulm, 2007.”

    Inputs:
        alpha: trainable variable for model parameter covariance
        beta_d: trainable variable for distance variance in polar coodinate. dim: (3, 1)
        beta_theta: trainable variable for angular variance in polar coordinate
        beta_const: trainable variable for base variance in ego cartesian coordinate
        phi_t: batch-wise basis function of time
        phi_d: batch-wise basis function of agent distance in polar frame
        d_norm: euclidian distance between ego and agent
        theta (alpha in paper): angle in polar frame
        R: Rotation matrix from ego frame to world frame
        num_points: number of sample points in one trajectory
    Outputs:
        tf distribution
    '''
    def mvn_agt(alpha, beta_d, beta_theta, beta_const, phi_t, phi_d, d_norm, theta, R):      
        var_d = tf.squeeze(phi_d @ tf.math.softplus(beta_d)) # coefficients are all positively defined
        var_theta =  tf.math.softplus(beta_theta)
        
        var_lon_lon = var_d * tf.math.pow(tf.math.cos(theta), 2) + var_theta * tf.math.pow(d_norm, 2) * tf.math.pow(tf.math.sin(theta), 2)
        var_lat_lat = var_d * tf.math.pow(tf.math.sin(theta), 2) + var_theta * tf.math.pow(d_norm, 2) * tf.math.pow(tf.math.cos(theta), 2)
        var_lon_lat = var_d * tf.math.sin(theta) * tf.math.cos(theta) - var_theta *  tf.math.pow(d_norm, 2) * tf.math.sin(theta) * tf.math.cos(theta)
                
        by_eye = tf.convert_to_tensor([[0,1],[1,0]], dtype=tf.float64)
        b_lon_lat_by_diag = tf.experimental.numpy.kron(by_eye, tf.linalg.diag(var_lon_lat))

        b_lon_lat_diag = tf.linalg.diag(tf.concat([var_lon_lon, var_lat_lat], axis = 1))
        b_diag_const = tf.eye((2* num_points), dtype=tf.float64) * tf.math.softplus(beta_const)
        b_lon_lat = b_lon_lat_diag + b_lon_lat_by_diag + b_diag_const

        cov = R @ b_lon_lat @ tf.transpose(R, perm=[0, 2, 1]) + (phi_t @ alpha )  @ (tf.transpose(phi_t @ alpha, perm=[0, 2, 1]))
        
        return tfd.MultivariateNormalTriL(loc=tf.zeros((2* num_points), dtype = tf.float64), scale_tril=tf.linalg.cholesky(cov))

    
    return tfp.experimental.util.DeferredModule(build_fn=mvn_agt, alpha=alpha, beta_d=beta_d, beta_theta=beta_theta, beta_const = beta_const,
                                                phi_t = phi_t, phi_d=phi_d, d_norm=d_norm, theta = theta, R=R)



def train_agt(alpha, beta_d, beta_theta, beta_const, t_scale_factor, degree, opti, data_loader, epochs = 100, model_losses = [], tf_summary_writer = None, verbose = False, early_stop = True):
    best_alpha = None
    best_beta_d, best_beta_theta, best_beta_const = None, None, None
    best_epoch_loss = np.inf
    best_epoch = 0
       
    for epoch in tqdm(range(epochs)):
        batch_losses = []
        for t_samples, agt_traj_samples, d_norm_samples, theta_samples, R_samples in data_loader:
            # phi_t is scaled for better numerical stability
            phi_t_batch = utils.expand(((t_samples)/t_scale_factor), bf=utils.polynomial_basis_function, bf_args=range(1, degree+1)).transpose((1, 0, 2))
            phi_t_kron = np.kron(np.eye(2), phi_t_batch[:, :, 1:])
            
            phi_d = utils.expand(d_norm_samples, bf=utils.polynomial_basis_function, bf_args=range(1, 2+1)).transpose((1, 0, 2))
            
            R_samples = np.transpose(R_samples, (0, 2, 3, 1))
            
            R = np.block([[tf.linalg.diag(R_samples[:, 0, 0]), tf.linalg.diag(R_samples[:, 0, 1])],
                          [tf.linalg.diag(R_samples[:, 1, 0]), tf.linalg.diag(R_samples[:, 1, 1])]])
            
            # Cast to float64 for better numerical stability
            phi_t_kron = tf.cast(phi_t_kron, dtype = tf.float64)
            phi_d = tf.cast(phi_d, dtype = tf.float64)
            R = tf.cast(R, dtype = tf.float64)
            d_norm_samples = tf.cast(d_norm_samples, dtype = tf.float64)
            theta_samples = tf.cast(theta_samples, dtype = tf.float64)
            agt_traj_samples = tf.cast(agt_traj_samples, dtype = tf.float64)
            
            # build multivariate normal distribution with learnable variables.
            mvn_test = build_agt_mvn(alpha=alpha, beta_d=beta_d, beta_theta=beta_theta, beta_const=beta_const, phi_t=phi_t_kron, 
                                     phi_d=phi_d, d_norm = d_norm_samples, theta=theta_samples, R=R, num_points=t_samples.shape[1])

            batch_loss = fit_distribution(mvn_test, agt_traj_samples, opti, epoch)   
            batch_losses.append(batch_loss)
            
            tf.keras.backend.clear_session() # clear the initiated model in this loop
        gc.collect() # clear all cache in this loop, otherwise leads to out of memory issue.
            
        assert not tf.math.is_nan(np.mean(batch_losses))
        
        epoch_loss = np.mean(batch_losses)
        
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch = epoch
            best_alpha, best_beta_d, best_beta_theta,  best_beta_const= deepcopy(alpha), deepcopy(beta_d), deepcopy(beta_theta), deepcopy(beta_const)
        
        model_losses.append(epoch_loss)
        
        if tf_summary_writer:
            with tf_summary_writer.as_default():
                tf.summary.scalar('loss', np.mean(batch_losses), step=epoch)
        
        # Early stop if epoch loss doesn't decrease for more then 20 epochs 
        if early_stop and epoch - best_epoch >=20:
            print('Early Stop at ' + str(epoch) + '(' + str(best_epoch) + ')' + ' epoch')
            break
        
        if(epoch %10 == 0 and verbose):
            print('Epoch ', epoch, ', Loss: ', model_losses[-1])

        
    return best_epoch_loss, best_epoch, best_alpha, best_beta_d, best_beta_theta, best_beta_const