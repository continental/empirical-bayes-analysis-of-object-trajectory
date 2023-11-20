'''
Copyright (c) 2021-2022 Continental AG.

@author: Yue Yao
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Ellipse
from scipy.special import binom
from scipy.stats.distributions import chi2

# Helper class for saving numpy and tf data
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int16):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, tf.Tensor):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
    
    
def extract_traj_data(file_path, 
                      target_num_of_points:int, 
                      traj_type: str, 
                      origin_num_of_points:int = 91, start_idx = None) -> (np.array, np.array, int):
    '''
    Randomly extracts a sub-trajectory from the original 9s trajectory.
    Inputs: 
        file_path: the path of json file from step 1
        origin_num_of_points: the number of sample points in the original 9s trajectory
        target_num_of_points: the number of sample points in the sub-trajectory, should be <= origin_num_of_points
    Outputs:
        T: timestamps
        traj: trajectory with dim [target_num_of_points, 4], features are [x, y, vx, vy]
        start_point_idx: the index where sub-trajectory starts        
    '''
    assert origin_num_of_points >= target_num_of_points
    
    with open(file_path, "r") as read_file:
         data = json.load(read_file)
    
    if start_idx is None:
        start_point_idx = np.random.randint(origin_num_of_points-target_num_of_points +1)
    else:
        start_point_idx = start_idx
        
    end_point_idx = start_point_idx + target_num_of_points
    
    T = np.array(data['timestamp'])[start_point_idx:end_point_idx] - np.array(data['timestamp'])[start_point_idx] # Let timestamp start from zero

    traj_xy = np.array(data[traj_type])[start_point_idx:end_point_idx, :2] - np.array(data[traj_type])[start_point_idx, :2] # Let trajectory starts from (0,0)

    traj_v = np.array(data[traj_type])[start_point_idx:end_point_idx, [4,5]]

    traj = np.concatenate((traj_xy, traj_v), axis = 1)
    return T, traj, start_point_idx


#Expands a vector to a polynomial design matrix: from a constant to the deg-power
def polyBasisScale(x_last, deg):
    #Expands a vector to a polynomial design matrix: from a constant to the deg-power
    return np.diag(np.squeeze((np.column_stack([x_last**deg for deg in range(0, deg+1)]))))

def polynomial_basis_function(x, power):
    return x ** power


def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.array([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args]).T
    
    
def compute_AIC_BIC(nll, deg, dof_in_ob, num_points, intercept = False):
    # Compute Akaike and Bayesian Information Criterion
    
    # Compute Bayesian information criterion
    if intercept:
        degree_of_freedom = dof_in_ob + (2*(deg+1))*(2*(deg+1)+1) / 2
    else:
        degree_of_freedom = dof_in_ob + (2*deg)*(2*(deg)+1) / 2
    bic_score = nll + 0.5 * np.log(num_points) * degree_of_freedom
    
    # Compute Akaike information criterion
    aic_score = nll + degree_of_freedom
    
    return aic_score, bic_score


def calculate_result(degrees, bic_scores, aic_scores, A_list, B_list, losses, best_epochs, lr, optimizer, epochs, batch_size):
    '''
    Formatting the result of empirical analysis
    Inputs: 
        degree: degree of polynomial
        bic_scores: bic scores of polynomials
        aic_scores: aic scores of polynomials
        A_list: model parameter covariance of polynomials
        B_list: observation covariance of polynomials
        losses: negative log likelihood of polynomials
        best_epochs: the epoch with least loss of polynomials
        lr: learning rate
        optimizer:
        epochs:
        batch_size:
    Outputs:
        result: Dict      
    '''
    # BIC Results
    best_bic_deg_idx = np.where(bic_scores == np.amin(bic_scores))[0][0]
    best_bic_deg = degrees[best_bic_deg_idx]
    bic_best_A = A_list[best_bic_deg_idx]
    bic_best_B = B_list[best_bic_deg_idx]
    best_bic = bic_scores[best_bic_deg_idx]
    
    # AIC Results
    best_aic_deg_idx = np.where(aic_scores == np.amin(aic_scores))[0][0]
    best_aic_deg = degrees[best_aic_deg_idx]
    aic_best_A = A_list[best_aic_deg_idx]
    aic_best_B = B_list[best_aic_deg_idx]
    best_aic = aic_scores[best_aic_deg_idx]
    
    last_losses = [loss[-1] for loss in losses]
    
    result = {'degree': degrees,
          'losses': last_losses,
          'A_list': A_list,
          'B_list': B_list,
              
          'bic_scores': bic_scores,
          'best_bic': best_bic,
          'best_bic_A': bic_best_A,
          'best_bic_B': bic_best_B,
          'best_bic_deg': best_bic_deg,
          'best_bic_deg_idx': best_bic_deg_idx,
            
          'aic_scores': aic_scores,
          'best_aic': best_aic,
          'best_aic_A': aic_best_A,
          'best_aic_B': aic_best_B,
          'best_aic_deg': best_aic_deg,
          'best_aic_deg_idx': best_aic_deg_idx,
              
          'lr': lr,
          'optimizer': optimizer,
          'epochs': epochs,
          'best_epochs': best_epochs,
          'batch_size': batch_size
        }
    
    return result

def save_result(folder_dir, file_name, result):  
    with open(folder_dir + '/' + file_name + '.json', "w") as write_file:
        json.dump(result, write_file, cls=NumpyEncoder)
        
        
def plot_losses(losses, degrees, y_lim = (-50,50)):
    fig, ax = plt.subplots()
    for i,loss in enumerate(losses):
        ax.plot(range(len(loss)), loss, label = str(degrees[i]))
    
    ax.set_ylim(y_lim)
    ax.legend()

    plt.show()
    return fig, ax


def project_error(error_x, error_y, vector_tan, vector_norm, return_abs = True):
    error_xy = np.concatenate((error_x[:, :, None], error_y[:, :, None]), axis = 2)
   
    error_lon = np.einsum('ijk,ijk->ij', error_xy, vector_tan) / np.linalg.norm(vector_tan, axis=2)
    error_lat = np.einsum('ijk,ijk->ij', error_xy, vector_norm) / np.linalg.norm(vector_norm, axis=2)
    
    if return_abs:
        return np.abs(error_lon), np.abs(error_lat), error_xy
    
    return error_lon, error_lat, error_xy


def draw_confidence_ellipse(mean, cov, ax, **kwarg):
    d,u = np.linalg.eigh(cov)
    
    confidence = chi2.ppf(0.95,2)
    height, width = 2*np.sqrt(confidence*d)
    angle = np.degrees(np.arctan2(u[1,-1], u[0,-1]))
    
    
    
    ellipse = Ellipse(
        xy=mean, 
        width=width, 
        height=height, 
        angle = angle,
        fill = False,
        **kwarg
    )
    
    ax.add_artist(ellipse)
    
    ax.scatter(*mean, s=100, marker='', **kwarg)