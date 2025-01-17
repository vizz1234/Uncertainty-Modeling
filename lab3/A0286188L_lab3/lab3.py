""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: Vishwanath Dattatreya Doddamani
Email: e1237250@u.nus.edu
Student ID: A0286188L
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans
import copy


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    Alphas = np.zeros([len(x_list), len(x_list[0]), n_states])
    Betas = np.zeros([len(x_list), len(x_list[0]), n_states])

    for i,x in enumerate(x_list):

        #alpha
        alpha = np.zeros((len(x), n_states))
        cn = np.zeros(len(x))
        alpha[0] = np.multiply(pi,(1 / (np.sqrt(2 * np.pi * (phi['sigma'] ** 2)))) * np.exp(-((x[0] - phi['mu']) ** 2) / (2 * (phi['sigma'] ** 2))))
        cn[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / cn[0]
        for t in range(1, len(x)):
            for j in range(n_states):
                mu = phi['mu'][j]
                sigma = phi['sigma'][j]
                emission_prob = (1 / (np.sqrt(2 * np.pi * (sigma ** 2)))) * np.exp(-((x[t] - mu) ** 2) / (2 * (sigma ** 2)))
                sec_term = np.sum(np.multiply(alpha[t-1],A[:,j]))
                alpha[t,j] = emission_prob * sec_term
            cn[t] = np.sum(alpha[t])
            alpha[t] = alpha[t] / cn[t]
        Alphas[i] = copy.deepcopy(alpha)
    
        #beta
        beta = np.ones((len(x), n_states))
        for t in range(len(x)-2,-1,-1):
            for j in range(n_states):
                first_term = beta[t+1]
                emission_prob = (1 / (np.sqrt(2 * np.pi * (phi['sigma'] ** 2)))) * np.exp(-((x[t+1] - phi['mu']) ** 2) / (2 * (phi['sigma'] ** 2)))
                third_term = A[j,:]
                beta[t,j] = np.sum(first_term * emission_prob * third_term)
            beta[t] = beta[t] / cn[t+1]
        Betas[i] = copy.deepcopy(beta)

    
        #Gamma
        for t in range(len(x)):
            p_X = np.sum(np.multiply(alpha[t],beta[t]))
            gamma_list[i][t] = (alpha[t] * beta[t]) / p_X
        
        #Xi
        for t in range(len(x) - 1):
            for j in range(n_states):
                for k in range(n_states):
                    first_term = alpha[t][j]
                    mu_k = phi['mu'][k]
                    sigma_k = phi['sigma'][k]
                    sec_term = (1 / (np.sqrt(2 * np.pi * (sigma_k ** 2)))) * np.exp(-((x[t+1] - mu_k) ** 2) / (2 * (sigma_k ** 2)))
                    third_term = A[j,k]
                    fourth_term = beta[t + 1][k]
                    xi_list[i][t][j, k] = first_term * sec_term * third_term * fourth_term
            xi_list[i][t] = xi_list[i][t] / np.sum(xi_list[i][t])

    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    x_list = np.array(x_list)
    gamma_list = np.array(gamma_list)
    xi_list = np.array(xi_list)
    Pi1 = np.zeros([len(x_list),n_states])
    A1 = np.zeros([len(x_list), n_states, n_states])
    Mu1 = np.zeros([len(x_list), n_states])
    Sigma1 = np.zeros([len(x_list), n_states]) 
    for i,x in enumerate(x_list):
        Pi1[i] = gamma_list[i,0]
        Pi1[i] = Pi1[i] / np.sum(Pi1[i])
        for j in range(n_states):
            for k in range(n_states):
                A1[i,j,k] = np.sum(xi_list[i,:,j,k])
        for k in range(n_states):
            first_term = gamma_list[i,:,k]
            sec_term = x_list[i]
            num = np.sum(first_term * sec_term)
            Mu1[i,k] = num
    
    pi = np.mean(Pi1, axis=0)
    norm_factor = np.sum(gamma_list, axis=(0,1))
    z1 = np.sum(A1, axis=0)
    z2 = np.sum(A1, axis=(0,2))
    for l in range(n_states):
        z1[l] = z1[l] / z2[l]
    A = z1
    phi['mu'] = np.sum(Mu1, axis=0) / norm_factor

    for i,x in enumerate(x_list):
        for k in range(n_states):
            first_term = gamma_list[i,:,k]
            sec_term_sig = (x_list[i] - phi['mu'][k]) ** 2
            Sigma1[i,k] = np.sum(first_term * sec_term_sig)

    phi['sigma'] = np.sqrt(np.sum(Sigma1, axis=0) / norm_factor)
    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """
    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(int(n_states), x_list)
    prev_pi = copy.deepcopy(pi)
    prev_A = copy.deepcopy(A)
    prev_phi = copy.deepcopy(phi)
    epoch = 0
    while True:
        gamma_list, xi_list = e_step(x_list, pi, A, phi)
        pi, A, phi = m_step(x_list, gamma_list, xi_list)
        # print(epoch)
        epoch = epoch+1
        if np.all(np.abs(prev_phi['sigma'] - phi['sigma']) < 1e-4) == True and np.all(np.abs(prev_phi['mu'] - phi['mu']) < 1e-4) == True and np.all(np.abs(prev_pi - pi) < 1e-4) == True:
            break
        else:
            prev_pi = copy.deepcopy(pi)
            prev_A = copy.deepcopy(A)
            prev_phi = copy.deepcopy(phi)

    return pi, A, phi
