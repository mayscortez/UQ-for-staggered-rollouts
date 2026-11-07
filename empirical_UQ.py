import sys
import os
#sys.path.insert(0, "../")
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from helper import sbm_2level_prob_matrix, SBM, complete_staggered_rollout_two_stage, complete_coeffs, homophily_effects, pom_ugander_yin, clustering_metrics
from joblib import Parallel, delayed 
import pickle
import time
import numpy as np

startTime = time.time()
n = 10000  # Number of units

# Parameters
betas = [2, 3]  # Polynomial degrees for outcome model
p = 0.125  # Treatment budget proportion

# Define values of nu (subset sizes) and q (treatment probabilities)
nus = np.linspace(int(n * p), n, 15, dtype=int)
qs = np.divide((n * p), nus)

r = 1000  # Number of replications

# Stage 1 assignment designs and clustering strategies
stage1_design = 'complete'  # Available assignment designs for stage 1; other options are ['bernoulli','complete'] or ['bernoulli']
cluster_strategy = "p_cross_NONE"  # Different clustering strategies for treatment assignment; all possible strategies are ["p_cross_NONE", "p_cross_SMALL"] 

##############################################

# Initialize dictionary to store results
data = {"beta":[], "q":[], "bias":[], "var":[], "var_s":[]}

def pi_estimate_tte_two_stage_complete(Y, nu, k):
    '''
    Computes TTE estimate from polynomial interpolation.
    :param Y: Potential outcomes, shape (beta+1, r, n)
    :param nu: Expected size of subset U
    :param k: Treatment budget (in units)
    :return: Estimated treatment effect
    '''
    n = Y.shape[-1]
    H = complete_coeffs(beta, nu, k)  # Compute polynomial coefficients
    return 1 / nu * H @ np.sum(Y, axis=-1)

# Function to estimate TTE in a two-stage experiment
def estimate_two_stage(fY,Cl,r,nu,K,cl_design):
    tte_hat = []
    e_tte_hat_given_u = []

    # Generate treatment and control assignments, compute potential outcomes
    Z,U = complete_staggered_rollout_two_stage(n, Cl, nu, K, r) 
    Y = fY(Z)

    # Compute TTE estimates
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage_complete(Y,nu,int(n*p)))

    # Compute conditional expectations
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, (1/nu)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (nu, tte_hat, e_tte_hat_given_u)

# Initialize nested dictionaries for storing results
# e.g. Bias_dict[b][q] constains a list of the biases of the two-stage estimator under a model with degree b, treatment probability q
TTE_hat_dict = {b: {q:[] for q in qs} for b in betas}
Bias_dict = {b: {q:[] for q in qs} for b in betas}
E_given_U_dict = {b: {q:[] for q in qs} for b in betas}

# Construct the network
clusterNum = 1000 # number of fine clusters (total)
clusterNumBig = 100 # number of coarse clusters (total) ; number of fine clusters per coarse cluster is clusterNum / clusterNumBig
clusterSize = 10 # number of units per fine cluster ; number of units per coarse cluster is (clusterNum / clusterNumBig) * clusterSize

p_cross = 0 # probability of edge between two units of different coarse clusters
p_out = 0.01 # probability of edge between two units of different fine clusters but same coarse cluster
p_in = 0.5 # probability of edge between two units of same fine cluster in same coarse cluster

probs = sbm_2level_prob_matrix(p_in, p_out, p_cross, clusterNum, clusterNumBig)
A = SBM(clusterSize, probs, self_loops = True)

G = [A[[i],:].nonzero()[1] for i in range(n)] # adjacency list
h = homophily_effects(A)
clusterings = {"none":[]} # Initialize clustering dictionary

# Define the clustering
Cl = 0 # PLACEHOLDER

# Run experiments for different beta values
for beta in betas:
    print("\nbeta={}".format(beta))

    K = np.linspace(0,int(n*p),beta+1,dtype=int)
    fY = pom_ugander_yin(A,h,beta)
    TTE_true = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

    for (nu,TTE_hat,E_given_U) in Parallel(n_jobs=-2, verbose=5)(delayed(lambda nu : estimate_two_stage(fY,Cl,r,nu,K,stage1_design))(nu) for nu in nus):
        q = (n*p)/nu
        Bias_dict[beta][q].append(TTE_hat - TTE_true)
        TTE_hat_dict[beta][q].append(TTE_hat)
        E_given_U_dict[beta][q].append(E_given_U)
        var_cluster_avgs, effect_of_cut_edges = clustering_metrics(n, Cl, fY)

# save the data
for beta in betas:
    for nu in nus:
        q = (n*p)/nu
        data["q"].append(q)
        data["beta"].append(beta)
        data["bias"].append(np.average(Bias_dict[beta][q]))
        data["var"].append(np.var(TTE_hat_dict[beta][q]))
        data["var_s"].append(np.var(E_given_U_dict[beta][q]))

file = open("basic.pkl", "wb")
pickle.dump((data), file)
file.close()

executionTime = (time.time() - startTime)
print('Total runtime in minutes: {}'.format(executionTime/60)) 
    
        