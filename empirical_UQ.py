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
rng = np.random.default_rng(2727)

# Graph parameters
clusterNum = 1000 # number of fine clusters (total)
clusterNumBig = 100 # number of coarse clusters (total) ; number of fine clusters per coarse cluster is clusterNum / clusterNumBig
clusterSize = 10 # number of units per fine cluster 
clusterSizeBig = (clusterNum // clusterNumBig) * clusterSize # number of units per coarse cluster
n = clusterNum*clusterSize # total population size

# Parameters
beta = 2  # Polynomial degree for outcome model
p = 0.125  # Treatment budget proportion

r = 1  # Number of randomized design replications 
r_graph = 1 # Number of graphs we sample over
r_bootstrap = 10
alpha = 0.05 # 1-alpha confidence level, e.g. alpha = 0.05 means we're constructing 95% confidence intervals
##############################################
def pi_estimate_tte_two_stage_complete(Y, nu, k):
    '''
    Computes TTE estimate from polynomial interpolation.
    :param Y: Potential outcomes, shape (beta+1, r, n)
    :param nu: Expected size of subset U
    :param k: Treatment budget (in units)
    :return: Estimated treatment effect
    '''
    H = complete_coeffs(beta, nu, k)  # Compute polynomial coefficients
    return 1 / nu * H @ np.sum(Y, axis=-1)

# Function to estimate TTE in a two-stage experiment
# TODO: e_tte_hat_given_u possibly unused downstream, consider deleting
def estimate_two_stage(fY,Cl,r,nu,K):
    tte_hat = []
    e_tte_hat_given_u = []

    # Generate treatment and control assignments, compute potential outcomes
    Z,U = complete_staggered_rollout_two_stage(n, Cl, nu, K, r) 
    Y = fY(Z)

    # Compute TTE estimates
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage_complete(Y,nu,int(n*p)))

    # Compute conditional expectations
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, (1/nu)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (nu, tte_hat, e_tte_hat_given_u, Y)

# TODO: check logic for this function that computes TTE estimates on a single coarse cluster
def estimate_two_stage_clusters(Y,Cl,q):
    tte_hat = []
    nu = int(clusterSizeBig * p / q) # recomputed # TODO: should we use the realized value instead? or original value?

    # Get the outcomes Y only from units in Cl; Y shape = (beta+1, r, n)
    Y_cl = Y[:,:,Cl]

    # Compute TTE estimates
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage_complete(Y_cl,nu,int(clusterSizeBig*p)))

    return tte_hat

# Network edge probabilities
p_cross = 0 # probability of edge between two units of different coarse clusters
p_out = 0.01 # probability of edge between two units of different fine clusters but same coarse cluster
p_in = 0.5 # probability of edge between two units of same fine cluster in same coarse cluster
probs = sbm_2level_prob_matrix(p_in, p_out, p_cross, clusterNum, clusterNumBig)

# Parameters with respect to the whole population of size n (used to get first point estimates (PE1))
nus = np.linspace(int(n * p), n, 15, dtype=int) # values of subset U sizes
qs = np.divide((n * p), nus) # stage 2 treatment probabilities 

# Define the FINE clustering 
Cl_fine = []
add = 0
for k in range(clusterNum):
    Cl_fine.append(list(range(k+add,k+add+clusterSize)))
    add+=clusterSize-1

# Define the COARSE clustering
Cl_coarse = []
add = 0
for k in range(clusterNumBig):
    Cl_coarse.append(list(range(k+add,k+add+clusterSizeBig)))
    add+=clusterSizeBig-1

# Initliaizing dictionaries that hold the results
TTE_true_list = {rep: 0 for rep in range(r_graph)}

# e.g. TTE_hat_dict[rep][q] holds a tte estimate (PE1) for graph repetition rep and stage-2 treatment probability q
TTE_hat_dict = {rep: {q:[] for q in qs} for rep in range(r_graph)} # PE1
Bias_dict = {rep: {q:[] for q in qs} for rep in range(r_graph)} # Bias of PE1

# e.g. TTE_hat_dict_Cl[rep][q][c] holds a list of tte estimates on each coarse cluster for a graph repetition rep and stage-2 treatment probability q
TTE_hat_dict_Cl = {rep: {q:[] for q in qs} for rep in range(r_graph)}# PE2
Bias_dict_Cl = {rep: {q:[] for q in qs} for rep in range(r_graph)} # PE2 biases

# Bootstrapping results for each graph repetitiion rep and value of q
bootstrap_res = {rep: {q: {key: 0 for key in ["var", "LB", "UB", "pe2"]} for q in qs} for rep in range(r_graph)}
coverage = {rep: {q: False for q in qs} for rep in range(r_graph)}

for rep in range(r_graph):
    A = SBM(clusterSize, probs, self_loops = True)
    h = homophily_effects(A)
    # G = [A[[i],:].nonzero()[1] for i in range(n)] # adjacency list

    K = np.linspace(0,int(n*p),beta+1,dtype=int)
    fY = pom_ugander_yin(A,h,beta)
    TTE_true = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
    TTE_true_list[rep] = TTE_true

    for (nu,TTE_hat,E_given_U,Y) in Parallel(n_jobs=-2, verbose=5)(delayed(lambda nu : estimate_two_stage(fY,Cl_fine,r,nu,K))(nu) for nu in nus): # type: ignore[misc]
        q = (n*p)/nu
        TTE_hat_dict[rep][q].append(TTE_hat) # list of PE1 for a given q
        Bias_dict[rep][q].append(TTE_hat - TTE_true) # list of biases in estimating PE1 for a given q

        # Estimating TTE on each coarse cluster
        coarse_estimates = [estimate_two_stage_clusters(Y,Cl,q) for Cl in Cl_coarse]
        TTE_hat_dict_Cl[rep][q] = coarse_estimates
        Bias_dict_Cl[rep][q] = coarse_estimates-TTE_true

        # Create empirical distribution based on coarse cluster estimates to get variance estimate and confidence intervals
        # References: https://faculty.washington.edu/yenchic/19A_stat535/Lec10_bootstrap.pdf and https://www2.stat.duke.edu/~banks/111-lectures.dir/lect13.pdf
        # TODO: vectorize the bootstrap loop
        empirical_dist = []
        for i in range(r_bootstrap):
            sample = rng.choice(coarse_estimates, size=len(coarse_estimates), replace=True) # sample coarse cluster estimates with replacement
            empirical_dist.append(np.mean(sample)) # the average of these cluster-level estimates is a point on our empirical distribution

        LB = np.percentile(empirical_dist, 100*alpha/2) # lower bound for (1-alpha)*100 confidence interval 
        UB = np.percentile(empirical_dist, 100*(1-(alpha/2))) # upper bound for (1-alpha)*100 confidence interval

        bootstrap_res[rep][q]["var"] = np.var(empirical_dist, ddof=1) # use sample variance of empirical distribution as our variance estimate # type: ignore[misc]
        bootstrap_res[rep][q]["LB"] = LB # type: ignore[misc]
        bootstrap_res[rep][q]["UB"] = UB # type: ignore[misc]
        bootstrap_res[rep][q]["pe2"] = (UB + LB)/2 # point estimate 2, center of confidence interval # type: ignore[misc]

        coverage[rep][q] = (LB <= TTE_true) and (TTE_true <= UB) # is the true TTE contained in the confidence interval?

for rep in range(r_graph):
    print(f"rep = {rep}")
    for nu in nus:
        q = (n*p)/nu
        print(f"nu = {nu}, coverage = {coverage[rep][q]}: TTE = {round(TTE_true_list[rep],4)} is in [{round(bootstrap_res[rep][q]["LB"],5)},{round(bootstrap_res[rep][q]["UB"],5)}]")
        # print(f"TTE_hat_dict length: {len(TTE_hat_dict[rep][q])}")
        # print(f"TTE_hat_dict_Cl length: {len(TTE_hat_dict_Cl[rep][q])}")
        # print(f"bootstrap_res number of keys: {len(bootstrap_res[rep][q])}")
    print()

data = {"pe1_TTE_hat": TTE_hat_dict,
        "pe1_bias": Bias_dict,
        "cluster_TTE_hat": TTE_hat_dict_Cl,
        "cluster_bias": Bias_dict_Cl,
        "bootstrap_res": bootstrap_res,
        "coverage": coverage}
file = open("data.pkl", "wb")
pickle.dump((data), file)
file.close()

executionTime = (time.time() - startTime)
print('Total runtime in minutes: {}'.format(executionTime/60)) 
    
        