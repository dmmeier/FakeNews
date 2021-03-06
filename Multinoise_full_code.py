#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:06:36 2018

@author: davidmeier

This is an implementation of the fake news model by DC Brody and DM Meier.
Details below. 

"""



import pickle
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from scipy import integrate as integrate
from scipy.stats import beta

from joblib import Parallel, delayed

def ind(a, b):
    'indicator function'
    b = b*np.ones(a.shape) #do this explicitly; would work automatically, too'
    return 1*(a >= b)

def funex(x, C, m):
    'fake news function; magnitude C, damping constant m, x argument array'
    boolean = (x >=0)
    x = x*boolean
    return C*x*np.exp(-m*x)

def funex_der(x, C, m):
    """
    derivative of fake news function; x argument array
    We set to zero when x < 0; this is the case not of interest
    """
    boolean = (x >=0)
    x = x*boolean
    return C*np.exp(-m*x) - C*m*x*np.exp(-m*x)
    

def dens_tau(mu, x, density): #allow x to be stacked list for noises > 1
    if noises == 1 and integr == 'Riemann':
        if x<0:
            return 'error'
        'density of tau'
        if density == 'exponential':
            if x < shift_par:
                return 0
            elif x >= shift_par:
                return mu*np.exp(-mu*(x-shift_par)) #shifted exponential
        elif density == 'uniform':
            return 1*(x < 1)
    else: #unshifted exponential in multinoise case
        return mu*np.exp(-mu*x)


def info(s, X, t_array, BM, tau, C, m):
    '''info process; t_array the time array, BM the BM array, tau fake news time,
       C magnitude, m damping constant
       X a scalar
       In multinoise case tau is a list of fake news waiting times. 
       '''
    fake_times = np.cumsum(tau, axis = 1)
    fake_news = 0
    for k in range(tau.shape[1]):
        fake_news = fake_news + funex(t_array - fake_times[0, k], C, m) * ind(t_array, fake_times[0, k])
    
    info = s*X*t_array + BM + fake_news
    return info

def exp_expression(xi_n, t_n, tau, s, X, C, m):
    """
    computes the exp expression inside the posterior probability; X can be array
    Returns an array
    
    We subtract X_max in the argument of exp (knowing that this will cancel overall
    in the expression for posterior probabilities) in order to avoid numerical overflows.
    
    Allow tau to be stacked list now. 
    """
    if noises == 1 and integr == 'Riemann':
        X_max = np.max(X)
        arg = xi_n*s*(X - X_max) - 0.5*s**2*X**2*t_n - s*X*funex(t_n - tau, C, m)*ind(t_n, tau)
        return np.exp(arg)
    else:
        X_max = np.max(X)
        #remove fake_news
        
        fake_times = np.cumsum(tau, axis = 1)
        fake_news = 0
        for k in range(tau.shape[1]):
            fake_times_k = fake_times[:, k].reshape(tau.shape[0], 1)
            fake_news = fake_news + funex(t_n - fake_times_k, C, m) * ind(t_n, fake_times_k)
            #fake_news dimension is (tau.shape[0], 1)
        arg = xi_n*s*(X - X_max) - 0.5*s**2*X**2*t_n - s*X*fake_news
        return np.exp(arg)

def post_prob(prior, xi_n, t_n, tau, s, X, C, m):
    """
    X an array of values, prior an array of corresponding 
    returns: An array of posterior probabilities
    """
    if noises == 1 and integr == 'Riemann':
        exp_expr = exp_expression(xi_n, t_n, tau, s, X, C, m)
        denominator = (prior @ (exp_expr).T)[0, 0]
        numerator = prior * exp_expr
        posterior = numerator / denominator
        return posterior
    else:
        exp_expr = exp_expression(xi_n, t_n, tau, s, X, C, m)
        denominator = np.sum(prior * exp_expr, axis = 1, keepdims = True) #shape (tau.shape[0], 1)
        #denominator = (prior @ (exp_expr).T)[0, 0]
        numerator = prior * exp_expr #shape (tau.shape[0], prior.shape[1])
        posterior = numerator / denominator
        return posterior #shape (tau.shape[0], prior.shape[1])

def F(prior, xi_n, t_n, tau, s, X, C, m):
    """
    The F function appearing in the integral
    xi_n and t_n scalars
    X an array
    """
    if noises == 1 and integr == 'Riemann':
        post = post_prob(prior, xi_n, t_n, tau, s, X, C, m)
        return (X @ post.T)[0, 0]
    else:
        post = post_prob(prior, xi_n, t_n, tau, s, X, C, m)
        return np.sum(X * post, axis = 1, keepdims = True) #shape (tau.shape[0], 1)
    

def BM(t_array):
    """
    Create BM for the time array given
    """
    
    n = t_array.shape[1]
    step = t_array[0, 1] - t_array[0, 0]
    increments = np.sqrt(step) * np.random.randn(1, n-1)
    BM = np.cumsum(increments).reshape((1, increments.shape[1]))
    BM = np.insert(BM, 0, 0, axis = 1)
    return BM

def gen_t_array(T, N):
    """
    Create the time array for final time T and N steps
    """
    t_array = np.linspace(0, T, N+1).reshape((1, N+1))
    return t_array

def info_incr(info):
    """
    returns: Array of increments of info process
    """
    n = info.shape[1]
    info_shift = info[0, 1:n]
    info = info[0, 0:n-1]
    increments = info_shift - info
    return increments

def exp_eq33(y, info, t_array, s, X, C, m, index):
    """
    Computing right hand side of equation 33, June 2018 note
    index is the time index relative to which we condition
    X a scalar; y list of increments (although we now allow stacked lists)
    """
    if noises == 1 and integr == 'Riemann':
        step = t_array[0, 1] - t_array[0, 0]
        a = -s*X*step + step*funex_der(t_array - y, C, m)*ind(t_array, y)
        a = a[0, 0:index-1].reshape((1, index-1))
        increments = info_incr(info).reshape(1, info.shape[1]-1)
        b = increments[0, 0:index-1] - a
        return np.exp((-1/(2*step)* b@b.T)[0, 0])
    else:
        fake_times = np.cumsum(y, axis = 1)
        fake_news_der = 0
        for k in range(y.shape[1]):
            fake_times_k = fake_times[:, k].reshape(y.shape[0], 1)
            fake_news_der = fake_news_der + funex_der(t_array - fake_times_k, C, m) * ind(t_array, fake_times_k)
        
        step = t_array[0, 1] - t_array[0, 0]
        a = -s*X*step + step*fake_news_der
        a = a[:, 0:index-1].reshape((y.shape[0], index-1))
        increments = info_incr(info).reshape(1, info.shape[1]-1)
        b = increments[0, 0:index-1] - a
        return np.exp((-1/(2*step) * np.sum(b*b, axis = 1, keepdims = True))) #returning a (y.shape[0], 1) array
        

def tau_post_unnormalized(y, info, t_array, s, X_array, C, m, prior, mu, index):
    """
    Posterior density for tau, but not yet normalized
    index is the time index relative to which we condition
    y stacked list
    """
    if noises == 1 and integr == 'Riemann':
        list_exp = []
        K = X_array.shape[1]
    
        for k in range(K):
            X = X_array[0,k]
            temp = exp_eq33(y, info, t_array, s, X, C, m, index)
            list_exp.append(temp)
    
        exp_array = np.array(list_exp).reshape((1, K))
        debug = dens_tau(mu, y, density)
        return (dens_tau(mu, y, density) * prior @ exp_array.T)[0, 0]
    else:
        sum_exp = 0
        K = X_array.shape[1]
        
        for k in range(K):
            X = X_array[0,k]
            temp = exp_eq33(y, info, t_array, s, X, C, m, index)
            sum_exp = sum_exp + prior[0, k] * temp
        
        #exp_array = np.array(list_exp).reshape((1, K))
        
        prior_densities = dens_tau(mu, y, density) #gives a stacked array which we need to take product of horizontally
        return np.prod(prior_densities, axis = 1, keepdims = True) * sum_exp #array of shape (y.shape[0], 1)

def tau_normalization(info, t_array, s, X_array, C, m, prior, mu, index):
    """
    Normalization constant for the posterior density
    """
    factor, error = integrate.quad(tau_post_unnormalized, 0, 100, limit = 100, args = (info, t_array, s, X_array, C, m, prior, mu, index))
    #print(factor)
    return factor

def tau_normalization_own(info, t_array, s, X_array, C, m, prior, mu, index):
    """
    Normalization constant, but basic Riemann sum integration
    """
    if noises ==1 and integr == 'Riemann' :
        stop = integration_upper_bound
        integral = 0
        N = int(stop * integration_steps_per_unit) #steps_per_unit globally defined
        step = stop/N
        for k in range(N):
            y = k*step
            integral = integral + step * tau_post_unnormalized(y, info, t_array, s, X_array, C, m, prior, mu, index)
        factor = integral
        #testing
        testsum = 0
        test_division = 0
        if debugging == 1:
            for k in range(N):
                y = N*step + k*step
                testsum = testsum + step * tau_post_unnormalized(y, info, t_array, s, X_array, C, m, prior, mu, index)
            #testing smaller division
            for k in range(2*N):
                y = k*step/2
                test_division = test_division + step/2 * tau_post_unnormalized(y, info, t_array, s, X_array, C, m, prior, mu, index)
        partition_error = np.abs((factor - test_division)/factor)
        relerror = np.abs(testsum/factor)
        return factor, relerror, partition_error
    else:
        integration_temps = []
        debug = []
        debug_var = 0
        
        for k in range(MC_integration_outer_steps):
            beg = time.time()
            if k%100 ==0:
                print(k)
            summation = 0
            int_samples = integration_upper_bound * np.random.rand(MC_integration_inner_steps, noises)
            #debugging
            #int_samples = np.array([[k*integration_upper_bound/MC_integration_outer_steps]])
            #debugging
            values = tau_post_unnormalized(int_samples, info, t_array, s, X_array, C, m, prior, mu, index)
            summation = 1/MC_integration_inner_steps * np.sum(values, axis = 0)
            integration_temps.append(summation)
#            if MCgraphs == 1:
#                #debugging
#                debug_var = debug_var + summation
#                debug.append(MC_measure * 1/(k+1) * debug_var)
#                end = time.time()
#                if k%100 ==0:
#                    print(beg - end)
#                if k%100 == 0 and k > 10000:
#                    plt.plot(debug[10000:])
#                    plt.title(debug[k])
#                    plt.show()  
        factor = 1/MC_integration_outer_steps * np.sum(integration_temps)
        return MC_measure * factor, 0, 0 #not returning any errors for now; the measure factor will cancel below

def tau_post(y, info, t_array, s, X_array, C, m, prior, mu, index, factor):
    """
    Posterior density for tau, normalized
    index is the time index relative to which we condition
    """
    if noises == 1 and integr == 'Riemann':
        list_exp = []
        K = X_array.shape[1]
        
        for k in range(K):
            X = X_array[0, k]
            temp = exp_eq33(y, info, t_array, s, X, C, m, index)
            list_exp.append(temp)
        
        exp_array = np.array(list_exp).reshape((1, K))
        return 1/factor * (dens_tau(mu, y, density) * prior @ exp_array.T)[0, 0]
    else:
        sum_exp = 0
        K = X_array.shape[1]
        
        for k in range(K):
            X = X_array[0,k]
            temp = exp_eq33(y, info, t_array, s, X, C, m, index)
            sum_exp = sum_exp + prior[0, k] * temp
        
        #exp_array = np.array(list_exp).reshape((1, K))
        prior_densities = dens_tau(mu, y, density) #gives a stacked array which we need to take product of horizontally
        
        return 1/factor * np.prod(prior_densities, axis = 1, keepdims = True) * sum_exp #array of shape (y.shape[0], 1)
        

def integrand(y, info, t_array, s, X_array, C, m, prior, mu, index):
    """
    Integrand in the final integral; this is a function of y (stacked list)
    index is the time index relative to which we condition (counting from 0!)
    """
    #Comment this out trying to avoid computing tau twice (once for normalization and then
    #in the integrand)
    #taup = tau_post(y, info, t_array, s, X_array, C, m, prior, mu, index, factor)
    #taup has shape (y.shape[0], 1)
    taup = tau_post_unnormalized(y, info, t_array, s, X_array, C, m, prior, mu, index)
    
    xi_n = info[0,index]
    t_n = t_array[0,index]
    F_fun = F(prior, xi_n, t_n, y, s, X_array, C, m) #shape (y.shape[0], 1)
    
    integrand = taup * F_fun
    #integrand = taup #testing
    #integrand = taup #for testing if normalized
    return integrand, taup #shape (y.shape[0], 1) for integrand; also return taup in order to get normalization. 

def cond_exp(info, t_array, s, X_array, C, m, prior, mu, index):
    result, error = integrate.quad(integrand, 0, 50, limit = 100, args = (info, t_array, s, X_array, C, m, prior, mu, index))
    return result

def cond_exp_own(info, t_array, s, X_array, C, m, prior, mu, index, int_samples_in = 0):
    if noises == 1 and integr == 'Riemann':
        #normalization factor
        factor, rel, part = tau_normalization_own(info, t_array, s, X_array, C, m, prior, mu, index)
        stop = integration_upper_bound
        integral = 0
        N = stop * integration_steps_per_unit #steps defined globally
        step = stop/N
        testsum = 0
        test_division = 0
        for k in range(N):
            y = k*step
            integral = integral + step * integrand(y, info, t_array, s, X_array, C, m, prior, mu, index, factor)
        if debugging == 1:
            for k in range(N):
                y = N*step + k*step
                testsum = testsum + step * integrand(y, info, t_array, s, X_array, C, m, prior, mu, index, factor)
            #testing smaller division
            for k in range(2*N):
                y = k*step/2
                test_division = test_division + step/2 * integrand(y, info, t_array, s, X_array, C, m, prior, mu, index, factor)
        relerror = np.abs(testsum/integral)
        partition_error = np.abs((integral - test_division)/integral)
        return integral, relerror, partition_error
    else:
        #Don't need the factor now, because doing it in one go. 
        #factor, _, _  = tau_normalization_own(info, t_array, s, X_array, C, m, prior, mu, index)
        integration_temps = []
        normalization_temps = []
        debug=[]
        debug_normal = []
        debug_var = 0
        debug_var_normal = 0
        normalized_integral = []
        for k in range(MC_integration_outer_steps):
            if k%100 ==0:
                print(k)
            summation = 0
            sum_tau_norm = 0
            if int_samples_in == 0: #then generate own integration samples
                int_samples = integration_upper_bound * np.random.rand(MC_integration_inner_steps, noises)
            else: #int_samples have been passed in as a list of arrays, list index the outer counter
                int_samples = int_samples_in[k]
            #print("This is sample {}".format(int_samples[0, :]))
            values, taup_vals = integrand(int_samples, info, t_array, s, X_array, C, m, prior, mu, index)
            summation = 1/MC_integration_inner_steps * np.sum(values, axis = 0)
            sum_tau_norm = 1/MC_integration_inner_steps * np.sum(taup_vals, axis = 0)
            integration_temps.append(summation)
            normalization_temps.append(sum_tau_norm)
            #Have to uncomment the below on the mac -- otherwise parallel module complains!
            #debugging
#            if MCgraphs == 1:
#                debug_var = debug_var + summation
#                debug.append(MC_measure * 1/(k+1) * debug_var)
#                
#                debug_var_normal = debug_var_normal + sum_tau_norm
#                debug_normal.append(MC_measure * 1/(k+1) * debug_var_normal)
#                
#                normalized_integral.append(debug[k]/debug_normal[k])
#                
#                if k%100 == 0 and k>10000:
#                    plt.plot(normalized_integral[10000:])
#                    plt.title(normalized_integral[k])
#                    plt.show()
                    
        integral = 1/MC_integration_outer_steps * np.sum(integration_temps)
        factor = 1/MC_integration_outer_steps * np.sum(normalization_temps)
        return MC_measure * integral/factor, 0, 0 #not returning any errors for now; the measure factor in fact cancels with the measure factor in 'factor'

def plot_integrand_eq39(start, stop, steps, info, t_array, s, X_array, C, m, prior, mu, index):
      rangearg = np.linspace(start, stop, steps)
      factor, rel, part = tau_normalization_own(info, t_array, s, X_array, C, m, prior, mu, index)
      vals = []
      for i in rangearg:
          vals.append(integrand(i, info, t_array, s, X_array, C, m, prior, mu, index, factor))
      plt.plot(vals)
      plt.title('Integrand equation 39')
      plt.show()
      return 0
  
def plot_exp_eq33(start, stop, steps, which_X, info, t_array, s, X_array, C, m, prior, mu, index):
    rangearg = np.linspace(start, stop, steps)
    vals = []
    for i in rangearg:
        vals.append(exp_eq33(i, info, t_array, s, X_array[0, which_X], C, m, index))
    plt.plot(vals)
    plt.title('Equation 33')
    plt.show()
    return 0

def plot_info(info_path, t_array, C, m, tau):
    plt.plot(info_path[0], label = 'Info process with fake news')
    #clean from fake news
    fake_times = np.cumsum(tau, axis = 1)
    fake_news = 0
    for k in range(tau.shape[1]):
        fake_news = fake_news + funex(t_array - fake_times[0, k], C, m) * ind(t_array, fake_times[0, k])
    
    info_path_cleaned = info_path - fake_news
    plt.plot(info_path_cleaned[0], label = 'True info process')
    plt.title('Info processes')
    plt.legend()
    plt.show()
    return 0

def plot_fake_news(t_array, tau, C, m):
    A = funex(t_array - tau, C, m)*ind(t_array, tau)
    plt.plot(A[0])
    plt.title('Fake news')
    plt.show()

def get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau, int_samples = 0): #can pass in int_samples in case want to fix MC integration variables
    best_estimate, relerror, parterror = cond_exp_own(info_path, t_array, s, X_array, C, m, prior, mu, index, int_samples)
    return best_estimate , relerror, parterror

def get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau):
    C = 0 #setting C to zero means fake news not taken into account
    best_estimate, relerror, parterror = cond_exp_own(info_path, t_array, s, X_array, C, m, prior, mu, index)
    return best_estimate , relerror, parterror

def get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau):
    #clean from fake news
    fake_times = np.cumsum(tau, axis = 1)
    fake_news = 0
    for k in range(tau.shape[1]):
        fake_news = fake_news + funex(t_array - fake_times[0, k], C, m) * ind(t_array, fake_times[0, k])
    
    #First subtract fake news from info_path
    info_path = info_path - fake_news
    #Ignore possibility of fake news
    C = 0
    best_estimate, relerror, parterror = cond_exp_own(info_path, t_array, s, X_array, C, m, prior, mu, index)
    return best_estimate, relerror, parterror
    
def compare_models_tau_deterministic(random_runs, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau, verbose = 0):
    model2 = []
    ignorant = []
    true = []
    true_direct = []
    relerrors = []
    parterrors = []
    for i in range(random_runs):
        if i%10 == 0:
            print(i)
        BM_path = BM(t_array)
        info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
        best_estimate, relerror, parterror = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        #best_estimate1, _, _ = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
        
        #ignorant estimate is true_best_estimate_direct with C = 0
        best_estimate1 = true_model_direct(info_path, t_array, s, 0, m, mu, prior, X_array, T, N, index, which_X, tau)
        #best_estimate2, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        true_direct = 0
        best_estimate2 = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        model2.append(best_estimate)
        ignorant.append(best_estimate1)
        true.append(best_estimate2)
        
        relerrors.append(relerror)
        parterrors.append(parterror)
        if verbose == 1:
            print('Estimates are {} in model 2 and {} in ignorant model'.format(best_estimate, best_estimate1))
            print('\nRelative integration errors are {} for doubling range and {} for halving step size'.format(relerror, parterror))  
    return model2, ignorant, true, relerrors, parterrors, true_direct

def Par_compare_models_tau_deterministic(random_runs,  A): #A is a tuple of the form
    #(t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau) = A
    list_i = range(random_runs)
    list_t = [A for i in range(random_runs)]
    
    results = Parallel(n_jobs=-1)(delayed(Par_helper)(i,t) for i,t in zip(list_i,list_t))
        
    return results

def Par_helper(i, pars):
    print("Doing step {}".format(i))
    (t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau) = pars
    BM_path = BM(t_array)
    info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
    best_estimate, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    #best_estimate1, _, _ = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
    best_estimate1 = true_model_direct(info_path, t_array, s, 0, m, mu, prior, X_array, T, N, index, which_X, tau)
    #best_estimate2, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    #best_estimate3 = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    best_estimate2 = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    return (best_estimate, best_estimate1, best_estimate2, info_path)

def Par_compare_models_tau_random(random_runs,  A): #A is a tuple of the form
    #(t_array, s, C, m, mu, prior, X_array, T, N, index, which_X) = A
    list_i = range(random_runs)
    list_t = [A for i in range(random_runs)]
    
    results = Parallel(n_jobs=-1)(delayed(Par_helper_tau_random)(i,t) for i,t in zip(list_i,list_t))
        
    return results

def Par_helper_tau_random(i, pars):
    print("Doing step {}".format(i))
    (t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, Nfake_rand) = pars
    if density == 'exponential':
        #assuming exponential distribution for tau, with parameter mu
        if Nfake_rand == 1:
            tau = np.random.exponential(1/mu) + shift_par
        else:
            tau = np.random.exponential(1/mu, (1, Nfake_rand))
            #print("Tau is {}".format(tau))
    elif density == 'uniform':
        #assuming uniform
        tau = np.random.rand()
#    if density == 'exponential':
#        #assuming exponential distribution for tau, with parameter mu
#        tau = np.random.exponential(1/mu) + shift_par
#    elif density == 'uniform':
#        #assuming uniform
#        tau = np.random.rand()
    BM_path = BM(t_array)
    info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
    #best_estimate, relerror, parterror = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    #best_estimate1, _, _ = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
    #best_estimate2, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    #best_estimate3 = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    best_estimate, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    best_estimate1 = true_model_direct(info_path, t_array, s, 0, m, mu, prior, X_array, T, N, index, which_X, tau)
    best_estimate2 = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    return (best_estimate, best_estimate1, best_estimate2, tau, info_path)

def compare_models_tau_random(random_runs, t_array, s, m, mu, prior, X_array, T, N, index, which_X, verbose = 0, Nruns_rand = 1):
    model2 = []
    ignorant = []
    true = []
    true_direct = []
    relerrors = []
    parterrors = []
    taus = []
    for i in range(random_runs):
        if i % 10 == 0:
            print(i)
        if density == 'exponential':
            #assuming exponential distribution for tau, with parameter mu
            if Nfake_rand == 1:
                tau = np.random.exponential(1/mu) + shift_par
            else:
                tau = np.random.exponential(1/mu, (1, Nfake_rand))
        elif density == 'uniform':
            #assuming uniform
            tau = np.random.rand()
        taus.append(tau)
        BM_path = BM(t_array)
        info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
        best_estimate, relerror, parterror = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        #best_estimate1, _, _ = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
        
        #ignorant estimate is true_best_estimate_direct with C = 0
        best_estimate1 = true_model_direct(info_path, t_array, s, 0, m, mu, prior, X_array, T, N, index, which_X, tau)
        #best_estimate2, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        true_direct = 0
        best_estimate2 = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        
        
        
#        best_estimate, relerror, parterror = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
#        best_estimate1, _, _ = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
#        best_estimate2, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
#        best_estimate3 = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        model2.append(best_estimate)
        ignorant.append(best_estimate1)
        true.append(best_estimate2)
        #true_direct.append(best_estimate3)
        relerrors.append(relerror)
        parterrors.append(parterror)
        if verbose == 1:
            print('Estimates are {} in model 2 and {} in ignorant model'.format(best_estimate, best_estimate1))
            print('\nRelative integration errors are {} for doubling range and {} for halving step size'.format(relerror, parterror))  
    return model2, ignorant, true, relerrors, parterrors, true_direct, taus


def percentage_gains(model2, ignorant, true):#takes lists
    M2 = np.array(model2)
    I = np.array(ignorant)
    T = np.array(true)
    Diff = M2 - T
    Num = I - T
    per = np.abs(M2-T)/np.abs(I-T + 1e-100) #percentage remaining for M2 of error of I
    #percentage gain where improved
    improvement = 1*(per < 1)
    ones_ar = np.ones(M2.shape) * (per < 1)
    per_gain = ones_ar - per
    
    worsening = 1*(per > 1) #gives the factor of how much worse it is
    
    
    return 100*(per_gain + worsening)

def percentage_gains_ignorant(model2, ignorant, true): #flip things around
    M2 = np.array(model2)
    I = np.array(ignorant)
    T = np.array(true)
    Diff = I - T
    Num = M2 - T
    per = np.abs(I-T)/np.abs(M2-T + 1e-100) #percentage remaining for M2 of error of I
    #percentage gain where improved
    improvement = 1*(per < 1)
    ones_ar = np.ones(M2.shape) * (per < 1)
    per_gain = ones_ar - per
    
    worsening = 1*(per > 1) #gives the factor of how much worse it is
    
    
    return 100*(per_gain + worsening)

def absolute_gains(model2, ignorant, true):
    M2 = np.array(model2)
    I = np.array(ignorant)
    T = np.array(true)
    curve1 = np.abs(M2 - T)
    curve2 = np.abs(I - T)
    return curve1, curve2

def true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau):
    #Direct coding
    if noises == 1 and integr == 'Riemann':
        #First subtract fake news from info_path
        info_path = info_path - funex(t_array - tau, C, m)*ind(t_array, tau)
        info_current = info_path[0, index]
        time_current = t_array[0,index]
        list_exp = []
        K = X_array.shape[1]
        
        for k in range(K):
            X = X_array[0, k]
            temp = np.exp(info_current*s*X - 0.5 * s**2 * X**2 * time_current)
            list_exp.append(temp)
        
        exp_array = np.array(list_exp).reshape((1, K))
        denominator = prior @ exp_array.T
        numerator = X_array @ (prior * list_exp).T
        best_estimate = numerator / denominator
        
        return best_estimate[0, 0]
    else:
        #clean from fake news
        fake_times = np.cumsum(tau, axis = 1)
        fake_news = 0
        for k in range(tau.shape[1]):
            fake_news = fake_news + funex(t_array - fake_times[0, k], C, m) * ind(t_array, fake_times[0, k])
        
        info_path = info_path - fake_news
        info_current = info_path[0, index]
        time_current = t_array[0,index]
        
        list_exp = []
        K = X_array.shape[1]
        
        for k in range(K):
            X = X_array[0, k]
            temp = np.exp(info_current*s*X - 0.5 * s**2 * X**2 * time_current)
            list_exp.append(temp)
        
        exp_array = np.array(list_exp).reshape((1, K))
        denominator = prior @ exp_array.T
        numerator = X_array @ (prior * list_exp).T
        best_estimate = numerator / denominator
        
        return best_estimate[0, 0]
        

def single_simulation(t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau, plotting = 1):
    BM_path = BM(t_array)
    info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
    
    if plotting == 1:
        plot_fake_news(t_array, tau, C, m)
        plot_info(info_path, t_array, C, m, tau)
        plot_exp_eq33(0, 1, 1000, which_X, info_path, t_array, s, X_array, C, m, prior, mu, index)
        plot_integrand_eq39(0, 1, 1000, info_path, t_array, s, X_array, C, m, prior, mu, index)
    
    best_estimate, relerror, parterror = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    best_estimate1, relerror1, parterror1 = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
    best_estimate2, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    if plotting == 1 :
        print('Estimates are {} in model 2 and {} in ignorant model'.format(best_estimate, best_estimate1))
        print('\nRelative integration errors are {} for doubling range and {} for halving step size'.format(relerror, parterror))  
    return best_estimate, best_estimate1, best_estimate2

def compare_models_analysis_tau_deterministic(sim_runs, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau, verbose = 0):
    model2, ignorant, true, relerrors, parterrors, true_direct = compare_models_tau_deterministic(sim_runs, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau, verbose)

    plt.plot(model2, label='Model 2')
    plt.plot(ignorant, label = 'Ignorant')
    plt.plot(true, label = 'True')
    #plt.plot(true_direct, label = 'True direct')
    plt.legend()
    plt.show()
    
    difference = np.abs(np.array(true_direct) - np.array(true))
    plt.plot(difference)
    plt.title('Is implementation of true correct? Although now true set to 0, so graph not relevant. ')
    plt.show()
    
    plt.plot(relerrors, label = 'Error doubling range')
    plt.plot(parterrors, label = 'Error halving steps')
    plt.legend()
    plt.show()
    
    gains = percentage_gains(model2, ignorant, true)
    plt.plot(gains)
    plt.title('Percentage gains')
    plt.show()
    
    absolute1, absolute2 = absolute_gains(model2, ignorant, true)
    plt.plot(absolute1, label = 'Model2 difference to True')
    plt.plot(absolute2, label = 'Ignorant difference to True')
    plt.title('Absolute differences')
    plt.legend()
    plt.show()
    
    plt.hist(gains[gains > 0], bins = 20)
    plt.title('Percentage gains in {} cases'.format(np.sum(gains>0)))
    plt.show()
    
    plt.hist(gains[gains < 0], bins = 20)
    plt.title('Percentage losses in {} cases'.format(np.sum(gains<0)))
    plt.show()
    
    gains_ignorant = percentage_gains_ignorant(model2, ignorant, true)
    plt.hist(gains_ignorant[gains<0], bins = 20)
    plt.title('Percentage gain of ignorant in {} cases'.format(np.sum(gains<0)))
    plt.show()
    
    
def compare_models_analysis_tau_random(sim_runs, t_array, s, m, mu, prior, X_array, T, N, index, which_X, verbose = 0):
    model2, ignorant, true, relerrors, parterrors, true_direct, taus = compare_models_tau_random(sim_runs, t_array, s, m, mu, prior, X_array, T, N, index, which_X, verbose)

    plt.plot(model2, label='Model 2')
    plt.plot(ignorant, label = 'Ignorant')
    plt.plot(true, label = 'True')
    #plt.plot(true_direct, label = 'True direct')
    plt.plot(taus, label = 'Taus')
    plt.legend()
    plt.show()
    
    difference = np.abs(np.array(true_direct) - np.array(true))
    plt.plot(difference)
    plt.title('Is implementation of true correct?')
    plt.show()
    
    plt.plot(relerrors, label = 'Error doubling range')
    plt.plot(parterrors, label = 'Error halving steps')
    plt.legend()
    plt.show()
    
    gains = percentage_gains(model2, ignorant, true)
    plt.plot(gains)
    plt.title('Percentage gains')
    plt.show()
    
    absolute1, absolute2 = absolute_gains(model2, ignorant, true)
    plt.plot(absolute1, label = 'Model2 difference to True')
    plt.plot(absolute2, label = 'Ignorant difference to True')
    diff = absolute1-absolute2
    worst = np.max(diff[absolute1>absolute2])
    plt.title('Absolute differences, max loss is {}'.format(worst))
    plt.legend()
    plt.show()
    
    plt.hist(gains[gains > 0], bins = 20)
    plt.title('Percentage gains in {} cases'.format(np.sum(gains>0)))
    plt.show()
    
    plt.hist(gains[gains < 0], bins = 20)
    plt.title('Percentage losses in {} cases'.format(np.sum(gains<0)))
    plt.show()
    
    gains_ignorant = percentage_gains_ignorant(model2, ignorant, true)
    plt.hist(gains_ignorant[gains<0], bins = 20)
    plt.title('Percentage gain of ignorant in {} cases'.format(np.sum(gains<0)))
    plt.show()
    
    taus_ar = np.array(taus)
    plt.hist(taus_ar[gains<0], bins = 20)
    plt.title('Taus where loss occurs')
    plt.show()
    
    plt.hist(taus)
    plt.title('Tau histogram')
    plt.show()
    
def prediction_curves(Ngrid, t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau):
    BM_path = BM(t_array)
    info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
    
    plot_info(info_path, t_array, C, m, tau)
    
    #Ngrid the resolution of the prediction curve on the time axis
    conditioning_grid = np.linspace(1/Ngrid, 1, Ngrid)
    model2_list = []
    ignorant_list = []
    true_list = []
    for i in conditioning_grid:
        index = int(N * i)
        #if index%100 == 0:
            #print(index)
        model2, relerror, parterror = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        ignorant, relerror1, parterror1 = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
        true, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        model2_list.append(model2)
        ignorant_list.append(ignorant)
        true_list.append(true)
    
    plt.plot(model2_list, label = 'Model 2')
    plt.plot(ignorant_list, label = 'Ignorant')
    plt.plot(true_list, label = 'True')
    plt.legend()
    plt.show()
    return model2_list, ignorant_list, true_list

def Par_prediction_curves(Ngrid, A): #A a tuple of the form
    #(t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau) = A
    (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau) = A
    BM_path = BM(t_array)
    info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
    conditioning_grid = np.linspace(1/Ngrid, 1, Ngrid)
    B = (info_path, A)
    
    #debug
    #K = Par_prediction_helper(conditioning_grid[0], B)
    #debug 
    
    list_t = [B for i in conditioning_grid]
    
    results = Parallel(n_jobs=-1)(delayed(Par_prediction_helper)(i,t) for i,t in zip(conditioning_grid,list_t))  
    return results, info_path

    
def Par_prediction_helper(gridpoint, B): #B tuple of form (info_path, A) where A tuple of the form
    #(t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau) = A
    info_path = B[0]
    A = B[1]
    (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau) = A
    index = int(N * gridpoint)
    model2, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    #ignorant, _, _ = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
    ignorant = true_model_direct(info_path, t_array, s, 0, m, mu, prior, X_array, T, N, index, which_X, tau)
    #true, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    true = true_model_direct(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
    return (model2, ignorant, true)

def Par_prediction_plotting(t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, data): #data a tuple
    #of the form (predictions, info_path)
    model2 = []
    ignorant = []
    true = []
    predictions = data[0]
    info_path = data[1]
    for k in range(len(predictions)):
        current = predictions[k]
        model2v = current[0]
        ignorantv = current[1]
        truev = current[2]
        model2.append(model2v)
        ignorant.append(ignorantv)
        true.append(truev)
        
    plt.plot(model2, label = 'Model 2')
    plt.plot(ignorant, label = 'Ignorant')
    plt.plot(true, label = 'True')
    plt.legend()
    plt.show()
    
    plot_info(info_path, t_array, C, m, tau) #plotting info processes
        

def Par_compare_models_plotting_tau_random(t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, data): #data a tuple
    model2= []
    ignorant= []
    true = []
    taus = []
    for k in range(len(data)):
        model2v, ignorantv, truev, tau, info_path = data[k]
        model2.append(model2v)
        ignorant.append(ignorantv)
        true.append(truev)
        taus.append(tau)
    
    
    plt.plot(model2, label='Model 2')
    plt.plot(ignorant, label = 'Ignorant')
    plt.plot(true, label = 'True')
    #plt.plot(true_direct, label = 'True direct')
    #plt.plot(taus, label = 'Taus')
    plt.legend()
    plt.show()
    
    gains = percentage_gains(model2, ignorant, true)
    plt.plot(gains)
    plt.title('Percentage gains')
    plt.show()
    
    absolute1, absolute2 = absolute_gains(model2, ignorant, true)
    plt.plot(absolute1, label = 'Model2 difference to True')
    plt.plot(absolute2, label = 'Ignorant difference to True')
    diff = absolute1-absolute2
    worst = 0
    if any(absolute1 > absolute2):
        worst = np.max(diff[absolute1>absolute2])
    plt.title('Absolute differences, max loss is {}'.format(worst))
    plt.legend()
    plt.show()
    
    plt.hist(gains[gains > 0], bins = 20)
    plt.title('Percentage gains in {} cases'.format(np.sum(gains>0)))
    plt.show()
    
    plt.hist(gains[gains < 0], bins = 20)
    plt.title('Percentage losses in {} cases'.format(np.sum(gains<0)))
    plt.show()
    
    gains_ignorant = percentage_gains_ignorant(model2, ignorant, true)
    plt.hist(gains_ignorant[gains<0], bins = 20)
    plt.title('Percentage gain of ignorant in {} cases'.format(np.sum(gains<0)))
    plt.show()
    
    if Nfake_rand == 1:
        taus_ar = np.array(taus)
        plt.hist(taus_ar[gains<0], bins = 20)
        plt.title('Taus where loss occurs')
        plt.show()
        
        plt.hist(taus)
        plt.show()
    
def Par_compare_models_plotting_tau_deterministic(t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, data): #data a tuple
    model2= []
    ignorant= []
    true = []
    for k in range(len(data)):
        model2v, ignorantv, truev, info_path = data[k]
        model2.append(model2v)
        ignorant.append(ignorantv)
        true.append(truev)
    
    
    plt.plot(model2, label='Model 2')
    plt.plot(ignorant, label = 'Ignorant')
    plt.plot(true, label = 'True')
    #plt.plot(true_direct, label = 'True direct')
    #plt.plot(taus, label = 'Taus')
    plt.legend()
    plt.show()
    
   
    gains = percentage_gains(model2, ignorant, true)
    plt.plot(gains)
    plt.title('Percentage gains')
    plt.show()
    
    absolute1, absolute2 = absolute_gains(model2, ignorant, true)
    plt.plot(absolute1, label = 'Model2 difference to True')
    plt.plot(absolute2, label = 'Ignorant difference to True')
    diff = absolute1-absolute2
    worst = 0
    if any(absolute1 > absolute2):
        worst = np.max(diff[absolute1>absolute2])
    plt.title('Absolute differences, max loss is {}'.format(worst))
    plt.legend()
    plt.show()
    
    plt.hist(gains[gains > 0], bins = 20)
    plt.title('Percentage gains in {} cases'.format(np.sum(gains>0)))
    plt.show()
    
    plt.hist(gains[gains < 0], bins = 20)
    plt.title('Percentage losses in {} cases'.format(np.sum(gains<0)))
    plt.show()
    
    gains_ignorant = percentage_gains_ignorant(model2, ignorant, true)
    plt.hist(gains_ignorant[gains<0], bins = 20)
    plt.title('Percentage gain of ignorant in {} cases'.format(np.sum(gains<0)))
    plt.show()

def pred_curves_var_prior(t_array, s, C, m, mu, X_array, T, N, which_X, tau): #prediction curves for various priors
    #only works for binary choices of X_array
    
    #generate the one and only info_path
    BM_path = BM(t_array)
    info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
    plot_info(info_path, t_array, C, m, tau)
    
    curve_list = []
    
    priorgrid = np.linspace(0, 1, Nmacroscopic)
    #approximate priors in middle of these intervals
    for i in range(len(priorgrid)-1):
        print("Dealing with prior {}".format(i))
        p = 0.5 * (priorgrid[i+1] - priorgrid[i]) #probability of X_array[0, 0]
        prior = np.array([[p, 1-p]])
        #generate prediction curves
        A = (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau)
        predictions, info_path = Par_prediction_curves_var_prior(Ngrid, A, info_path) 
        single_curve = predictions
        curve_list.append(single_curve)
    return curve_list
        
def Par_prediction_curves_var_prior(Ngrid, A, info_path): #A a tuple of the form
    #(t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau) = A
    (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau) = A
    #BM_path = BM(t_array)
    #info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
    conditioning_grid = np.linspace(1/Ngrid, 1, Ngrid)
    B = (info_path, A)
    
    #debug
    #K = Par_prediction_helper(conditioning_grid[0], B)
    #debug 
    
    list_t = [B for i in conditioning_grid]
    
    results = Parallel(n_jobs=-1)(delayed(Par_prediction_helper)(i,t) for i,t in zip(conditioning_grid,list_t))  
    return results, info_path
        
def Macro_plotting(data, beta_a, beta_b, ignorant_weight, model2_weight, true_weight):
    ignorant_dict = {}
    model2_dict = {}
    true_dict = {}
    for k in range(len(data)): #k number of curves
        ignorant = []
        model2 = []
        true = []
        curve = data[k]
        for j in range(len(curve)):
            current = curve[j]
            model2v = current[0]
            ignorantv = current[1]
            truev = current[2]
            model2.append(model2v)
            ignorant.append(ignorantv)
            true.append(truev)
        ignorant_dict["ignorant{}".format(k)] = np.array(ignorant)
        model2_dict["model2{}".format(k)] = np.array(model2)
        true_dict["true{}".format(k)] = np.array(true)
        #element k in dictionary is an array representing the k-th curve
    
    #now we average according to the beta distribution, and the type I/II/III distribution
    priorgrid = np.linspace(0, 1, Nmacroscopic)
    beta_cdf_vals = beta.cdf(priorgrid, Beta_a, Beta_b)
    beta_weights = []

    for i in range(len(priorgrid) - 1): #number of intervals that must assign beta weights to
        beta_weight = beta_cdf_vals[i+1] - beta_cdf_vals[i]
        beta_weights.append(beta_weight)
    
    #averaging curves
    average = 0
    for k in range(len(data)):
        average = average + beta_weights[k] * (Ignorant_weight * ignorant_dict["ignorant{}".format(k)]
                    + Model2_weight * model2_dict["model2{}".format(k)]
                    + True_weight * true_dict["true{}".format(k)])
        
    plt.plot(average)
    plt.show()
   
def zero_approx(funepshalfplus, funepshalfminus, funa, epsilon):
    f_dash =1/epsilon * (funepshalfplus - funepshalfminus)
    delta = (0.5-funa)/f_dash #want f to be fifty percent
    return delta #root approximation increment

def probarray(x):
    return np.array([[x, 1-x]])

def Par_fifty(pgs, A): #pgs priorgrids; A tuple of the form A = (info_path, t_array, s, C, m, mu, X_array, T, N, which_X, tau, int_samples)
    B = (pgs, A)
    
    
    list_i = range(len(pgs[0]))
    list_t = [B for i in list_i]
    
    results = Parallel(n_jobs=-1)(delayed(Par_fifty_helper)(i,t) for i,t in zip(list_i,list_t))  
    return results

def Par_fifty_helper(i, B):
    (pgs, A) = B
    (info_path, t_array, s, C, m, mu, X_array, T, N, index, which_X, tau, int_samples) = A
    pg_ig = pgs[0]
    pg_m2 = pgs[1]
    pg_t = pgs[2]
    
    prior_ig = probarray(pg_ig[i])
    prior_m2 = probarray(pg_m2[i])
    prior_t = probarray(pg_t[i])
    
    print(prior_ig)
    mod2, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, prior_m2, X_array, T, N, index, which_X, tau, int_samples)
    igno = true_model_direct(info_path, t_array, s, 0, m, mu, prior_ig, X_array, T, N, index, which_X, tau)
    true = true_model_direct(info_path, t_array, s, C, m, mu, prior_t, X_array, T, N, index, which_X, tau)
    
    return (mod2, igno, true)
    
def fifty_percent_line(info_path, t_array, s, C, m, mu, X_array, T, N, index, which_X, tau):
    conditioning_grid = np.linspace(1/Ngrid, 1, Ngrid)
    #prior probabilities that make two options equivalent along conditioning_grid
    #at time zero, this is p = 0.5 for all versions
    fifty_ignorant = [0.5]
    fifty_model2 = [0.5]
    fifty_true = [0.5]
    
    fifty_values_ignorant = []
    fifty_values_model2 = []
    fifty_values_true = []
    
    #need MC integration variables to be the same, otherwise get spurious noise in derivatives
    int_samples = []
    for k in range(MC_integration_outer_steps):
        int_samples_inner = integration_upper_bound * np.random.rand(MC_integration_inner_steps, noises)
        int_samples.append(int_samples_inner)
    
    for k in range(len(conditioning_grid)):
        print("Gridpoint {}".format(k))
        index = int(N*conditioning_grid[k])
        #previous fifty_percent prior
        prior_prev_ig =probarray(fifty_ignorant[k])
        prior_prev_m2 = probarray(fifty_model2[k])
        prior_prev_t = probarray(fifty_true[k])
        
#        #take previous prior as the starting point; construct searchgrid
#        prior_searchgrid_ignorant = np.linspace(fifty_ignorant[k]-step_root, fifty_ignorant[k]+step_root, par_root)
#        prior_searchgrid_model2 = np.linspace(fifty_model2[k]-step_root, fifty_model2[k]+step_root, par_root)
#        prior_searchgrid_true = np.linspace(fifty_true[k]-step_root, fifty_true[k]+step_root, par_root)
#        pgs = [prior_searchgrid_ignorant, prior_searchgrid_model2, prior_searchgrid_true]
        
        prior_searchgrid_ignorant = [fifty_ignorant[k]-epsilon_root/2, fifty_ignorant[k],  fifty_ignorant[k]+epsilon_root/2]
        prior_searchgrid_model2 = [fifty_model2[k]-epsilon_root/2, fifty_model2[k],  fifty_model2[k]+epsilon_root/2]
        prior_searchgrid_true = [fifty_true[k]-epsilon_root/2, fifty_true[k] , fifty_true[k]+epsilon_root/2]
        pgs = [prior_searchgrid_ignorant, prior_searchgrid_model2, prior_searchgrid_true]
#        
        #go into parallelized routines
        A = (info_path, t_array, s, C, m, mu, X_array, T, N, index, which_X, tau, int_samples)
        results = Par_fifty(pgs, A)
        
        (model2epshalfminus, ignorantepshalfminus, trueepshalfminus) = results[0]
        (model2_prev, ignorant_prev, true_prev) = results[1]
        (model2epshalfplus, ignorantepshalfplus, trueepshalfplus) = results[2]
        
        
#        model2_prev, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, prior_prev_m2, X_array, T, N, index, which_X, tau, int_samples)
#        ignorant_prev = true_model_direct(info_path, t_array, s, 0, m, mu, prior_prev_ig, X_array, T, N, index, which_X, tau)
#        true_prev = true_model_direct(info_path, t_array, s, C, m, mu, prior_prev_t, X_array, T, N, index, which_X, tau)
#        
#        model2epshalfplus, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, probarray(fifty_model2[k] + epsilon_root/2), X_array, T, N, index, which_X, tau, int_samples)
#        model2epshalfminus, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, probarray(fifty_model2[k] - epsilon_root/2), X_array, T, N, index, which_X, tau, int_samples)
#        #ignorant, _, _ = get_best_estimate_ignorant(info_path, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau)
#        ignorantepshalfplus = true_model_direct(info_path, t_array, s, 0, m, mu, probarray(fifty_ignorant[k] + epsilon_root/2), X_array, T, N, index, which_X, tau)
#        ignorantepshalfminus = true_model_direct(info_path, t_array, s, 0, m, mu, probarray(fifty_ignorant[k] - epsilon_root/2), X_array, T, N, index, which_X, tau)
#        #true, _, _ = get_true_best_estimate(info_path, t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
#        trueepshalfplus = true_model_direct(info_path, t_array, s, C, m, mu, probarray(fifty_true[k] + epsilon_root/2), X_array, T, N, index, which_X, tau)
#        trueepshalfminus = true_model_direct(info_path, t_array, s, C, m, mu, probarray(fifty_true[k] - epsilon_root/2), X_array, T, N, index, which_X, tau)
        
        delta_model2 = zero_approx(model2epshalfplus, model2epshalfminus, model2_prev, epsilon_root)
        delta_ignorant = zero_approx(ignorantepshalfplus, ignorantepshalfminus, ignorant_prev, epsilon_root)
        delta_true = zero_approx(trueepshalfplus, trueepshalfminus, true_prev, epsilon_root)
        
        ignorant_prior = fifty_ignorant[k] + delta_ignorant
        model2_prior = fifty_model2[k] + delta_model2
        true_prior = fifty_true[k] + delta_true
        
        #checking
        model2_current, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, probarray(model2_prior), X_array, T, N, index, which_X, tau, int_samples)
        ignorant_current = true_model_direct(info_path, t_array, s, 0, m, mu, probarray(ignorant_prior), X_array, T, N, index, which_X, tau)
        true_current = true_model_direct(info_path, t_array, s, C, m, mu, probarray(true_prior), X_array, T, N, index, which_X, tau)
            
        #Here we can build a high-precision root finder that searches an interval around the new suggested priors
        #to get closer to the root. The interval ideally would be a difference of old prior - new prior, both sides
        #to the old prior. 
            
        if high_precision_root_finding == 1:
            
            fine_grid_igno = np.linspace(ignorant_prior - frac*np.abs(delta_ignorant), ignorant_prior + frac*np.abs(delta_ignorant), par_root)
            fine_grid_m2 = np.linspace(model2_prior - frac*np.abs(delta_model2), model2_prior + frac*np.abs(delta_model2), par_root)
            fine_grid_true = np.linspace(true_prior - frac*np.abs(delta_true), true_prior + frac*np.abs(delta_true), par_root)
            
            pgs_fine = [fine_grid_igno, fine_grid_m2, fine_grid_true]
            
            #go into parallelized routines
            A = (info_path, t_array, s, C, m, mu, X_array, T, N, index, which_X, tau, int_samples)
            results = Par_fifty(pgs_fine, A)
            
            res_igno = []
            res_m2 = []
            res_true = []
            
            #unpack results
            for indx in range(len(results)):
                current = results[indx]
                res_m2.append(current[0])
                res_igno.append(current[1])
                res_true.append(current[2])
                
            (optindx_igno, decider_igno, weight_igno) = opt_index(res_igno)
            (optindx_m2, decider_m2, weight_m2) = opt_index(res_m2)
            (optindx_true, decider_true, weight_true) = opt_index(res_true)
#            
            if decider_igno == 0:
                ignorant_prior_tentative = fine_grid_igno[optindx_igno]
            elif decider_igno == 1:
                ignorant_prior_tentative = (1-weight_igno) * fine_grid_igno[optindx_igno] + weight_igno * fine_grid_igno[optindx_igno-1]
#            
            if decider_m2 == 0:
                model2_prior_tentative = fine_grid_m2[optindx_m2]
            elif decider_m2 == 1:
                model2_prior_tentative = (1-weight_m2) * fine_grid_m2[optindx_m2] + weight_m2 * fine_grid_m2[optindx_m2-1]
#            
            if decider_true == 0:
                true_prior_tentative = fine_grid_true[optindx_true]
            elif decider_true == 1:
                true_prior_tentative = (1-weight_true) * fine_grid_true[optindx_true] +  weight_true * fine_grid_true[optindx_true-1]
                
            #now need to check that these priors are better than the ones we had before
            #checking
            m2_result, _, _ = get_best_estimate(info_path, t_array, s, C, m, mu, probarray(model2_prior_tentative), X_array, T, N, index, which_X, tau, int_samples)
            igno_result = true_model_direct(info_path, t_array, s, 0, m, mu, probarray(ignorant_prior_tentative), X_array, T, N, index, which_X, tau)
            true_result = true_model_direct(info_path, t_array, s, C, m, mu, probarray(true_prior_tentative), X_array, T, N, index, which_X, tau)
            
            if np.abs(igno_result - 0.5) < np.abs(ignorant_current - 0.5):
                ignorant_prior = ignorant_prior_tentative
                ignorant_current = igno_result
            if np.abs(m2_result - 0.5) < np.abs(model2_current - 0.5):
                model2_prior = model2_prior_tentative
                model2_current = m2_result
            if np.abs(true_result - 0.5) < np.abs(true_current - 0.5):
                true_prior = true_prior_tentative
                true_current = true_result
#                
        fifty_ignorant.append(ignorant_prior)
        fifty_model2.append(model2_prior)
        fifty_true.append(true_prior)
        
        #these should be close to 0.5
        fifty_values_ignorant.append(ignorant_current)
        fifty_values_model2.append(model2_current)
        fifty_values_true.append(true_current)
        
    plt.plot(fifty_ignorant, label = 'ignorant')
    plt.plot(fifty_model2, label = 'model2')
    plt.plot(fifty_true, label = 'true')
    plt.title('Fifty percent division lines on space of priors')
    plt.legend()
    plt.show()
    
    plt.plot(fifty_values_ignorant, label = 'ignorant')
    plt.plot(fifty_values_model2, label = 'model2')
    plt.plot(fifty_values_true, label = 'true')
    plt.title('Evals; should be close to 0.5')
    plt.legend()
    plt.show()
    
    return (fifty_ignorant, fifty_model2, fifty_true, fifty_values_ignorant, fifty_values_model2, fifty_values_true, info_path)

def opt_index(res): #res a list of values
    ar = np.array(res)
    diff = ar - 0.5
    absdiff = np.abs(diff)
    optindx = np.argmin(absdiff)
    if optindx == 0 or optindx == len(res)-1:
        print("Optimum at end of range")
        weight = 1
        return (optindx, 0, weight) #0 for end of range
    else:
        if np.sign(diff[optindx]) != np.sign(diff[optindx + 1]):
            #weight of optindx point (of lower point)
            weight = 1 - np.abs(res[optindx]-0.5)/np.abs(res[optindx+1]-res[optindx])
            return (optindx + 1, 1, weight) #1 for not end of range; returns upper index
        elif np.sign(diff[optindx - 1]) != np.sign(diff[optindx]):
            #weight of optindx-1 point
            weight = 1 - np.abs(res[optindx-1]-0.5)/np.abs(res[optindx-1]-res[optindx])
            return (optindx, 1, weight)
        else:
            print("Something strange is happening here!")
    
def poll_plotting(fifty_ignorant, fifty_model2, fifty_true, fifty_values_ignorant, fifty_vales_model2, fifty_values_true, beta_a, beta_b): #lists
    #plotting the proportion of population corresponding to the fifty percent lines in space of priors
    ignorant_prop = beta.cdf(fifty_ignorant, beta_a, beta_b)
    model2_prop = beta.cdf(fifty_model2, beta_a, beta_b)
    true_prop = beta.cdf(fifty_true, beta_a, beta_b)
    
    overall_prop = Ignorant_weight * ignorant_prop + Model2_weight * model2_prop + True_weight * true_prop
    inverse_prop = 1-np.array(overall_prop)
    
    
    plt.plot(ignorant_prop, label = 'ignorant')
    plt.plot(model2_prop, label = 'model2')
    plt.plot(true_prop, label = 'true')
    plt.legend()
    plt.title('Proportion of population supporting candidate 1')
    plt.show()
    
    plt.plot(fifty_ignorant, label = 'ignorant')
    plt.plot(fifty_model2, label = 'model2')
    plt.plot(fifty_true, label = 'true')
    plt.title('Fifty percent division lines on space of priors')
    plt.legend()
    plt.show()
    
    print(fifty_values_ignorant)
    plt.plot(fifty_values_ignorant, label = 'ignorant')
    plt.plot(fifty_values_model2, label = 'model2')
    plt.plot(fifty_values_true, label = 'true')
    plt.title('Evals; should be close to 0.5')
    plt.legend()
    plt.show()
    
    plt.plot(overall_prop, label = 'Candidate 1')
    plt.plot(inverse_prop, label = 'Candidate 2')
    plt.title('Overall proportions taking into account type weights')
    plt.legend()
    plt.show()
    
    
        
        

"""
Main program begins here. 

There are various options for: Integration (MC or Riemann, latter for single fake
news), different distributions assumed (for multinoise only
exponential, but single noise can also be uniform and exponential can
be shifted), taugen parameter to set whether generate tau randomly
(‘random’), whether we give it deterministically (‘deterministic’), or
whether we want to plot whole prediction curves (‘predcurve’). The
first two options choose a particular ‘time_of_conditioning’ and
develop statistics relative to that; the latter draws curves for
prediction as a function of conditioning time. For MC method we can
choose MC_integration_inner_steps for number of simulations run at once
using vectorization, and MC_integration_outer_steps for the number of
loops over that.

"""
    
"""
Global variables, debugging indicator, random seed, etc
"""
integration_steps_per_unit = 100000 #Global variable used in (Riemann) integration routines
#np.random.seed(0) #may enable for testing
debugging = 0 #enable for measuring (Riemann) integration errors (doubling range, halving step)
shift_par = 0 #for shifting the exponential distribution; must set to zero for multinoise

density = 'exponential' #'exponential' or 'uniform' -- true and assumed distribution
#In principle, we could have different true and assumed distributions; we do not model this. 
#In the multinoise case, only 'exponential' is modelled, without shift -- that is, waiting times are exponential distributed
#In the single noise case we can have 'uniform' on [0, 1] or a (shifted) exponential

verbose = 0 #some comments are added 

taugen ='fifty' #'deterministic' or 'random' or 'predcurve' or 'macro' or 'fifty'
#if 'random' we generate the waiting times tau randomly as well as the information process.
#if 'deterministic' we give the waiting times explicitly below as variable tau while randomly varying information process on each iteration.
#if 'predcurve' we generate prediction curves (conditioning time is being varied for the same information process).
#if 'macro' then we get aggregation of curves based on beta distributed priors present in the population
#if 'fifty' we plot the proportion supporting candidate 1 over time, based on a beta population distribution on space of priors

location = 'single location' #'laptop' or 'cloud' or 'single location'
#'single location' if one machine is used for computations and plotting
#'cloud' if the machine does the computation and then pickles the result
#'laptop' if the machine does the unpickling and plotting


integration_upper_bound = 1  #for own integration routines with noises = 1 as well as MC bounds for uniform distribution from which we draw integration variable samples
MC_integration_outer_steps = 10 #number of MC steps we do using a for loop
MC_integration_inner_steps = 20 #number of (length of tau)-dimensional integration variables we use in MC in vectorized computation (function values computed all at once)
#Total number of MC simulations is the product MC_integration_outer_steps * MC_integration_inner_steps

integr = 'MC' #'Riemann' (only for noises = 1) or 'MC'
#If 'Riemann' must set everything consistent with noises = 1 (only one item of fake news); integrations are done using Riemann sums
#If MC then we can have multiple sources of fake news; integrations are done using MC simulation

MCgraphs = 0 #showing or not graphs of MC simulations; shows how the relevant values converge

epsilon_root = 1e-06 #for root finding of the fifty percent line


par_root = 16 #number of root finding points we compute at each step
high_precision_root_finding = 1 #if 1, then do par_root search steps parallelized; otherwise approximate derivative using only 3 searches
frac = 0.5 #Used in high precision root finding (how much of prev_prior - new prior (using approximate method) interval is searched over)

#weights of type I/II/III must add up to 1; relative weights of ignorant, model2, true in population
Ignorant_weight = 0.6
Model2_weight = 0.3
True_weight = 1 - Ignorant_weight - Model2_weight #must add up to 1; 
assert(True_weight >= 0)
"""
Parameters and initializations
"""

start = time.time()

if location == 'cloud' or location == 'single location' or location == '': #if we're on laptop then these values come from the pickled file
    s = 1 #information flow rate
    C = 8 #magnitude of fake news
    m = 8 #decay rate of fake news
    mu = 5 #rate parameter of the exponential distribution for waiting times
    prior = np.array([[0.5, 0.5]]) #prior probabilities
    X_array = np.array([[0, 1]]) #the true underlying random variable we attempt to detect
    T = 1 #final time
    N = 1024 #number of steps on the time axis when discretizing the information process
    time_of_conditioning = 0.9 #time at which we make the prediction, in the cases where this is fixed (in case of taugen being 'random' or 'deterministic', not 'predcurve')
    index = int(N * time_of_conditioning) #index on the time array with respect to which we condition
    which_X =  0 #tells us which value of X_array is the true one; that is, true value is X_array[0, which_X]
    tau = np.array([[0.1, 0.2, 0.2, 0.3]]) #array of waiting times (interarrival times) if these are fixed (in case of taugen being 'deterministic' or 'predcurve', ot 'random')
    Ngrid = 5 #number of steps discretizing time for the prediction curves (in case taugen is 'predcurve' or 'macro' or 'fifty'); that is, resolution of predcurves (Ngrid predictions are required)
    Nruns = 1000 #number of runs in case taugen is 'deterministic' or 'random'
    Nprediction_pics = 1 #number of pictures of prediction curves the algorithm produces
    Nfake_rand = 15 #when taugen is 'random', this determines how many fake news (inter)arrival times we model
    #Ideally, this number is large enough so that probability of >Nfake_rand items of fake news arriving before time T is small
    
    Nmacroscopic = 20 #for aggregation, how many grid points we approximate beta distribution with
    #(a, b) of the beta distribution in aggregation
    Beta_a = 0.5
    Beta_b = 0.5 
    
    t_array = gen_t_array(T, N) #generate t_array


if location != 'laptop': #otherwise tau will be set by pickle file  
    MC_measure = integration_upper_bound**tau.shape[1] #for MC integrations, global variable (Giving the measure of area over which integration is carried out)
    if taugen == 'deterministic' or taugen == 'predcurve' or taugen == 'macro' or taugen == 'fifty':
        noises = tau.shape[1] #Number of noises; if N>1 always assume/generate exponential dist. 
    elif taugen == 'random':
        noises = Nfake_rand
########################
    
#BM_path = BM(t_array)
#info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
#plot_info(info_path, t_array, C, m, tau)

if location == 'cloud' or location == 'single location': #beginning of computations

    """If we want prediction curves"""
    if taugen == 'predcurve':
        result = []
        for k in range(Nprediction_pics):
            A = (t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau)
            predictions, info_path = Par_prediction_curves(Ngrid, A) 
            single_result = (predictions, info_path)
            result.append(single_result)
        
    
    """When tau deterministic"""
    if taugen == 'deterministic':
        A = (t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)
        result = Par_compare_models_tau_deterministic(Nruns, A)
    
    """When tau random"""
    
    if taugen == 'random':
        A = (t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, Nfake_rand)
        result = Par_compare_models_tau_random(Nruns, A)
        
    """If we want aggregation curves"""
    if taugen == 'macro': 
        result = pred_curves_var_prior(t_array, s, C, m, mu, X_array, T, N, which_X, tau)
        
    """If we want polling curves"""
    if taugen == 'fifty':
        BM_path = BM(t_array)
        info_path = info(s, X_array[0, which_X], t_array, BM_path, tau, C, m)
        plot_info(info_path, t_array, C, m, tau)
        result = fifty_percent_line(info_path, t_array, s, C, m, mu, X_array, T, N, index, which_X, tau)
    
    """Pickle the result"""
    
    output = open('result.pkl', 'wb')
    pars = (t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, Nprediction_pics)
    pickle.dump((result, pars), output) #also pickle all parameters
    output.close()

if location == 'laptop' or location == 'single location': #plotting begins
    
    """Unpickle the result"""
    
    pkl_file = open('result.pkl', 'rb')
    (result, pars) = pickle.load(pkl_file)
    pkl_file.close()
    
    (t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, Nprediction_pics ) = pars
    
    """When tau deterministic"""
    if taugen == 'deterministic':
        Par_compare_models_plotting_tau_deterministic(t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, result)
    
    """When tau random"""
    if taugen == 'random':
        Par_compare_models_plotting_tau_random(t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, result)
    
    """When prediction curves needed"""
    if taugen == 'predcurve':
        for k in range(Nprediction_pics):
            single_result = result[k]
            Par_prediction_plotting(t_array, s, C, m, mu, prior, X_array, T, N, time_of_conditioning, which_X, tau, Ngrid, single_result)
        
    """If average of prediction curves over range of priors needed"""
    if taugen == 'macro':
        Macro_plotting(result, Beta_a, Beta_b, Ignorant_weight, Model2_weight, True_weight)
        
    """If want plots of poll curves"""
    if taugen == 'fifty':
        Beta_a = 0.5
        Beta_b= 0.5
        (fifty_ignorant, fifty_model2, fifty_true, fifty_values_ignorant, fifty_values_model2, fifty_values_true, info_path) = result
        plot_info(info_path, t_array, C, m, tau)
        poll_plotting(fifty_ignorant, fifty_model2, fifty_true, fifty_values_ignorant, fifty_values_model2, fifty_values_true, Beta_a, Beta_b)


#single_simulation(t_array, s, C, m, mu, prior, X_array, T, N, index, which_X, tau)

#compare_models_analysis_tau_deterministic(Nruns, t_array, s, m, mu, prior, X_array, T, N, index, which_X, tau, verbose)

#compare_models_analysis_tau_random(1, t_array, s, m, mu, prior, X_array, T, N, index, which_X, verbose)

#for i in range(5):
#    prediction_curves(Ngrid, t_array, s, C, m, mu, prior, X_array, T, N, which_X, tau)  

endvar = time.time()
print(endvar-start)


    
    
    



    
    
    
    
    
    
    
    





