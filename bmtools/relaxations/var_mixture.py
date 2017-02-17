# -*- coding: utf-8 -*-
""" Boltzman machine relaxation variational mixture fitting."""

import numpy as np
import scipy.linalg as la
import bmtools.exact.variational as var


def mixture_of_variational_distributions_moments(
        relaxation, rng, n_init=100, n_step=1000, step_size=0.2,
        init_scale=0.5, tol=1e-4):
    """
    Fits a Gaussian mixture to a Boltzmann machine relaxation by forming
    a weighted mixture of Gaussian variational distributions fitted from random
    initialisations, weighted according to variational objective (and with
    simple heuristic to avoid multiple equivalent mixture components).
    """
    var_obj_list = []
    var_first_mom_list = []
    var_biases_list = []
    for j in range(n_init):
        var_biases = rng.normal(size=(relaxation.n_unit,)) * init_scale
        for i in range(n_step):
            var_obj, grads_wrt_var_biases, var_first_mom = (
                var.var_obj_and_grads_mean_field(
                    var_biases, relaxation.W, relaxation.b)
            )
            grads_wrt_var_biases = np.array(grads_wrt_var_biases)
            var_biases -= step_size * grads_wrt_var_biases
        in_list = False
        for vm in var_first_mom_list:
            diff = vm - var_first_mom
            dist = diff.dot(diff)**0.5
            if dist < tol:
                in_list = True
        if not in_list:
            var_obj_list.append(var_obj)
            var_first_mom_list.append(np.array(var_first_mom))
            var_biases_list.append(var_biases)
    var_weights = np.exp(-np.array(var_obj_list))
    var_weights /= var_weights.sum()
    var_mean = np.zeros((relaxation.n_dim_r))
    var_covar = np.zeros((relaxation.n_dim_r, relaxation.n_dim_r))
    for var_first_mom, w in zip(var_first_mom_list, var_weights):
        m = relaxation.Q.T.dot(var_first_mom)
        var_mean += w * m
        var_covar += w * np.outer(m, m)
    var_covar += np.eye(relaxation.n_dim_r) - np.outer(var_mean, var_mean)
    var_covar_chol = la.cholesky(var_covar, True)
    var_log_norm = (
        0.5 * relaxation.n_dim_r * np.log(2 * np.pi) -
        relaxation.n_unit * np.log(2) +
        0.5 * relaxation.D.diagonal().sum() +
        np.log(np.exp(-np.array(var_obj_list)).sum())
    )
    return var_mean, var_covar_chol, var_log_norm
