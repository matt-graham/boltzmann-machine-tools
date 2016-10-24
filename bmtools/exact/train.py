# -*- coding: utf-8 -*-
"""Boltzmann machine training.

Functions for 'exact' maximum likelihood training of Boltzmann machine models
(exact in the sense that the expectations required to compute the
log-likelihood gradient with respect to the weight and bias parameters are
computed exactly i.e. by exhaustively summing across all state configurations).
Due to the exact moment calculations this is only viable for small models.
Moments can be calculated in parallel over multiple threads using OpenMP.
"""

import time
import os
import logging
import numpy as np
import moments as mom


logger = logging.getLogger(__name__)


def train_boltzmann_machine(data, n_step, step_size_func, n_hidden=None,
                            W_init=None, b_init=None, n_thread=1,
                            seed=1234, W_scale=0.1, b_scale=0.1,
                            W_penalty=None, b_penalty=None, save_dir=None):
    n_data, n_visible = data.shape
    if n_hidden is None:
        if W_init is not None:
            n_hidden = W_init.shape[0] - n_visible
        elif b_init is not None:
            n_hidden = b_init.shape[0] - n_visible
        else:
            raise ValueError('Number of hidden units not specified and no '
                             'weight matrix or bias vector to infer from.')
    logger.info(
        'Training Boltzmann machine with:\n'
        '  {0} hidden units\n'
        '  {1} visible units'.format(n_hidden, n_visible)
    )
    logger.info('------------')
    n_unit = n_hidden + n_visible
    log_liks = np.empty(n_step) * np.nan
    W, b = set_init_params(W_init, b_init, n_unit, W_scale, b_scale, seed)
    force_zero_diagonal_symmetric_weights(W)
    W_hh, W_hv, W_vh, W_vv, b_h, b_v = partition_params(W, b, n_hidden)
    dW = np.empty((n_unit, n_unit), dtype=np.double)
    db = np.empty(n_unit, dtype=np.double)
    dW_hh, dW_hv, dW_vh, dW_vv, db_h, db_v = partition_params(dW, db, n_hidden)
    # allocate arrays for inplace moment accumulation updates
    # arrays needed for each thread in parallel updates however overall
    # results accumulated in to first slice
    norm_const_h = np.empty(n_thread, dtype=np.double)
    norm_const_hv = np.empty(n_thread, dtype=np.double)
    first_mom_h = np.empty((n_thread, n_hidden), dtype=np.double)
    first_mom_hv = np.empty((n_thread, n_unit), dtype=np.double)
    second_mom_h = np.empty((n_thread, n_hidden, n_hidden), dtype=np.double)
    second_mom_hv = np.empty((n_thread, n_unit, n_unit), dtype=np.double)
    if save_dir is not None:
        save_dir = os.path.join(
            save_dir, a=time.strftime('train_bm_%Y-%m-%d_%H-%M-%S'))
        os.makedirs(save_dir)
        log_file = os.path.join(save_dir, 'log.txt')
        max_log_lik = -np.inf
    for i in range(n_step):
        logger.info('Starting iteration {0}...'.format(i))
        iter_start_time = time.time()
        # set log likelihood and update arrays to zero ready for accum
        log_lik = 0.
        dW *= 0.
        db *= 0.
        # calculate positive phase statistics
        logger.info('  Starting calculation of positive phase statistics...')
        pp_start_time = time.time()
        for v in data:
            c = b_h + W_hv.dot(v)
            mom.calculate_moments_parallel(
                W_hh, c, True, n_thread, norm_const_h, first_mom_h,
                second_mom_h
            )
            dW_vv += v[:, None] * v[None, :]
            dW_hh += second_mom_h[0]
            dW_hv += first_mom_h[0, :, None] * v[None, :]
            db_v += v
            db_h += first_mom_h[0]
            log_lik += v.dot(0.5*W_vv.dot(v)+b_v) + np.log(norm_const_h[0])
        # make sure weight update is symmetric
        dW_vh += dW_hv.T
        logger.info('  Finished in {0}s.'.format(time.time() - pp_start_time))
        # calculate negative phase statistics
        logger.info('  Starting calculation of negative phase statistics...')
        np_start_time = time.time()
        mom.calculate_moments_parallel(W, b, True, n_thread, norm_const_hv,
                                       first_mom_hv, second_mom_hv)
        logger.info('  Finished in {0}s.'.format(time.time() - np_start_time))
        # add negative phase terms to parameter updates
        dW /= n_data
        dW -= second_mom_hv[0]
        db /= n_data
        db -= first_mom_hv[0]
        # apply L2 weight / bias penalties if specified
        if W_penalty is not None:
            dW -= W_penalty * W
        if b_penalty is not None:
            db -= b_penalty * b
        # add joint log normalisation constant term to log likelihood
        log_lik -= n_data * np.log(norm_const_hv[0])
        log_liks[i] = log_lik
        # calculate step size for current iteration and apply updates
        step_size = step_size_func(i)
        W += step_size * dW
        b += step_size * db
        logger.info(
            'Iteration {0} completed in {1}s. New NLL: {2}.'
            .format(i, time.time() - iter_start_time, -log_lik)
        )
        logger.info('------------')
        # save latest parameter values to disk if save directory specified
        if save_dir:
            np.save(os.path.join(save_dir, 'W_current.npy'), W)
            np.save(os.path.join(save_dir, 'b_current.npy'), b)
            with open(log_file, 'a') as f:
                f.write('{0} iteration {1} log-likelihood {2:.2f}\n'
                        .format(time.strftime('%Y-%m-%d %H:%M:%S'), i,
                                log_lik))
            if log_lik > max_log_lik:
                max_log_lik = log_lik
                np.save(os.path.join(save_dir, 'W_best.npy'), W)
                np.save(os.path.join(save_dir, 'b_best.npy'), b)
    return W, b, log_liks


def log_lik_grad(weights, biases, data, n_thread):
    n_data, n_visible = data.shape
    n_hidden = weights.shape[0] - n_visible
    n_unit = n_hidden + n_visible
    W_hh, W_hv, W_vh, W_vv, b_h, b_v = partition_params(weights, biases,
                                                        n_hidden)
    dW = np.zeros((n_unit, n_unit), dtype=np.double)
    db = np.zeros(n_unit, dtype=np.double)
    dW_hh, dW_hv, dW_vh, dW_vv, db_h, db_v = partition_params(dW, db,
                                                              n_hidden)
    norm_const_h = np.empty(n_thread, dtype=np.double)
    norm_const_hv = np.empty(n_thread, dtype=np.double)
    first_mom_h = np.empty((n_thread, n_hidden), dtype=np.double)
    first_mom_hv = np.empty((n_thread, n_unit), dtype=np.double)
    second_mom_h = np.empty((n_thread, n_hidden, n_hidden), dtype=np.double)
    second_mom_hv = np.empty((n_thread, n_unit, n_unit), dtype=np.double)
    # positive phase statistics
    for v in data:
        c = b_h + W_hv.dot(v)
        mom.calculate_moments_parallel(W_hh, c, True, n_thread,
                                       norm_const_h, first_mom_h,
                                       second_mom_h)
        dW_vv += v[:, None] * v[None, :]
        dW_hh += second_mom_h[0]
        dW_hv += first_mom_h[0, :, None] * v[None, :]
        db_v += v
        db_h += first_mom_h[0]
    # update upper triangle using symmetricity
    dW_vh += dW_hv.T
    # negative phase statistics
    mom.calculate_moments_parallel(weights, biases, True, n_thread,
                                   norm_const_hv, first_mom_hv,
                                   second_mom_hv)
    dW -= n_data * second_mom_hv[0]
    db -= n_data * first_mom_hv[0]
    return dW, db


def log_lik(weights, biases, data, n_thread):
    return np.array(mom.log_likelihood(data, weights, biases,
                                       n_thread)).sum()


def check_grad(weights, biases, data, n_thread, h=1e-5):
    n_unit = weights.shape[0]
    dW, db = log_lik_grad(weights, biases, data, n_thread)
    dW_fd = np.empty((n_unit, n_unit)) * np.nan
    db_fd = np.empty(n_unit) * np.nan
    for i in range(n_unit):
        b_f = biases.copy()
        b_b = biases.copy()
        b_f[i] += h
        b_b[i] -= h
        ll_f = log_lik(weights, b_f, data, n_thread)
        ll_b = log_lik(weights, b_b, data, n_thread)
        db_fd[i] = (ll_f - ll_b) / (2*h)
        for j in range(i+1):
            W_f = weights.copy()
            W_b = weights.copy()
            W_f[i, j] += h
            W_f[j, i] += h
            W_b[i, j] -= h
            W_b[j, i] -= h
            ll_f = log_lik(W_f, biases, data, n_thread)
            ll_b = log_lik(W_b, biases, data, n_thread)
            dW_fd[i, j] = (ll_f - ll_b) / (2*h)
            dW_fd[j, i] = dW_fd[i, j]
    print('Calculated weights gradient dL/dW:\n{0}'.format(dW))
    print('FD approximated weights gradient dL/dW:\n{0}'.format(dW_fd))
    print('Diff between calculated and approx:\n{0}'.format(dW - dW_fd))
    print('-------------')
    print('Calculated biases gradient dL/db:\n{0}'.format(db))
    print('FD approximated biases gradient dL/db:\n{0}'.format(db_fd))
    print('Diff between calculated and approx:\n{0}'.format(db - db_fd))


def set_init_params(W_init, b_init, n_unit, W_scale, b_scale, seed):
    prng = np.random.RandomState(seed)
    # sample random Gaussian initial biases if no initialisation specified
    if b_init is None:
        b = prng.normal(size=n_unit) * b_scale / n_unit**0.5
    else:
        b = b_init
    # sample random Gaussian initial weights if no initialisation specified
    if W_init is None:
        W = prng.normal(size=(n_unit, n_unit)) * W_scale / n_unit**0.5
    else:
        W = W_init
    return W, b


def force_zero_diagonal_symmetric_weights(W):
    # ensure weight matrix symmetric and zero diagonal
    W -= np.diag(W.diagonal())
    W += W.T
    W *= 0.5


def partition_params(W, b, n_hidden):
    # partition hidden and visible parameters with views for easier updates
    W_hh = W[:n_hidden, :n_hidden]
    W_hv = W[:n_hidden, n_hidden:]
    W_vh = W[n_hidden:, :n_hidden]
    W_vv = W[n_hidden:, n_hidden:]
    b_h = b[:n_hidden]
    b_v = b[n_hidden:]
    return W_hh, W_hv, W_vh, W_vv, b_h, b_v
