# -*- coding: utf-8 -*-
"""
Unit tests for Swendsen-Wang Cython helper functions
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import numpy as np
import bmtools.mcmc.swendsen_wang_helpers as sw
import bmtools.mcmc.randomkit_wrapper as rk

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def test_swendsen_wang_helpers_2():
    biases = np.array([-2., -3., -5., -7.])
    bonds_state = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]], dtype=np.uint8
    )
    correct_clusters = [
        np.array([-2, -2, 1, 0]),
        np.array([3, 2, -2, -2])
    ]
    correct_biases = [
        np.array([-2. + -7., -3. + -5., -5., -7.]),
        np.array([-2., -3., -3. + -5., -2. + -7.])
    ]
    correct_flip_probs =  np.array([_sigmoid(-2. + -7.), _sigmoid(-3. + -5.),
                                    _sigmoid(-3. + -5.), _sigmoid(-2. + -7.)])
    _swendsen_wang_helpers_tester(biases, bonds_state, correct_clusters, 
                                  correct_biases, correct_flip_probs)
def test_swendsen_wang_helpers_1():
    biases = np.array([2., 3., 5., 7.])
    bonds_state = np.array([
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 0, 0]], dtype=np.uint8
    )
    correct_clusters = [
        np.array([-3, 0, -1, 0]),
        np.array([3, 3, -1, -3])
    ]
    correct_biases = [
        np.array([2. + 3. + 7., 3., 5., 7.]),
        np.array([2., 3., 5., 2. + 3. + 7.])
    ]
    correct_flip_probs =  np.array([
        _sigmoid(2. + 3. + 7.), _sigmoid(2. + 3. + 7.),
        _sigmoid(5.), _sigmoid(2. + 3. + 7.)
    ])
    _swendsen_wang_helpers_tester(biases, bonds_state, correct_clusters, 
                                  correct_biases, correct_flip_probs)
                                  
def _swendsen_wang_helpers_tester(biases, bonds_state, correct_clusters, 
                                  correct_biases, correct_flip_probs):
    rng = rk.RandomKit(1234)
    clusters = sw.construct_clusters_and_update_biases(bonds_state, biases)
    flip_probs = abs(np.array(sw.calc_flip_probs_and_sample(clusters, biases, rng)))
    assert (all(clusters == correct_clusters[0]) or 
            all(clusters == correct_clusters[1])), (
        'Clustering failed: expected one of {0} got {1}'
        .format(correct_clusters, clusters))
    j = 0 if all(clusters == correct_clusters[0]) else 1
    assert all(biases == correct_biases[j]), (
        'Bias update failed: expected {0} got {1}'
        .format(correct_biases[j], biases))
    assert all(flip_probs == correct_flip_probs), (
        'Flip probability calculation failed: expected {0} got {1}'
        .format(correct_flip_probs, flip_probs))
        
if __name__ == '__main__':
    test_swendsen_wang_helpers_1()
    test_swendsen_wang_helpers_2()
