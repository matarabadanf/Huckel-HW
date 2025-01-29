# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:30:59 2025

@author: matar
"""
from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt


class chain:

    def __init__(
        self, n_atoms, alternated_distances=False, alternated_atoms=False
    ):
        self.n_atoms = n_atoms
        self._alternated_distances = alternated_distances
        self._alternated_atoms = alternated_distances
        self._beta_minus = -1.0
        self._alpha_prime = 0.0
        self._ring = False

    # ========================================================================
    #   Public methods
    # ========================================================================

    def plot_eigenvectors(self, n_states):
        if n_states < 5:

            for i in range(n_states):
                plt.plot(self.eigen[1][:, i], label=f"State {i}")
        else:
            plt.plot(self.eigen[1][:, n_states - 1], label=f"State {n_states}")
        plt.legend()
        plt.show()

    def plot_eigenvalues(self):
        plt.plot(np.arange(self.n_atoms), self.eigen[0])

    def set_beta_minus(self, beta_minus: float = -1):
        self._beta_minus = beta_minus
        self._clear_cached_properties()

    def set_alpha_prime(self, alpha_prime: float = 0):
        self._alpha_prime = alpha_prime
        self._clear_cached_properties()

    def close_ring(self):
        self._ring = True
        self._clear_cached_properties()

    def open_ring(self):
        self._ring = False
        self._clear_cached_properties()

    # ========================================================================
    #     Property methods
    # ========================================================================

    @cached_property
    def huckel_matrix(self):

        huckel_matrix = np.zeros([self.n_atoms, self.n_atoms])

        # Handle betas
        for i in range(self.n_atoms - 1):
            if i % 2 == 0:
                huckel_matrix[i + 1, i] = huckel_matrix[i, i + 1] = -1
            else:
                huckel_matrix[i + 1, i] = huckel_matrix[i, i + 1] = (
                    self._beta_minus
                )

        # Handle alphas
        for i in range(self.n_atoms):
            if i % 2 == 0:
                huckel_matrix[i, i] = huckel_matrix[i, i] = 0
            else:
                huckel_matrix[i, i] = huckel_matrix[i, i] = self._alpha_prime

        # Handle the ring case
        if self._ring:

            huckel_matrix[0, self.n_atoms - 1] = huckel_matrix[
                self.n_atoms - 1, 0
            ] = -1

        return huckel_matrix

    @cached_property
    def eigen(self):
        self._eigenvalues, self._eigenvectors = np.linalg.eig(
            self.huckel_matrix
        )
        idx = np.argsort(self._eigenvalues)
        # Sort eigenvalues and eigenvectors
        sorted_eigenvalues = self._eigenvalues[idx]
        sorted_eigenvectors = self._eigenvectors[:, idx]

        self._eigenvalues = sorted_eigenvalues
        self._eigenvectors = sorted_eigenvectors

        return sorted_eigenvalues, sorted_eigenvectors

    # ========================================================================
    #     Private internal methods
    # ========================================================================

    def _clear_cached_properties(self):
        cached_properties = ["eigen", "huckel_matrix"]

        for prop in cached_properties:
            if prop in self.__dict__:
                del self.__dict__[prop]


if __name__ == "__main__":
    c1 = chain(100)
    c2 = chain(100)
    c2.set_beta_minus(-0.5)
    c3 = chain(100)
    c3.set_beta_minus(-0.1)

    plt.plot(range(c1.n_atoms), c1.eigen[0])
    plt.plot(range(c2.n_atoms), c2.eigen[0])
    plt.plot(range(c3.n_atoms), c3.eigen[0])
    plt.show()

    c1.close_ring()
    c2.close_ring()
    c2.close_ring()
    plt.plot(range(c1.n_atoms), c1.eigen[0])
    plt.plot(range(c2.n_atoms), c2.eigen[0])
    plt.plot(range(c3.n_atoms), c3.eigen[0])
