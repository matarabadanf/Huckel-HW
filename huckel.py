#!/usr/bin/env python3
from functools import cached_property
from contextlib import redirect_stdout
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)

class Chain:

    def __init__(self, n_atoms):
        self.n_atoms = n_atoms
        self._beta_minus = -1.0
        self._alpha_prime = 0.0
        self._ring = False

    # ========================================================================
    #   Public methods
    # ========================================================================

    def plot_eigenvectors(self, n_states) -> None:
        if n_states < 5:

            for i in range(n_states):
                plt.plot(self.eigen[1][:, i], label=f"State {i}")
        else:
            plt.plot(self.eigen[1][:, n_states - 1], label=f"State {n_states}")
        plt.legend()
        plt.show()

    def plot_eigenvalues(self) -> None:
        plt.plot(np.arange(self.n_atoms), self.eigen[0])

    def set_beta_minus(self, beta_minus: float = -1) -> None:
        self._beta_minus = beta_minus
        self._clear_cached_properties()

    def set_alpha_prime(self, alpha_prime: float = 0) -> None:
        self._alpha_prime = alpha_prime
        self._clear_cached_properties()

    def close_ring(self) -> None:
        self._ring = True
        self._clear_cached_properties()

    def open_ring(self) -> None:
        self._ring = False
        self._clear_cached_properties()

    # ========================================================================
    #     Property methods
    # ========================================================================

    @cached_property
    def huckel_matrix(self) -> np.ndarray:

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
    def eigen(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Calculate eigenvalues and eigenvectors of the Huckel hamiltonian.

        '''
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

    def _clear_cached_properties(self) -> None:
        '''Clear cached class properties.'''
        cached_properties = ["eigen", "huckel_matrix"]

        for prop in cached_properties:
            if prop in self.__dict__:
                del self.__dict__[prop]

def load_input(input_filename) -> Optional[List[str]]:
    try:
        with open(input_filename, 'r') as inp_file:
            content = inp_file.readlines()
            print(f'Loaded input file "{input_filename}"')
            return content
    except FileNotFoundError:
        print(f'Error: Input file "{input_filename}" was not found.')
    
def run_calculation(input_filename:str, output_filename:str='') -> None:

    content = load_input(input_filename)

    if content is None:
        return

    keyval_pairs = [ 
        tuple(line.replace('=', ' ').strip().split()[:2])
        for line in content 
        if len(line.replace('=', ' ').strip().split()) >= 2
    ]

    cont = dict(keyval_pairs)

    if 'n_atoms' not in cont.keys():
        raise ValueError('Definition of number of atoms is mandatory')

    print(f'\nParameters read from "{input_filename}":')
    for key in cont.keys():
        print(f'    {key} = {cont[key]}')

    c = Chain(n_atoms=int(cont['n_atoms']))

    print('\nChain object was generated.')

    if 'ring' in cont.keys():
        if cont['ring'].capitalize() == 'True':
            c.close_ring()
            print('\nChain was closed to form a ring.')

    if 'alternate_alpha' in cont.keys():
        alt_alpha = float(cont['alternate_alpha'])
        c.set_alpha_prime(alt_alpha)
        print(f'Different atom types found in the system: Alternating alpha is {alt_alpha:5.2f}')

    if 'alternate_beta' in cont.keys():
        alt_beta = float(cont['alternate_beta'])
        c.set_beta_minus(alt_beta)
        print(f'Different bond lengths found in the system: B minus is {alt_beta:5.2f}')

    print('\nThe Huckel matrix of this system is:\n')

    for row in c.huckel_matrix:
        print(' '.join(f'{val:>5.2f}' for val in row))

    eigenval, eigenvec = c.eigen

    print('\nThe eigenvalues of the Huckel matrix are:\n')

    for eigen in eigenval:
        print(f'{eigen:8.5f}')

    print('\nThe eigenvectos (in columns) of the Huckel matrix are:\n')
    for vec in eigenvec:
        print(' '.join(f'{val:>8.5f}' for val in vec))
    
    if output_filename == '':
        output_filename = input_filename

    with open(output_filename + '_eigenvalues', 'w') as f:
        for eigen in eigenval:
            f.write(f'{eigen:8.5f}')
    
    with open(output_filename + '_eigenvectors', 'w') as f:
        for vec in eigenvec:
            f.write(' '.join(f'{val:>8.5f}' for val in vec))

def print_help() -> None:
    print('The usage of this script is the following:')
    print('\n    python3 huckel.py input_filename [output_filename]')
    print('\nThe input file should contain the following syntax:')
    print('\n    n_atoms = 100')
    print('Optional values:')
    print('    ring = False  ')
    print('    alternate_alpha = None')
    print('    alternate_beta = None\n')

def main(input_file:str, output_file: str = '') -> None:
    run_calculation(input_file, output_file)

if __name__ == "__main__":

    # Check that an input file is provided
    if len(sys.argv) <= 1:
        print_help()
        exit()

    inp_file = sys.argv[1]
    print(len(sys.argv))  
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
        with open(output_file, 'w') as f:
            with redirect_stdout(f):
                run_calculation(inp_file, output_file)
    else:
        run_calculation(inp_file)
    
