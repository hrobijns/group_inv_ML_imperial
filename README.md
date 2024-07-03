# Geometric analysis with machine learning: summer project with Imperial College, London
Summer project supervised by Daniel Platt (Imperial College, London, maths department) and Daattavya Argarwal (Cambridge University, computer science department).

## Packages and requirements:
- Python
- TensorFlow

## Contents:
##### requirements.txt:
- text file containing required software to run files in this repository.

### Pre-project work:

##### SHodge_learn.py:
- defined and trained a simple classification neural network, allowing us to predict Sasakian Hodge numbers from their respective weights, essentially reproducing the result in [this paper (1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub).
- experimented with different architecture to improve accuracy.
- mainly an educational/warm-up exercise to introduce NNs in the context of algebraic topology.

### Project 1: Other ML techniques, incl. group invariant ML
Based on the same paper as the pre-project work [(1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub), we attempted to improve the results, especially of the CN invariant which was predicted little better than random guessing. This was done through trying different ML techniques, including group invariant ML.

##### CNI_learn.py:
- implemented a classification NN to attempt to learn the CN invariants.
- accuracy similarly low to the paper.

##### deep_sets_SHodge.py:
- an attempt to recreate [this paper (2)](https://arxiv.org/abs/1703.06114), and implement a NN which is independant to permutation of the input vectors.
- essentially worked by training 5 NNs in parallel on all the individual elements in the input weight vectors, summing these outputs, and then further training.
- trained on Sasakian Hodge numbers (group invariancy was achieved, but no statistically significant change in accuracy observed). 

##### deep_sets_CNI.py:
- group invariant NN turned into a classifier and applied to learning the CN invariant.
- no statistically significant change in accuracy observed.

### Project 2: Harmonic 1-forms on T<sup>3</sup>
Does there exist a metric on the 3-dimensional torus T<sup>3</sup> such that every harmonic 1-form has a vanishing zero?

##### simple_PINN.py:
- implementing a simple physics informed neural network as an eductional exercise (in progress).

## References:
(1) [Aggarwal et al., 2023, *Machine learning Sasakian and G2 topology on contact Calabi-Yau 7-manifolds*](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub) <br/>
(2) [Zaheer et al., 2017, *Deep Sets*](https://arxiv.org/abs/1703.06114)
