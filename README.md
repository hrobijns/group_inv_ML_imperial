# Geometric analysis with machine learning: summer project with Imperial College, London
Summer project in collaboration with Meg Dearden-Hellawell, supervised by Daniel Platt (Imperial College, London, maths department) and Daattavya Argarwal (Cambridge University, computer science department).

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
Based on the same paper as the pre-project work [(1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub), we attempted to improve the results, especially of the CN invariant which was predicted little better than random guessing. This was done through trying different ML techniques, including group invariant ML - since permuting the input weights should not change the outputs, we were interested to see if enforcing the NN to be invariant to permutation of the input vectors could increase accuracy.

##### CNI_learn.py:
- implemented a classification NN to attempt to learn the CN invariants.
- accuracy similarly low to the paper.

##### deep_sets(a)_SHodge.py:
- a re-implementation of [this paper (2), Deep Sets.](https://arxiv.org/abs/1703.06114), in an attempt to create a NN which is equivariant to the permutation of the input vector.
- specifically, implementing a NN which follows from Lemma 3 (Section 2.2, page 2)
- successfully created a permutation equivariant NN.

##### deep_sets(b)_SHodge.py:
- a re-implementation of a different part of the same [Deep Sets paper (2)](https://arxiv.org/abs/1703.06114), this time Theorem 2 (Section 2.2., page 2)
- successfully created a permutation equivariant NN.

##### deep_sets(b)_CNI.py:
- turned NN from deep_sets(b)_SHodge.py into a classifier to learn the CN invariants.

### Project 2: Harmonic 1-forms on T<sup>3</sup>
Does there exist a metric on the 3-dimensional torus T<sup>3</sup> such that every harmonic 1-form has a vanishing zero?

##### simple_PINN.py:
- implementing a simple physics informed neural network, as in [this paper (3)](https://arxiv.org/abs/1711.10561), as an eductional exercise.

## References:
(1) [Aggarwal et al., 2023, *Machine learning Sasakian and G2 topology on contact Calabi-Yau 7-manifolds*](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub) <br/>
(2) [Zaheer et al., 2017, *Deep Sets*](https://arxiv.org/abs/1703.06114) <br/>
(3) [Raissi et al., 2017, *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*](https://arxiv.org/abs/1711.10561)
