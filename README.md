# Geometric analysis with machine learning: summer project with Imperial College, London
Summer project in collaboration with Meg Dearden-Hellawell, supervised by Daniel Platt (Imperial College, London, maths department) and Daattavya Argarwal (Cambridge University, computer science department).

## Packages and requirements:
- Python
- TensorFlow

## Contents:
##### requirements.txt:
- text file containing required software to run files in this repository.

### Pre-project work:

##### vanilla_SHodge.py:
- defined and trained a simple classification neural network, allowing us to predict Sasakian Hodge numbers from their respective weights, essentially reproducing the result in [this paper (1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub).
- experimented with different architecture to improve accuracy.
- mainly an educational/warm-up exercise to introduce NNs in the context of algebraic topology.

### Project 1: Other ML techniques, incl. group invariant ML
Based on the same paper as the pre-project work [(1)](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub), we attempted to improve the results, especially of the CN invariant which was predicted little better than random guessing. This was done through trying different ML techniques, including group invariant ML - since permuting the input weights should not change the outputs, we were interested to see if enforcing the NN to be invariant to permutation of the input vectors could increase accuracy.

##### vanilla_CNI.py:
- implemented a classification NN to attempt to learn the CN invariants.
- accuracy similarly low to the paper.

Four methods were chosen to enforce group invariancy, with files labelled group_invariant_X.py, where the letter in place of X represents the method (see below). The models here were all trained to learn sasakian hodge numbers, but can be and were easily adapted into classifiers to learn the CN invariant instead (see vanilla_SHodge.py versus vanilla_CNI.py).

##### Deep Sets:
The first two methods, were a re-implementation of [this paper (2), Deep Sets.](https://arxiv.org/abs/1703.06114)
- (a): the main Deep Sets method, Lemma 3 (Section 2.2, page 2)
- (b): another method from the paper, Theorem 2 (Section 2.2, page 2) 

##### Definition of group equivariance:
The final two methods derive from the definition of group invariance.
- (c): split the input vector into its 120 permutations, train a NN on these in parallel, then sum together.
- (d): train a NN, then average its predicition over the 120 permutations of each test input.

We also considered [this paper on fundamental domain projections (3)](https://openreview.net/pdf?id=RLkbkAgNA58), however it proved futile for our problem since our data was already in the fundamental domain. 

Overall, as perhaps expected, accuracy was not seen to improve by incorporating the group invariant methods (IN PROGRESS).

More information can be found in all of the methods and their implementation in the project write-up. 

### Project 2: Harmonic 1-forms on T<sup>3</sup>
Does there exist a metric on the 3-dimensional torus T<sup>3</sup> such that every harmonic 1-form has a vanishing zero?

##### simple_PINN.py:
- implementing a simple physics informed neural network, as in [this paper (4)](https://arxiv.org/abs/1711.10561), as an eductional exercise.

##### periodic_PINN.py:
- a small adaption of simple_PINN where we enforced periodicity into the solution by mapping the input layers to trig functions.

## References:
(1) [Aggarwal et al., 2023, *Machine learning Sasakian and G2 topology on contact Calabi-Yau 7-manifolds*](https://www.sciencedirect.com/science/article/pii/S0370269324000753?via%3Dihub) <br/>
(2) [Zaheer et al., 2017, *Deep Sets*](https://arxiv.org/abs/1703.06114) <br/>
(3) [Platt et al., 2022](https://openreview.net/pdf?id=RLkbkAgNA58) <br/>
(4) [Raissi et al., 2017, *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*](https://arxiv.org/abs/1711.10561)
