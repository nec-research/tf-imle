# tf-imle
Tensorflow 2 implementation and Jupyter notebooks for Implicit Maximum Likelihood Estimation

There is also a PyTorch implementation available: https://github.com/uclnlp/torch-imle

# Introduction

Implicit MLE (I-MLE) makes it possible to integrete discrete combinatorial optimization algorithms, such as Dijkstra's algorithm or integer linear program (ILP) solvers, into standard deep learning architectures. The core idea of I-MLE is that it defines an implicit maximum likelihood objective whose gradients are used to update upstream parameters of the model. Every instance of I-MLE requires two ingredients:

An ability to approximately sample from a complex and intractable distribution. For this we use Perturb-and-MAP (aka the Gumbel-max trick) and propose a novel family of noise perturbations tailored to the problem at hand.
I-MLE reduces the KL divergence between the current distribution and the empirical distribution. Since in our setting, we do not have access to the empirical distribution and, therefore, have to design surrogate empirical distributions. Here we propose two families of surrogate distributions which are widely applicable and work well in practice.


## Requirements: 
* tensorflow==2.3.0 or tensorflow-gpu==2.3.0
* numpy==1.18.5
* matplotlib==3.1.1
* scikit-learn==0.24.1
* tensorflow-probability==0.7.0


## Reference

```bibtex
@inproceedings{niepert21imle,
  author    = {Mathias Niepert and
               Pasquale Minervini and
               Luca Franceschi},
  title     = {Implicit {MLE:} Backpropagating Through Discrete Exponential Family
               Distributions},
  booktitle = {NeurIPS},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR}},
  year      = {2021}
}
```
