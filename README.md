# tf-imle
Tensorflow 2 implementation and Jupyter notebooks for Implicit Maximum Likelihood Estimation (I-MLE). The NeurIPS 2021 paper is available here: https://arxiv.org/abs/2106.01798


There is also a PyTorch implementation available: https://github.com/uclnlp/torch-imle

## Introduction

Implicit MLE (I-MLE) makes it possible to integrete discrete combinatorial optimization algorithms, such as Dijkstra's algorithm or integer linear program (ILP) solvers, as well as complex discrete distributions into standard deep learning architectures. The figure below illustrates the setting I-MLE was developed for. <img src="https://render.githubusercontent.com/render/math?math=h_{\mathbf{v}}"> is a standard neural network, mapping some input <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}"> to the parameters <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\theta}"> of the discrete distribution or combinatorial optimization problem. In the forward pass, the discrete component is executed and its *discrete* output <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}"> fed into a downstream neural network <img src="https://render.githubusercontent.com/render/math?math=f_{\mathbf{u}}">. Now, with I-MLE it is possible to estimate gradients with respect to <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\theta}"> which are used during backpropagation to update the parameters <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> of the upstream neural network.

![Illustration of the problem addressed by I-MLE](https://github.com/nec-research/tf-imle/blob/main/images/i-mle-figure1.PNG)

The core idea of I-MLE is that it defines an implicit maximum likelihood objective whose gradients are used to update upstream parameters of the model. Every instance of I-MLE requires two ingredients:
1. A method to approximately sample from a complex and intractable distribution. For this we use Perturb-and-MAP (aka the Gumbel-max trick) and propose a novel family of noise perturbations tailored to the problem at hand.
2. A method to compute a surrogate empirical distribution: Vanilla MLE reduces the KL divergence between the current distribution and the empirical distribution. Since in our setting, we do not have access to an empirical distribution, we have to design surrogate empirical distributions. Here we propose two families of surrogate distributions which are widely applicable and work well in practice.


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
