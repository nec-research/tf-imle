# tf-imle
Tensorflow 2 and PyTorch implementation and Jupyter notebooks for Implicit Maximum Likelihood Estimation (I-MLE) proposed in the NeurIPS 2021 paper [Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions.](https://arxiv.org/abs/2106.01798)


There is also an alternative PyTorch implementation available: https://github.com/uclnlp/torch-imle

## Introduction

Implicit MLE (I-MLE) makes it possible to include discrete combinatorial optimization algorithms, such as Dijkstra's algorithm or integer linear programming (ILP) solvers, as well as complex discrete probability distributions in standard deep learning architectures. The figure below illustrates the setting I-MLE was developed for. <img src="https://render.githubusercontent.com/render/math?math=h_{\mathbf{v}}"> is a standard neural network, mapping some input <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}"> to the input parameters <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\theta}"> of a discrete combinatorial optimization algorithm or a discrete probability distribution, depicted as the black box. In the forward pass, the discrete component is executed and its *discrete* output <img src="https://render.githubusercontent.com/render/math?math=\mathbf{z}"> fed into a downstream neural network <img src="https://render.githubusercontent.com/render/math?math=f_{\mathbf{u}}">. Now, with I-MLE it is possible to estimate gradients of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\theta}"> with respect to a loss function, which are used during backpropagation to update the parameters <img src="https://render.githubusercontent.com/render/math?math=\mathbf{v}"> of the upstream neural network.

![Illustration of the problem addressed by I-MLE](https://github.com/nec-research/tf-imle/blob/main/images/i-mle-figure1.PNG)

The core idea of I-MLE is that it defines an implicit maximum likelihood objective whose gradients are used to update upstream parameters of the model. Every instance of I-MLE requires two ingredients:
1. A method to approximately sample from a complex and possibly intractable distribution. For this we use Perturb-and-MAP (aka the Gumbel-max trick) and propose a novel family of noise perturbations tailored to the problem at hand.
2. A method to compute a surrogate empirical distribution: Vanilla MLE reduces the KL divergence between the current distribution and the empirical distribution. Since in our setting, we do not have access to such an empirical distribution, we have to design surrogate empirical distributions which we term *target distributions*. Here we propose two families of target distributions which are widely applicable and work well in practice.


## Requirements: 

### TensorFlow 2 implementation:
* tensorflow==2.3.0 or tensorflow-gpu==2.3.0
* numpy==1.18.5
* matplotlib==3.1.1
* scikit-learn==0.24.1
* tensorflow-probability==0.7.0

### PyTorch implementation:


## Example: I-MLE as a Layer 

The following is an instance of I-MLE implemented as a layer. This is a class where the optimization problem is computing the k-subset configuration, the target distribution is based on perturbation-based implicit differentiation, and the perturb-and-MAP noise perturbations are drawn from the sum-of-gamma distribution.

```python
class IMLESubsetkLayer(tf.keras.layers.Layer):
    
    def __init__(self, k, _tau=10.0, _lambda=10.0):
        super(IMLESubsetkLayer, self).__init__()
        # average number of 1s in a solution to the optimization problem
        self.k = k
        # the temperature at which we want to sample
        self._tau = _tau
        # the perturbation strength (here we use a target distribution based on perturbation-based implicit differentiation
        self._lambda = _lambda  
        # the number of samples we want to take
        self.samples = None 
        
    @tf.function
    def sample_sum_of_gamma(self, shape):
        
        s = tf.map_fn(fn=lambda t: tf.random.gamma(shape, 1.0/self.k, self.k/t), 
                  elems=tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))   
        # now add the samples
        s = tf.reduce_sum(s, 0)
        # the log(m) term
        s = s - tf.math.log(10.0)
        # divide by k --> each s[c] has k samples whose sum is distributed as Gumbel(0, 1)
        s = self._tau * (s / self.k)

        return s
    
    @tf.function
    def sample_discrete_forward(self, logits): 
        self.samples = self.sample_sum_of_gamma(tf.shape(logits))
        gamma_perturbed_logits = logits + self.samples
        # gamma_perturbed_logits is the input to the combinatorial opt algorithm
        # the next two lines can be replaced by a custom black-box algorithm call
        threshold = tf.expand_dims(tf.nn.top_k(gamma_perturbed_logits, self.k, sorted=True)[0][:,-1], -1)
        y = tf.cast(tf.greater_equal(gamma_perturbed_logits, threshold), tf.float32)
        
        return y
    
    @tf.function
    def sample_discrete_backward(self, logits):     
        gamma_perturbed_logits = logits + self.samples
        # gamma_perturbed_logits is the input to the combinatorial opt algorithm
        # the next two lines can be replaced by a custom black-box algorithm call
        threshold = tf.expand_dims(tf.nn.top_k(gamma_perturbed_logits, self.k, sorted=True)[0][:,-1], -1)
        y = tf.cast(tf.greater_equal(gamma_perturbed_logits, threshold), tf.float32)
        return y
    
    @tf.custom_gradient
    def subset_k(self, logits, k):

        # sample discretely with perturb and map
        z_train = self.sample_discrete_forward(logits)
        # compute the top-k discrete values
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted=True)[0][:,-1], -1)
        z_test = tf.cast(tf.greater_equal(logits, threshold), tf.float32)
        # at training time we sample, at test time we take the argmax
        z_output = K.in_train_phase(z_train, z_test)
        
        def custom_grad(dy):

            # we perturb (implicit diff) and then resuse sample for perturb and MAP
            map_dy = self.sample_discrete_backward(logits - (self._lambda*dy))
            # we now compute the gradients as the difference (I-MLE gradients)
            grad = tf.math.subtract(z_train, map_dy)
            # return the gradient            
            return grad, k

        return z_output, custom_grad
  ```


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
