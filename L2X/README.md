
# Learning to Explain (L2X) experiments

## Data set
Uncompress the files at: http://people.csail.mit.edu/taolei/beer/ into the *data* subfolder.


In the Jupyter notebook, you can use the code block below to switch between 
the standard L2X, SoftSub, and I-MLE

```python
###############################################################################
#### here we switch between the different methods #############################
# the standard L2X approach
#tau = 0.1
#T = Sample_Concrete(tau, select_k)(logits_T)

# the I-MLE approach (ours)
# tau = temperature
# lambda = implicit differentiation perturbation magnitude:  q(z; theta') with theta' = theta - lambda dL/dz
T = IMLESubsetkLayer(select_k, _tau=10.0, _lambda=1000.0)(logits_T)
    
# the SoftSub relaxation
#tau = 0.5
#T = SampleSubset(tau, select_k)(logits_T)
###############################################################################
###############################################################################     
```
