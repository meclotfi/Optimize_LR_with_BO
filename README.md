# Optimize_LR_with_BO

 A program that aim to find the optimal learning rate of a small ResNet for classifying images (Fashion-MNIST) 
 using Bayesian Optimization (BO) as a global optimization approach

## How to run the program ?

The program can be run using the Colab notebook in this link. In the notebook, this repos is cloned and 
the functions defined in it are used to run the optimization loop and show all the plots during 
the progress of the method (observations, the posterior mean, uncertainty estimate and the acquisition function) 

## Structure of the program



## Exemple of the plots

In the above plot named surrogate we can find :
- The posterior mean (blue line) and  uncertainty estimate (grey area)
- The observations shown as blue dots as well as the next selected point shown with a red x

In the below plot named acquisition we can find :
- the Acquisition function defined as the expected improvement of each point (red line)

We can observe that the next selected point is the one that maximize the expected improvement 

![alt text](https://github.com/meclotfi/Optimize_LR_with_BO/blob/main/plots/Plots_iteration_2.png)
