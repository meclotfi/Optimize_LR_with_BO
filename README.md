# Optimize_LR_with_BO

 A program that aim to find the optimal learning rate of a small ResNet for classifying images (Fashion-MNIST) 
 using Bayesian Optimization (BO) as a global optimization approach

## How to run the program ?

The program can be run using the Colab notebook in this link. In the notebook, this repos is cloned and 
the functions defined in it are used to run the optimization loop and show all the plots during 
the progress of the method (observations, the posterior mean, uncertainty estimate and the acquisition function) 

## Structure of the program

 the program contains four python files :
 - Model.py : contains the implementation of the small Resnet ( contains 2 residual network blocks)
 - dataset.py: contains the method that load the dataset (FashionMnist) into pytorch dataloaders and preproccess the images.
 - train.py: contains the tools to train the model and get the validation accuracy.
 - BO.py : contains the implementation of the optimization method BO (in BO class) as well as the probelm class responsible for calculating the objective using the first three files.    


## Exemple of the plots

In the above plot named surrogate we can find :
- The posterior mean (blue line) and  uncertainty estimate (grey area)
- The observations shown as blue dots as well as the next selected point shown with a red x

In the below plot named acquisition we can find :
- the Acquisition function defined as the expected improvement of each point (red line)

We can observe that the next selected point is the one that maximize the expected improvement 

![alt text](https://github.com/meclotfi/Optimize_LR_with_BO/blob/main/plots/Plots_iteration_2.png)
