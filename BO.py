from numpy import arange,vstack,argmax,asarray
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings,simplefilter
import matplotlib.pyplot as plt
import numpy as np
from train import get_val_acc
from dataset import get_data_loaders
 
class Problem():
    def __init__(self):

        # load the fashion mnist dataset into the pytorch data loaders
        self.dataloaders,self.dataset_sizes=get_data_loaders()

    def evaluate(self,lr):
        acc=get_val_acc(lr,self.dataloaders,self.dataset_sizes)
        return acc

class BO():
    def __init__(self,n_iters=10):
        #
        self.problem=Problem()
        #initialize the surrogate
        self.surrogate = GaussianProcessRegressor()
        self.n_iter=n_iters

    # probability of improvement acquisition function
    def EI(X, Xsamples, model,xi=0.01):
        mu, sigma = model.predict(X, return_std=True)
        mu_sample = model.predict(Xsamples)

        sigma = sigma.reshape(-1, 1)
        
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
    
    def opt_acq(self,X, y, model,axis):
        # random search, generate random samples
        Xsamples = random(1000)
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        # calculate the acquisition function for each sample
        eis = self.EI(Xsamples, X, model)
        # locate the index of the largest scores
        self.plt_acq(axis,Xsamples,eis)
        ix = argmax(eis)
        return Xsamples[ix, 0]
    
    def plot_surrogate(self,axis):
        axis.set_xlabel('Learning rate')
        axis.set_ylabel('Accuracy')
        axis.set_title('Surrogate Gaussian Model')
        
        # scatter plot of inputs and real objective function
        axis.scatter(self.X,self.Y)
        
        # line plot of the posterior mean
        Xsamples = asarray(arange(0, 1, 0.001))
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        ysamples, y_std = self.surrogate.predict(Xsamples,return_std=True)                                                                                                                                                                                                                                                           
        axis.plot(Xsamples, ysamples)

        

        #Plot the standard deviation
        Xsamples = Xsamples.reshape(-1)
        ysamples = ysamples.reshape(-1)
        upper_bound=(ysamples - y_std).reshape(-1)
        lower_bound=(ysamples + y_std).reshape(-1)
        axis.fill_between(Xsamples,upper_bound,lower_bound,color='gray', alpha=0.2)

        #Plot the last selected point 
        axis.scatter(X[-1:],y[-1:],color='red',marker="x")
    
    def plot_acq(self,axis,Xsamples,ei):
        # plot the acquisition 
        arg=np.argsort(Xsamples.reshape(-1))
        axis.set_xlabel('Learning rate')
        axis.set_ylabel('Expected improvement')
        axis.set_title('Acquisition function')
        axis.plot(Xsamples[arg].reshape(-1),ei[arg].reshape(-1))

    def Run(self):
       X = random(1).reshape(1, 1)
       Y = asarray([self.problem.evaluate(X[0][0])]).reshape(len(X), 1)
       self.X = X
       self.Y = Y
       fig, axs = plt.subplots(2)
       fig.set_size_inches(12,10)
       plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
        

       print("Begin the Run")
       for i in range(self.n_iter):
            fig.suptitle('Plots of iteration '+str(i))
            # fitting the suurogate
            self.surrogate.fit(self.X, self.Y)

            # optimizing the acquisition function and select the next point
            x_next = self.opt_acq(self.X,self.Y,self.surrogate,axs[1])
            print(x_next)
            x_next=x_next[0]
            print(x_next)
            # evaluate the next point 
            y_next = self.problem.evaluate(x_next)
                
            # add the point to the dataset
            self.X = vstack((self.X, [[x_next]]))
            self.Y = vstack((self.Y, [[y_next]]))

            #Plot the all observations, the posterior mean, uncertainty estimate and the acquisition function after each iteration
            self.plot_surrogate(axs[0])

                
            # return new data iteration by iteration
            yield x_next, y_next
