from __future__ import division

import numpy as np
import pandas as pd
import math

from .base import BaseEstimator

from sklearn.utils import Bunch

class IWMV(BaseEstimator):
    """Implementation of Li & Yu '14: Error Rate Bounds and Iterative Weighted
    Majority Voting for Crowdsourcing
    """

    def __init__(self, mode='normal', n_iter=10):
        """Initialize a new Iterative Weighted Majority Voting model

        Note: Using mode 'log' is extremely risky when labels are noisy.

        Parameters
        ----------
        mode : multistep or onestep
        n_iter : number of iterations to run for
        """
        self.mode = mode
        self.n_iter = n_iter

    def fit(self, X, return_pi_T=False):
        """See Algorithm 1 in Li & Yu '14.

        Paramters
        ---------
        X : array-like, shape=(n_samples,n_experts)

        return_pi_T : boolean
            Whether or not to return (accuracies,labels) as a tuple instead
            of a Bunch object.

        Returns
        -------
            data : Bunch
                Dictionary-like object, the interesting attributes are:
                'accuracies', the estimated expert accuracies, 'labels', the
                best-guess labels, 'class_names' the name of each unique class
                this estimator observed, 'probs' the probability of each
                possible label for each sample (None if not available), and class_names
                the name (and ordering) of the classes.

                The ordering of columns in probs corresponds to that in class_names
        """
        Z = np.array(X)
        n_samples,n_experts = np.shape(Z)
        # Workaround for not getting NaNs in the list of classes
        # Since NaN == NaN evaluates to False
        classes = np.sort(pd.Series(Z.flatten()).dropna().unique())
        L = len(classes)

        # Initialize equal weights for all experts
        v = np.array([1 for i in range(n_experts)])

        # Identity matrix, response or no-response
        T = ~pd.isnull(Z)
        T = np.array(T).astype(int)
        s = 0 # Keep track of iterations
        converged = False
        # Initialize 'best-guess' with all one class
        y_prev = np.full(n_samples,classes[0])

        while (s<self.n_iter and not converged):
            # Estimate best-guess labels
            all_votes = np.array([np.sum(v*(Z==k).astype(int),axis=1) for k in classes])
            y_hat = np.array([classes[i] for i in np.argmax(all_votes,axis=0)])
            # Calculate expert accuracies (according to the updated best-guess labels)
            w_hat = np.sum((Z.T==y_hat).astype(int),axis=1)
            w_hat = w_hat / np.sum(T,axis=0)
            # Calculate new expert weights (how much their vote counts)
            if self.mode == 'log':
                MIN_INT = np.iinfo(np.int16).min
                v = np.array([MIN_INT if w_i == 0 else math.log((L-1)*w_i)/(1-w_i) for w_i in w_hat])
            else:
                # Derived in eq. 33 in Li & Yu paper
                v = L*w_hat-1

            # If the labels haven't changed since last time, it's converged!
            if (y_hat == y_prev).all():
                converged = True

            # Updated number of iterations completed
            s += 1
            y_prev = y_hat

        if return_pi_T:
            return w_hat,y_hat
        else:
            return Bunch(accuracies=w_hat,labels=y_hat,probs=None,class_names=classes)
