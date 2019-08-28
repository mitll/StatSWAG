from __future__ import division

import numpy as np
import pandas as pd

from random import shuffle,seed
from collections import Counter

from .base import BaseEstimator

from statswag.metrics import nan_accuracy

from sklearn.utils import Bunch

class MajorityVote(BaseEstimator):
    def __init__(self,hold_out=True,randomSeed=None):
        self.hold_out = hold_out
        self.accuracies = None
        self.labels = None
        if randomSeed is not None:
            np.random.seed(randomSeed)

    def _get_majority_vote(self,X):
        """
        Get the majority vote for each class.

        This will return a list of the class-name.
        If you need the actual class names, use
        self.class_names.
        """
        n_samples,_ = np.shape(X)
        majority_votes = []
        for i in range(n_samples):
            # label_counts = np.bincount(X[i])
            label_counts = []
            for class_name in self.class_names:
                label_counts.append(list(X[i]).count(class_name))
            mv = np.random.choice([j for j in range(self.num_classes) \
                                    if label_counts[j]==max(label_counts)])
            majority_votes.append(self.class_names[mv])
        return majority_votes

    def _fit(self,X):
        self.class_names = pd.Series(X.flatten()).dropna().unique()

        # Not necessary but helpful for debugging
        self.class_names.sort()

        self.num_classes = len(self.class_names)
        self.num_samples,self.num_experts = X.shape
        expert_labels = np.array(X)

        # Single each expert out and compute their accuracy against the others
        accuracies = []
        for i in range(self.num_experts):
            target_column = expert_labels[:,i]
            if self.hold_out: # Leave out the target column when computing majority vote
                expert_columns = np.delete(expert_labels,i,axis=1)
                expert_votes = self._get_majority_vote(expert_columns)
            else: # Include the target column when computing majority vote
                expert_votes = self._get_majority_vote(expert_labels)
            accuracies.append(nan_accuracy(target_column,expert_votes))

        self.accuracies = accuracies
        self.labels = self._get_majority_vote(expert_labels)

    def fit(self,X,return_pi_T=False):
        """Compute the majority vote labels and accuracies

        Calculates the majority vote labels by taking the
        most common label for each sample (with ties
        broken randomly). Then compares each column to this
        set of labels to compute accuracy for each expert.

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

        (accuracies, labels) : tuple if ``return_pi_T`` is True
        """
        self._fit(np.array(X))

        if return_pi_T:
            return self.accuracies,self.labels
        else:
            return Bunch(accuracies=self.accuracies,
                         labels=self.labels,
                         probs=None,
                         class_names=self.class_names)
