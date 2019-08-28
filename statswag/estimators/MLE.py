from __future__ import division

import logging
import unittest
import abc
import six

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.utils import Bunch


from .base import BaseEstimator

class ErrorModel(six.with_metaclass(abc.ABCMeta)):
    """
    Abstract class describing the error model of a labeler.
    """
    def __init__(self, n_classes, theta=None):
        self.theta = theta
        self.n_classes = n_classes

    def set(self, theta, n_classes=None):
        """
        Sets the class conditional error distribution to theta after checking that theta specifies valid parameters
        for self.
        :param theta: parameter(s) of the error model
        :param n_classes: number of available classes
        :return: None
        """
        if not self.check_parameters(theta, n_classes):
            logging.debug('theta: {}'.format(theta))
            raise ValueError('Invalid parameters to {}.set()'.format(type(self).__name__))
        self.theta = theta
        if n_classes is not None:
            self.n_classes = n_classes

    def difference(self, theta):
        """
        Calculates the difference between theta and the current parameters of self. There are three cases:
        (1) single parameter: absolute value of the difference
        (2) 1-D array: Euclidean distance (L_2 norm of the vector difference)
        (3) 2-D array/Matrix: Frobenius norm of the vector/matrix difference

        :param theta: parameters of an error model
        :return: the difference between theta and the current parameters of self
        """
        return np.linalg.norm(theta - self.theta)

    def score(self, prior=None):
        """
        Returns the accuracy of a labeler using self as an error model for labeling.
        :param prior: the prior distribution of class labels
        :return: the accuracy of a labeler
        """
        if prior is not None:
            return sum([self.error_proba(ell, ell) * prior(ell) for ell in range(self.n_classes)])
        else:
            raise NotImplementedError('score() is not implemented, a prior distribution is needed')

    @abc.abstractmethod
    def error_proba(self, observed, true):
        """
        Computes the probability under this error model of observing observed, when the true label is true.

        :param observed: observed label
        :type observed: int [n_classes]
        :param true: true label
        :type true: int [n_classes]
        :return: probability
        """
        pass

    @abc.abstractmethod
    def check_parameters(self, theta, n_classes):
        pass

    @abc.abstractmethod
    def calc_update(self, labels, posterior, class_names):
        pass

    def update_(self, i, posterior, class_names):
        """
        Updates the error probability (theta) of an error model

        :param i:
        :param posterior:
        :param class_names:
        :return:
        """
        new_theta = self.calc_update(i, posterior, class_names)
        logging.debug('new theta: {}'.format(new_theta))
        diff = self.difference(new_theta)
        self.set(new_theta)
        return diff


class SymmetricErrorModel(ErrorModel):
    def __init__(self, n_classes, theta=None):
        if theta is None:
            theta = np.random.random(1)[0]
        super(SymmetricErrorModel, self).__init__(n_classes, theta)

    def check_parameters(self, theta, n_classes):
        return 0 <= theta.all() <= 1

    def error_proba(self, observed, true):

        if observed == true: # correct
            return self.theta
        elif isinstance(observed, str): # if its not correct and is a string, it must be incorrect
            return (1.0 - self.theta) / (self.n_classes - 1)
        else: # if not correct and not a string, then either a nan, or an incorrect int
            if np.isnan(observed):
                return 1.0
            else:
                return (1.0 - self.theta) / (self.n_classes - 1)


    def score(self, prior=None):
        return self.theta

    def calc_update(self, i, posterior, class_names):
        """Calculates an update for the posterior estimation of each sample

        Parameters
        ----------
        labels : array-like, shape=[n_samples,n_experts]
            The expert labels for each sample
        posterior : array-like, shape=[n_samples,n_classes]
            The posterior likelihoood of each class for each sample
        Returns
        -------
        update : array-like, shape=[n_samples,]
        """
        # Remove nan labels
        rows = self.missing_labels[i][0]
        cols = self.missing_labels[i][1]
        int_cols = [list(class_names).index(c) for c in cols]
        update = posterior[rows,int_cols].mean(axis=0)
        return update


class ErrorModelCollection(object):
    """
    A class to represent a collection of error models, i.e. experts with different labeling error distributions
    """
    def __init__(self, n_classes, n_labelers, error_models=None, labelers=None, prior=None, update_prior=True):
        self.n_classes = n_classes
        if labelers is not None and error_models is not None:
            self.labelers = labelers
            self.error_models = error_models
            self.k = len(self.labelers)
        elif error_models is not None:
            self.k = n_labelers
            self.labelers = range(self.k)
            self.error_models = [error_models(n_classes=self.n_classes) for i in self.labelers]
        else:
            self.k = n_labelers
            self.labelers = range(self.k)
            self.error_models = [SymmetricErrorModel(n_classes=self.n_classes) for i in self.labelers]
        self.prior = prior
        self.update_prior = update_prior

    def set_missing_labels(self, missing_labels):
        """Set the missing labels for reference when updating
        """
        # Keep track of missing label indices
        for em in self.error_models:
            em.missing_labels = missing_labels

    def get_model(self, index):
        return self.error_models[self.labelers[index]]

    def score_model(self, index):
        return self.get_model(index).score(self.prior)

    def calc_joint(self, labels, class_names):
        """Calculate the joint probability of each class for each sample

        :param labels:
        :return: array-like, shape=[n_samples,n_classes]
        """
        n, k = np.shape(labels)
        Q = np.zeros((n, self.n_classes))
        for i in range(n): # Iterate over each sample
            class_index = 0
            for ell in class_names: #range(self.n_classes): # Iterate over each class
                value = 1.0
                for j in range(k): # Iterate over each expert
                    # Calculate expert j's probability of putting label ell for sample i
                    true_class = ell
                    obsv_class = labels[i,j]
                    v=self.get_model(j).error_proba(obsv_class,true_class)
                    value *= v
                Q[i, class_index] = value * self.prior[class_index]
                class_index += 1
        return Q

    def calc_posteriors(self, labels, class_names):
        # return normalize(self.calc_joint(labels), norm='l1', axis=1)
        arr = self.calc_joint(labels, class_names)
        # Calculates L1 norm with missing values
        posteriors = np.array(arr).T/np.nansum(np.array(arr),axis=1).T
        return posteriors.T

    def log_likelihood(self, labels, class_names):
        return np.log(self.calc_joint(labels, class_names).sum(axis=1)).sum()

    def update_(self, labels, class_names):
        q = self.calc_posteriors(labels, class_names)
        diff = [em.update_(i, q, class_names) for i, em in enumerate(self.error_models)]
        if self.update_prior:
            new_prior = q.mean(axis=0)
            logging.debug('new prior: {}'.format(new_prior))
            prior_diff = np.linalg.norm(new_prior - self.prior)
            diff.append(prior_diff)
            self.prior = new_prior
        return np.array(diff).mean()


class LikelihoodEstimator(object):
    """
    A class implementing EM for Donmez' method of unsupervised error estimation.
    """
    def __init__(self, prior_distribution=None, n_classes=None, error_model_param=dict(),max_iter=100,
                 tol=1e-6):
        """
        Initialize an estimator that uses Donmez' maximum likelihood method to estimate its parameters. If no prior
        distribution is specified, the number of classes must be included as an argument.
        :param prior_distribution: prior class distribution or None if this is to estimated from the data
        :type prior_distribution: ndarray shape=(n_classes,) | None
        :param n_classes: number of allowable classes or None if this is to estimated from the data
        :type n_classes: int | None
        :param error_model_param: parameters to pass to the __init__ function of a ErrorModelCollection
        :type error_model_param: dict
        :param max_iter: maximum number of iterations allowed in EM
        :param tol: tolerance used to decide when to stop EM iterations
        """
        if prior_distribution is not None:
            self.prior = np.array(prior_distribution)
            self.update_prior = False
            self.n_classes = len(self.prior)
        else:
            self.prior = prior_distribution
            self.update_prior = True
            self.n_classes = n_classes

        error_model_param['update_prior'] = self.update_prior
        self.error_model_param = error_model_param
        self.max_iter = max_iter
        self.tol = tol
        self.labels = None
        self.expert_models = None
        self.true_label_distribution = None

    def initialize(self, X):
        n,k = X.shape
        self.labels = X
        self.error_model_param['prior'] = self.prior
        self.error_model_param['n_labelers'] = k
        self.error_model_param['n_classes'] = self.n_classes
        self.expert_models = ErrorModelCollection(**self.error_model_param)

    def _fit(self, X, y=None):
        """
        Using EM, fit the parameters of the error models of each expert given labels in X. The argument y is an
        optional array of labels for a single system whose accuracy to assess.

        :param X: labels
        :type X: 2-D ndarray shape=(num instances, num experts)
        :param y: labels
        :type y: 1-D ndarray shape=(num instances,)
        :return: log likelihood of the fit model over all expert labels
        """
        self.num_samples,self.num_experts = np.shape(X)
        if y is not None:
            X = np.hstack((y, X))
        if self.update_prior and self.n_classes is None: # RED.
            # Try and guess the correct number of classes from X and y
            # self.n_classes = int(X.max() - X.min()) + 1
            self.n_classes = len(np.unique(pd.DataFrame(X).dropna())) # RED.
        if self.update_prior:
            self.prior = np.random.dirichlet(np.ones(self.n_classes))
            # print('initial val')
            # print(self.prior)

        # Precalculate list of tuples of indices and columns for non-nan entries
        self.missing_labels = []
        dfX = pd.DataFrame(X)
        for i in range(self.num_experts):
            nanfree = dfX[i].dropna()
            self.missing_labels.append((nanfree.index.values,nanfree.values))

        self.initialize(X)
        # Pass the list of non-nan entries to the error model collection
        self.expert_models.set_missing_labels(self.missing_labels)

        self._EM()
        self.true_label_distribution = self.expert_models.calc_posteriors(self.labels, self.class_names)
        return self.expert_models.log_likelihood(self.labels, self.class_names)

    def _EM(self):
        for i in range(self.max_iter):
            diff = self.expert_models.update_(self.labels,self.class_names)
            if diff < self.tol:
                break
        #logging.debug('iterations: {}, absolute difference: {}, theta: {}, log-likelihood: {}'.format(
        #    i, diff, self.theta, self.log_likelihood()))

    def _score(self):
        """
        Returns the accuracy of the model under test. This is assumed to be the first model, i.e. the system that
        provided the labels in the first column of the X argument to fit (assuming no y argument was provided).

        :return: estimated accuracy of the first model
        :rtype: float
        """
        accuracies = np.zeros(self.num_experts)
        for i in range(self.num_experts):
            accuracies[i] = self.expert_models.score_model(i)
        if self.n_classes == 2:
            if np.all(accuracies < 0.5):
                for i in range(self.num_experts):
                    accuracies[i] = 1.0 - accuracies[i]
                    self.expert_models.get_model(i).theta = accuracies[i]
                temp = self.expert_models.prior[0]
                self.expert_models.prior[0] = self.expert_models.prior[1]
                self.expert_models.prior[1] = temp
        self.accuracies = accuracies

    def _estimate_labels(self, X=None):
        """Estimate the best-guess labels from the probability distribution
        """
        if X is None:
            return self.true_label_distribution.argmax(axis=1)
        else:
            return self.expert_models.calc_posteriors(X, self.class_names).argmax(axis=1)

    def fit(self,X,return_pi_T=False):
        """Public-facing fit function.

        Parameters
        ----------
        X : array-like, shape=[n_samples,n_experts]
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
        self.num_samples,self.num_experts = np.shape(X)
        # Extract number of classes and class names for translation
        self.class_names = pd.Series(X.flatten()).dropna().unique()
        self.class_names.sort()
        self.n_classes = len(self.class_names)

        self._fit(np.array(X))
        self._score()
        # Pass label array to _estimate_labels so that it uses expert accuracies to calculate
        self.y_pred = [self.class_names[i] for i in self._estimate_labels(np.array(X))]
        self.true_label_distribution = self.expert_models.calc_posteriors(np.array(X), self.class_names)
        if return_pi_T:
            return self.accuracies,self.y_pred
        else:
            return Bunch(accuracies=self.accuracies,
                         labels=np.array(self.y_pred),
                         probs=self.true_label_distribution,
                         class_names=self.class_names)

class MLEOneParameterPerLabeler(LikelihoodEstimator):
    """
    An implementation of Expectation Maximization to approximate the maximum likehood estimate
    with the following caveats:
        (1) this method jointly models the error for multiple systems
        (2) this method assumes that the probability of the expert correctly labeling an instance
            does not depend on the correct label
        (3) each system is assumed to have a different error model

    """
    def __init__(self, prior_distribution=None, n_classes=None, error_model_param=dict(), max_iter=100,
                 tol=1e-6):
        error_model_param['error_models'] = SymmetricErrorModel
        super(MLEOneParameterPerLabeler, self).__init__(prior_distribution,
                                                        n_classes,
                                                        error_model_param,
                                                        max_iter,
                                                        tol)
