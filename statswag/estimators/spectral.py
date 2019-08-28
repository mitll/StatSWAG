#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import logging
import unittest

import sys
import numpy as np

from .base import BaseEstimator

from sklearn.utils import Bunch

class Spectral(BaseEstimator):

        def __init__(self, signassumption = 'half'):
            """
            :param signassumption: either half or first, half assumes at least half the labelers are better than random
            # and first assumes that the first labeler is better than random
            """

            self.sign = signassumption
            self.accuracy = None
            self.class_prior = None

        def fit(self, X, return_pi_T=False):
            """
            Fit the Jaffe estimator to a set of expert labels
            See "Estimating the accuracies of multiple classifiers without labeled data" Jaffe et al. 2015

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
            X_nominal = np.array(X)
            self.class_names = np.unique(X_nominal)
            self.num_classes = len(self.class_names)
            self.class_prior = np.zeros(self.num_classes)

            numCol = np.size(X_nominal,1)
            accuracy = 0.0
            meanVector_list = []
            sigma1_list = []
            u1_list = []
            b_list = np.zeros(self.num_classes)
            index = 0
            for c in self.class_names:

                tempX = np.asarray(X_nominal.copy(),dtype=object)

                # Maps class c to 1 and all other classes to -1
                tempX = np.asarray(self._transformLabels(tempX,c),dtype=float)

                # Calculate the mean vector and the covariance matrix of the transformed label matrix
                meanVector = np.mean(tempX, axis=0)
                meanVector_list.append(meanVector)
                covMatrix = np.cov(tempX, rowvar=False)

                # Compute the rank-one matrix R that has off-diagonal entries equal to the covariance matrix
                R, flag = self._calculateR(covMatrix)
                if flag == True:
                    print('Estimator failed.  Rank-one matrix could not be computed.')
                    return Bunch(accuracies=None,
                                 labels=None,
                                 probs=None,
                                 class_names=None)

                # Compute SVD of R to extract a rank one approximation
                U,s,V = np.linalg.svd(R,full_matrices=False)
                sigma1 = s[0]
                u1 = U[:,0]
                sigma1_list.append(sigma1)

                if self.sign == 'half':
                    # Choose the sign to match the assumption that at least half the experts/classifiers
                    # have balanced accuracy greater than 1/2
                    # Balanced accuracy is greater than 1/2 when the entry of the eigenvector is positive
                    numPos = 0
                    for i in range(numCol):
                        if (u1[i] >= 0):
                            numPos = numPos + 1
                    if numPos < numCol / 2:
                        u1 = -u1
                elif self.sign == 'first':
                    # Choose the sign to match the assumption that the classifier, first column, has
                    # balanced accuracy greater than 1/2
                    # Balanced accuracy is greater than 1/2 when the entry of the eigenvector is positive
                    if u1[0] <= 0:
                        u1 = -u1

                u1_list.append(u1)

                # Estimate b, the class imbalance parameter
                b = self._tensorEstimation(tempX,np.sqrt(sigma1)*u1)

                b_list[index] = b
                index += 1

            # Fix the b values, because underlying class prior parameters do not necessarily sum to 1
            self.class_prior = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                self.class_prior[c] = (1.0+b_list[c])/2.0
            self.class_prior = self.class_prior / np.sum(self.class_prior)
            for c in range(self.num_classes):
                b_list[c] = 2.0*self.class_prior[c] - 1.0

            class_accuracies = []
            for c in range(self.num_classes):
                # Compute the sensitivity and specificity of each expert/classifier
                r = np.sqrt(sigma1_list[c]) * u1_list[c]
                # class accuracy is not guaranteed to be between 0 and 1, so we need to clip
                class_accuracy = np.clip((1.0 / 2.0) * (1.0 + meanVector_list[c] + r *
                                                       np.sqrt((1.0 - b_list[c]) / (1.0 + b_list[c]))),0.0,1.0)
                class_accuracies.append(class_accuracy)

            # Update accuracy, using the class_accuracy of class c and the class c prior
            for c in range(self.num_classes):
                accuracy += (1.0+b_list[c])/2.0 * np.asarray(class_accuracies[c])
            self.accuracy = accuracy

            # Get a best-guess label via a weighted majority vote
            all_votes = np.array([np.sum(self.accuracy * (X_nominal == k).astype(int), axis=1)
                                  for k in self.class_names])
            self.labels = np.array([self.class_names[i] for i in np.argmax(all_votes, axis=0)])

            if return_pi_T:
                return accuracy, self.labels
            else:
                return Bunch(accuracies=self.accuracy,
                             labels=self.labels,
                             probs=None,
                             class_names=self.class_names)

        def _transformLabels(self, X, c):
            """
            # Maps class c to 1 and all others to -1
            :param X: label matrix
            :param c: class to be mapped to 1
            :return: transformed label matrix
            """

            numRow = np.size(X, 0)
            numCol = np.size(X, 1)
            for i in range(numRow):
                for j in range(numCol):
                    if X[i, j] == c:
                        X[i, j] = 1
                    else:
                        X[i, j] = -1

            return X

        def _calculateR(self, covMatrix):
            """
            Compute R as described in "Ranking and combining multiple predictors without labeled data" Parisi et al.
            2014
            :param covMatrix: labeler covariance matrix
            :return: matrix R
            """

            numCol = np.size(covMatrix, 1)
            R = covMatrix
            equationMat = np.zeros([int((numCol - 1) * (numCol) / 2), numCol])
            RHS = np.zeros([int((numCol - 1) * (numCol) / 2)])
            rowCount = -1
            for i in range(numCol):
                for j in range(numCol):
                    if (j > i):
                        rowCount = rowCount + 1
                        equationMat[rowCount, i] = 1
                        equationMat[rowCount, j] = 1
                        RHS[rowCount] = np.log(abs(covMatrix[i, j]))

            # Get least squares solution to the system, remove equations where RHS is a very large negative number
            # That happens when the covariance is very close to zero
            try:
                indices = RHS > np.log(0.0001)
                a = np.linalg.lstsq(equationMat[indices, :], RHS[indices], rcond=None)[0]
                for i in range(numCol):
                    R[i, i] = np.exp(2 * a[i])
                flag = 0
            except:
                R = None
                flag = 1

            return R,flag

        def _tensorEstimation(self,X,v):
            """
            Estimate class balance as described in "Estimating the accuracies of multiple classifiers without labeled
            data" Jaffe et al. 2015
            :param X: matrix of labels
            :param v: column vector such that vv^T == R
            :return: class imbalance parameter b
            """

            numRow = np.size(X,0)
            numCol = np.size(X,1)
            meanVector = np.mean(X, axis=0)

            T = np.zeros((numCol,numCol,numCol))
            eigest = np.zeros((numCol,numCol,numCol))
            for i in range(numCol):
                for j in range(numCol):
                    for k in range(numCol):
                        eigest[i,j,k] = v[i]*v[j]*v[k]
                        T[i,j,k] = np.mean((X[:,i]-meanVector[i])*(X[:,j]-meanVector[j])*(X[:,k]-meanVector[k]))

            Tvec = T.flatten()
            eigestvec = eigest.flatten()
            Tvec = Tvec[:,None]
            eigestvec = eigestvec[:,None]

            alpha = np.linalg.lstsq(eigestvec, Tvec, rcond=None)[0][0]

            b = -alpha/np.sqrt(4.0+alpha**2)

            return b
