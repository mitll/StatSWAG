from __future__ import division

import logging
import warnings
import numpy as np
from statswag.estimators import base
from collections import Counter,defaultdict
from sklearn.utils import Bunch

# Display progress logs on stdout
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')


class Agreement(base.BaseEstimator):
    def __init__(self):
        """Initialize a Agreement estimator.

        Parameters
        ----------
        bins : array-like, shape=(N,)
            The bins to use for estimating accuracy.
        """
        self.accuracies = None
        self.labels = None
        self.bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        self.num_classes = 0

        # This is in case a user wants to inspect various steps of the
        # algorithm and see which prob. correct / baserate / gt
        # corresponds to which class names, since those end up being
        # arbitrarily ordered.
        self.class_names = None

    def _percent_agreement(self,expert_labels):
        """Calculates the percent of agreement across all experts and samples

        Returns
        -------
        percent_agreement : float
            The amount of agreement between all experts.
        """
        num_samples,num_experts = expert_labels.shape
        num_agree = 0
        for i in range(num_experts):
            for j in range(i+1,num_experts):
                num_agree = num_agree + np.sum(expert_labels[:,i]==expert_labels[:,j])
        percent_agreement = num_agree/(num_samples*(num_experts-1)*num_experts/2)
        return percent_agreement

    def _prob_correct(self,expert_labels):
        """Probability of an expert being correct.

        There is one value for the entire matrix of experts & samples because
        all experts are assumed to share the same confusion matrix.

        In the case that the percent agreement is less than 1/N, where N is the
        number of classes, this function will return None. This is a known edge
        case that results in algorithm failure.
        """
        N = self.num_classes
        percent_agreement = self._percent_agreement(expert_labels)
        # TODO: Move the catch case up to here for invalid percent agreement

        # Eq. 2 from Agreement et al.
        val_under_sqrt = ((N-1)*percent_agreement-(N-1)/N)/N

        # Negative value occurs when percent agreement is less than 1/N
        if val_under_sqrt <= 0:
            logging.warn("Percent agreement among expert labels is less than 1/{}".format(N))
            return None
        else:
            prob_correct = (1/N) + np.sqrt(((N-1)*percent_agreement-(N-1)/N)/N)
        return prob_correct

    def _base_rates(self,expert_labels):
        """Base rate of each class label.
        """
        N = self.num_classes
        num_samples,num_experts = expert_labels.shape
        prob_correct = self._prob_correct(expert_labels)

        # If the last step didn't work, this step can't work either
        if prob_correct is None:
            return None

        counts = Counter(np.ndarray.flatten(expert_labels))
        base_rates = []

        # Calculate the base rate for each class label
        for class_name in self.class_names:
            observed_frequency = counts[class_name]/(num_samples*num_experts)
            # Eq. 5 from Agreement et al.
            base_rate = ((N-1)*observed_frequency-1+prob_correct)/(N*prob_correct-1)
            if base_rate<0:
                logging.warn("Base rate was negative.")
                return None
            base_rates.append(base_rate)

        logging.info("Base rates: {}".format(base_rates))

        return base_rates

    def _ground_truth_probs(self,expert_labels):
        """
        For each label, calculate the ground-truth probability of each possible
        label for each of the samples.
        """
        num_samples,num_experts = expert_labels.shape
        prob_correct = self._prob_correct(expert_labels)
        base_rates = self._base_rates(expert_labels)
        if base_rates is None:
            return None

        ground_truth_probs = np.zeros((num_samples,self.num_classes))
        for i in range(num_samples):
            for j in range(self.num_classes):
                ground_truth_probs[i,j] = base_rates[j]
                for k in range(num_experts):
                    if expert_labels[i,k] == self.class_names[j]:
                        ground_truth_probs[i,j] *= prob_correct
                    else:
                        ground_truth_probs[i,j] *= (1-prob_correct)/(self.num_classes-1)

        # Normalize
        for i in range(num_samples):
            ground_truth_probs[i,:] = ground_truth_probs[i,:]/np.sum(ground_truth_probs[i,:])

        logging.info("Ground truth probabilities: {}".format(ground_truth_probs))
        return ground_truth_probs

    def _generate_labels(self, labels):
        num_samples,num_experts = labels.shape
        ground_truth_probs = self._ground_truth_probs(labels)
        if ground_truth_probs is None:
            logging.warn("Did not receive ground truth probabilities.")
            return None
        highest_GTP_labels = []
        for row in ground_truth_probs:
            max_cols = np.nonzero(row==np.max(row))[0]
            highest_label = np.random.choice(max_cols)
            highest_GTP_labels.append(highest_label)
        gen_labels = [self.class_names[l] for l in highest_GTP_labels]
        return gen_labels

    def _sys_accuracy(self,expert_labels,classifier_labels):
        num_samples,num_experts = expert_labels.shape
        ground_truth_probs = self._ground_truth_probs(expert_labels)
        if ground_truth_probs is None:
            return None
        N = self.num_classes

        # Bin the data by 10 percentiles
        highest_GTP = []
        highest_GTP_idx = []
        for row in ground_truth_probs:
            max_cols = np.nonzero(row==np.max(row))[0] # Indices of max label
            highest_label = np.random.choice(max_cols) # Choose random index
            highest_GTP_idx.append(highest_label)
            highest_GTP.append(row[highest_label])

        highest_GTP = np.array(highest_GTP)
        highest_GTP_label = [self.class_names[i] for i in highest_GTP_idx]

        bin_count = []
        avg_bin = []
        bin_agreement = []
        percentiles = self.bins
        for i in range(len(percentiles)-1):
            inrange = [highest_GTP[j] for j in range(len(highest_GTP))
                       if (highest_GTP[j] > percentiles[i] and highest_GTP[j] <= percentiles[i+1])]
            if len(inrange) > 0:
                bin_count.append(len(inrange))
                avg_bin.append(np.mean(inrange))
                agreement_bool = [highest_GTP_label[j] == classifier_labels[j] for j in range(num_samples)
                           if (highest_GTP[j] > percentiles[i] and highest_GTP[j] <= percentiles[i + 1])]
                bin_agreement.append(np.mean(agreement_bool))
            else:
                bin_count.append(0)
                avg_bin.append(0)
                bin_agreement.append(0)

        logging.info("Bin Counts: {}".format(bin_count))
        # Had to pull this out into multi-line for loop to perform the conditional check
        bin_accuracy = []
        for i in range(len(bin_count)):
            if((N*avg_bin[i]-1) <= 0):
                logging.info("Bin count insufficient")
                bin_accuracy.append(bin_agreement[i])
            else:
                bin_accuracy.append(((N-1)*bin_agreement[i]-1+avg_bin[i])/(N*avg_bin[i]-1))

        bin_accuracy = np.asarray(bin_accuracy)
        logging.info("Bin Accuracy: {}".format(bin_accuracy))

        # Occurs when Pg > 1/N and Pg > Pa
        # Or when Pg < 1/N and Pg < Pa
        bin_accuracy[bin_accuracy > 1] = 1

        # Occurs when Pa < (1-Pg) / (N-1) where Pg = agv. ground truth probability
        bin_accuracy[bin_accuracy < 0] = 0

        # Ensure that there are no invalid entries in the bin accuracy list
        for ba in bin_accuracy:
            if ba>1 or ba<0:
                logging.warning("Bin accuracy elements are not between 0 and 1.")
                return None
        accuracy = np.sum(bin_accuracy*bin_count/num_samples)

        return accuracy

    def fit(self,X,return_pi_T=False):
        """This method fits the estimation model to a provided set of expert labels.

        In the special case that the rate of agreement for an expert is less than
        1/N, where N is the number of classes, that expert's estimated accuracy
        will be None.

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
        all_labels = np.array(X)
        num_samples,num_experts = all_labels.shape
        self.class_names = np.unique(all_labels)
        self.num_classes = len(self.class_names)

        # Iterate through to get accuracy for each expert/classifier
        accuracies = np.zeros(num_experts)
        for i in range(num_experts):
            classifier_labels = all_labels[:,i]
            expert_labels = np.delete(all_labels,i,axis=1)
            accuracies[i] = self._sys_accuracy(expert_labels,classifier_labels)
        self.accuracies = accuracies

        # Generate best-guess labels using weighted majority vote
        all_votes = np.array([np.sum(self.accuracies * (all_labels == k).astype(int), axis=1)
                              for k in self.class_names])

        self.labels = np.array([self.class_names[i] for i in np.argmax(all_votes, axis=0)])

        if return_pi_T:
            return self.accuracies,self.labels
        else:
            return Bunch(
                accuracies=self.accuracies,
                labels=self.labels,
                probs=None,
                class_names=self.class_names)
