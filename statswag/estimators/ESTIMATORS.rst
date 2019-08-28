==========
Estimators
==========

Majority Vote
-------------

Generic majority vote algorithm.

Algorithm:

#. For each sample, calculate the number of votes for each class.
#. Choose the best-guess label as the most popular vote for each sample.
#. If there are more than one most popular label, randomly pick one.
#. Use the best-guess labels to calculate labeler accuracy.

Iterative Weighted Majority Vote (IWMV)
---------------------------------------

Extension of majority vote in which weights are updated each iteration.

Paper: https://arxiv.org/pdf/1411.4086.pdf

Algorithm:

#. Initially, give all labeler votes equal weight (*v* is all ones)
#. **While** not converged:
#. Estimate the best-guess labels (*y_hat*) by calculating a weighted sum of the
   votes for each sample (using *v* as the weights)
#. Using the best-guess labels (*y_hat*), calculate labeler accuracies (*w_hat*)
#. Using the labeler accuracies (*w_hat*), update the labeler weights (*v*)
#. **If** the maximum number of iterations has been reached or the best-guess
   labels (*y_hat*) haven't changed since last round, the algorithm has
   converged

Agreement
---------

Uses a simple relationship between agreement and accuracy to estimate performance of a target labeler.

Paper: https://digitalcommons.wayne.edu/jmasm/vol14/iss1/13/

Algorithm:

#. **For each** expert column in the set of expert labels:
#. Separate the `target` column from the `expert` columns
#. Calculate the percent of agreement between all `expert` columns
#. Use this agreement rate to calculate each expert's probability of being correct
#. Use the probability of being correct to calculate base rates for each label
#. Using the base rates for each label and each experts probability of being
   correct, calculate the probability of each label for each sample
#. Best-guess labels are those with the highest probability from the last step
#. Calculate the `target` accuracy against these best-guess labels using the agreement relationship again.

Spectral
--------

Derives an estimate of accuracy for each class using a relationship between accuracy and the labeler covariance matrix.

Paper: http://proceedings.mlr.press/v38/jaffe15.pdf

Algorithm: Complex

MLEOneParameterPerLabeler
-------------------------

Approximates the maximum likelihood estimate of labeler accuracy using Expectation Maximization in a statistical model where the labelers are assumed to be conditionally independent, given the true label.

Papers: http://www.jmlr.org/papers/volume11/donmez10a/donmez10a.pdf, https://www.jstor.org/stable/2346806?seq=1#metadata_info_tab_contents

Algorithm: Complex