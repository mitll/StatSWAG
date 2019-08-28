"""
Generate samples of synthetic data sets and sets of corresponding expert labels.
"""

import numpy as np

def make_classification_labels(y,X=None,n_labelers=5,confusion=None):
    """Given a feature set and ground truth labels, generate additional labels.

    Parameters
    ----------
    y : array-like, shape=[n_samples,]
        The ground truth labels
    X : array-like, shape=[n_samples,n_features]
        The features, currently not used but in the future may be implemented
        for generating more complex expert labels with dependencies.
    n_labelers : integer, default=5
        How many sets of synthetic labels to generate
    confusion : array-like, shape=[n_labelers,]
        An array of confusion matrices for the labelers
        If only one is provided, will apply to all experts
        If None is provided, will be set to no confusion
        Row/column order must be according to standard built-in sort

    Returns
    -------
    Z : array-like, shape=[n_samples,n_labelers]
        The set of labeler responses for each sample
    """
    class_names = np.sort(np.unique(y))
    n_classes = len(class_names)
    n_samples = len(y)

    # Build a fresh confusion matrix for each labelers
    if confusion is None:
        confusion = []
        # Create a perfect no-confusion matrix
        c_mat = np.zeros((n_classes,n_classes))
        np.fill_diagonal(c_mat,1)
        confusion = c_mat

    # Initialize empty array of labels
    Z = np.zeros((n_samples,n_labelers),dtype='int64')

    # Generate labels
    for n in range(n_labelers):
        for i in range(n_samples):
            y_true = y[i]
            # If it's one confusion matrix instead of two
            if len(np.shape(confusion))<3:
                conf = confusion[y_true]
            else:
                conf = confusion[n][y_true]
            Z[i][n] = np.random.choice(class_names,p=conf)

    return Z
