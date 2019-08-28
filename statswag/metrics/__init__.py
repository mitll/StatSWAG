import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

def nan_accuracy(y_true,y_pred):
    """Compute accuracy of non-nan entries

    This method aligns the two sets of labels and removes any row where one or
    more label is NaN. With the remaining labels, it computes accuracy using
    sklearn's accuracy_score function.

    Parameters
    ----------
    y_true : The array of gold-standard labels
    y_pred : The array of predicted labels

    Returns
    -------
    accuracy_score : float
    """
    # Drop NaNs
    df = pd.DataFrame(np.array([y_true,y_pred]).T)
    # To handle string version of nan
    df = df.replace('nan',np.nan).dropna(axis=0)
    y_true = df[0].values
    y_pred = df[1].values

    return accuracy_score(y_true,y_pred)

def nan_confusion_matrix(y_true,y_pred):
    """Create a confusion matrix from the non-nan entries of two label sets

    Parameters
    ----------
    y_true : The array of gold-standard labels
    y_pred : The array of predicted labels
    Returns
    -------
    confusion_matrix : Array-like, shape=[n_classes,n_classes]
    """
    flat_list = np.concatenate((y_true,y_pred)).flatten()
    class_names = pd.Series(flat_list).unique()
    conf_array = np.zeros((len(class_names),len(class_names)))
    conf_mat = pd.DataFrame(conf_array,columns=class_names,index=class_names)
    for i in range(len(y_true)):
        conf_mat.loc[y_true[i],y_pred[i]] += 1
    return conf_mat
