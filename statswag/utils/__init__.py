import numpy as np

def remove_labels(Z,mode="random",**kwargs):
    """Removes labels from an expert label according to a specified mode
    If mode == random, *args should set "count" variable
        Labels will be randomly removed uniformly from each column according to
        count, per_sample and per_expert are set automatically to satisfy count
    If mode == stacked, *args should set "per_sample" and "per_expert" variables
        Labels will be removed in order from each column
    If mode == bin, *args should set "per_sample" and "per_expert" variables
        Labels will be removed randomly until per_sample is achieved and the
        closest per_expert can be achieved (last expert may not label as many)
    """
    # Build the empty array we'll copy labels into
    nan_Z = np.empty(np.shape(Z),dtype='object')
    nan_Z[:] = np.nan
    if mode == "random":
        if "count" in kwargs:
            count = kwargs["count"]
            n_samples,n_experts = np.shape(Z)
            sample_ids = range(n_samples)
            for expert in range(n_experts):
                to_label = np.random.choice(sample_ids,count,replace=False)
                for id in to_label:
                    nan_Z[id][expert] = Z[id][expert]
        else:
            per_sample,per_expert = kwargs["per_sample"],kwargs["per_expert"]
            Z = np.array(Z)

            n_samples,n_experts = np.shape(Z)
            sample_ids = range(n_samples)

            # Contains all the sample ids we want to label * num per sample
            instances = [x for i in [sample_ids for j in range(per_sample)] for x in i]
            # For each expert, select "per_expert" from available instances
            for expert in range(n_experts):
                # This will hold all the ids this expert is going to label
                to_label = []
                for n in range(per_expert):
                    # Randomly select a label that is not any sample number the expert
                    # has already chosen in their set to label
                    remaining = list(set(instances).difference(set(to_label)))
                    if len(remaining)==0:
                        break
                    new_label = np.random.choice(remaining)
                    to_label.append(new_label)
                    instances.remove(new_label)
                for id in to_label:
                    nan_Z[id][expert] = Z[id][expert]
    elif mode == "stacked":
        Z = np.array(Z)
        n_samples,n_experts = np.shape(Z)
        for i in np.arange(0,int(n_experts/per_sample)):
            nan_Z[per_expert*i:per_expert*(i+1),per_sample*i:per_sample*(i+1)] = \
                Z[per_expert*i:per_expert*(i+1),per_sample*i:per_sample*(i+1)]
    return nan_Z

def build_matrix(alpha,n_classes):
    """Builds a classifier matrix where alpha is the accuracy

    Errors are distributed uniformly across n_classes not along diagonal
    """
    c_mat = np.zeros((n_classes,n_classes))
    for i in range(n_classes):
        c_mat[i,i] = alpha
        for j in np.delete(np.arange(n_classes),i):
            c_mat[i,j] = (1-alpha)/(n_classes-1)
    return c_mat
