from __future__ import division

import numpy as np
from abc import ABCMeta, abstractmethod
import six

def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'
    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int
        The offset in characters to add at the begin of each line.
    printer : callable
        The function to convert entries to strings, typically
        the builtin str or repr
    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(six.iteritems(params))):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75 or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines



class BaseEstimator(six.with_metaclass(ABCMeta)):
    """Base class for all estimators in StatSWAG."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        """Placeholder for fit. Subclasses should implement this method!

        Fit the accuracy estimation algorithm to a set of expert/classifier
        labels. This method need ALWAYS be called prior to calling estimate
        or generate.

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
            best-guess labels, and 'probs' the probability of each possible
            label for each sample (if available).

        (accuracies, labels) : tuple if ``return_pi_T`` is True
        """
        pass

    def get_params(self, deep=True):
        """Get parameters for this generator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this generator and
            contained subobjects that are generators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out


    def set_params(self, **params):
        """Set the parameters of this generator.
        The method works on simple generators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for generator %s. '
                                 'Check the list of available parameters '
                                 'with `generator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
