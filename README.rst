===================================================
Statistics Without Affirmed Ground Truth - StatSWAG
===================================================

StatSWAG implements several statistical estimators that, given noisy categorical predictions (labels) from multiple labelers for a set of data samples, estimate both the accuracy of each individual labeler and the true label for each data instance.

Installation
------------

Complete the following from within the top-level directory.

Install the necessary requirements with::

    pip install -r requirements.txt

Install the package::

    pip install .

Now, you should be able to open a Python shell and run::

    import statswag

or any of its libraries.

Usage
-----

.. code:: python

  from statswag.estimators import IWMV
  import numpy as np

  labels = [[0,1,1],
            [0,0,1],
            [1,1,1],
            [0,0,0],
            [1,1,1],
            [0,0,1],
            [1,1,1],
            [0,1,0],
            [0,0,0],
            [1,0,1]]
  labels = np.asarray(labels)

  iwmv = IWMV()
  result = iwmv.fit(labels)
  print(result)

More detailed examples of basic usage can be found in `statswag/examples/`.

We recommend beginning with "Getting Started" which explains a little about the different estimators bundled in this package.  Options for each estimator are discussed as well.

"Using Simulated Data" shows how to create simulated data for testing of the estimators in different scenarios.

"Using Your Own Data" discusses how to format your own data to use with this package.  Specifically, it discusses the options for encoding your data as well as how to deal with missing labels.


Development
-----------

Testing
~~~~~~~

To run the tests, you will need to have the `pytest` package installed::

    pip install pytest

Launch the test suite from within the `StatSWAG` top-level directory with::

    pytest statswag