DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the U.S. Air Force.

© 2019 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

===================================================
Statistics Without Affirmed Ground Truth - StatSWAG
===================================================

This is the repository for StatSWAG code.
The project is using Sphinx for documentation.

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

Examples of basic usage can be found in `statswag/examples/`.

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
