Getting Started
===============

Installation
-------------

* Step 1: Download the main branch from our GitHub page as a zip file, or clone it to your working directory with:

    ``git clone https://github.com/openforcefield/tmos``

* Step 2 (Optional): If you are using conda and you want to create a new environment for this package you may install with:

    ``conda env create -f requirements.yaml``

* Step 3: Install package with:

    ``pip install tmos/.``

    or change directories and run

    ``pip install .``

    Adding the flag ``-e`` will allow you to make changes that will be functional without reinstallation.

* Step 4: Initialize pre-commits (for developers)

    ``pre-commit install``
