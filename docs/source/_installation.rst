Installation
============

**PINA** requires requires `torch`, `lightning`, `torch_geometric` and `matplotlib`. 

Installing via PIP
__________________

Mac and Linux users can install pre-built binary packages using pip.
To install the package just type:

.. code-block:: bash

    $ pip install pina-mathlab

To uninstall the package:

.. code-block:: bash

    $ pip uninstall pina-mathlab

Installing from source
______________________
The official distribution is on GitHub, and you can clone the repository using

.. code-block:: bash
    
    $ git clone https://github.com/mathLab/PINA

To install the package just type:
 
.. code-block:: bash

    $ pip install -e .


Install with extra packages
____________________________

To install extra dependencies required to run tests or tutorials directories, please use the following command:

.. code-block:: bash

    $ pip install "pina-mathlab[extras]" 


Available extras include:

* `dev` for development purpuses, use this if you want to Contribute.
* `test` for running test locally.
* `doc` for building documentation locally.
* `tutorial` for running tutorials
