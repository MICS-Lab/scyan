=============
API Reference
=============

Scyan model
~~~~~~~~~~~
The main class that you are going to use, including a PyTorch Lightning Trainer, a sampling method, a prediction methods and others. In particular, it wraps the ScyanModule described below.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   scyan.model.Scyan

The core functions are implemented inside a module called ScyanModule. In particular, this is where the loss function is defined. This module is an attribute of the main model above.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst

   scyan.module.scyan_module.ScyanModule

Plotting tools
~~~~~~~~~~~~~~
We provide many interpretation tools.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   scyan.plot

Other Modules
~~~~~~~~~~~~~
To go deeper inside the API.

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   scyan.data
   scyan.metric
   scyan.module
   scyan.utils