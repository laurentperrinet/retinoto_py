.. _retinoto_py-docs-index:

Welcome to retinoto_py's documentation!
========================================

Retinoto`_py`` is an open-source Python package for applying spatial transformations based on visual field representations in the brain of different species including the human brain. It allows researchers and engineers to analyze and interpret data using different transformation topologies, including log-polar mappings that simulate biological retina structures. The project provides tools for both theoretical modeling and practical application development in computational neuroscience and computer vision research.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
 
 
   :doc:`00_installation`
   :doc:`01_quickstart`
   :doc:`02_basic-usage`
   :doc:`03_advanced-usage`
 
 .. _tutorials-section:

Tutorials & Examples
--------------------

The following notebooks demonstrate the capabilities of retinoto_py, organized by topic for ease of navigation. The first tutorial series shows how to set up and use the package's core features through interactive examples - explore them in order or jump directly to topics relevant to your current work.


.. _setup-and-installation:

Setup & Installation
++++++++++++++++++++

The initial configuration covers installation steps and project setup. The notebooks walk through basic usage patterns for applying log-polar transformations: starting from the fundamental concepts, moving toward real-world applications like medical imaging. Each tutorial builds incrementally so you gain practical experience with the package's API as we go. All example codes include comments to help you customize them for your own projects - explore how data flows through each module and discover which functions work best for your use case:

* :doc:`00_installation.ipynb` - Step-by-step installation guide
* :doc:`04_imagenet-bbox-dataset.ipynb` - Working with ImageNet bounding box datasets


.. _learning-and-transfer:

Training & Transfers
++++++++++++++++++++|

The second section demonstrates how trained models can be transferred between architectures using retinoto_py's built-in transfer learning capabilities. These notebooks show progressive complexity from simple experiments to advanced multi-task learning - perfect for adapting our work on biological-inspired vision systems (such as those studied in [Perrinet 2013](https://doi.org/1XXX-YYY)) to your research questions:

* :doc:`20_transfer-learning-resnet101.ipynb`
* :doc:`21_transfer-learning-convnext.ipynb`
* :doc:`31_transfer-learning-resnet101-fovea.ipynb`


.. _retinotopic-transforms:

Retinotopic Transforms and Analysis
++++++++++++++++++++++++++++++++++++|

These notebooks focus on analyzing how visual information transforms through the brain's retinotopic mapping - key for understanding neural representations of space. We start with basic log-polar coordinates that represent visual field topology, then move into advanced techniques for measuring likelihood maps from real brain scan data:

* :doc:`30_retinotopic-mapping.ipynb`
* :doc:`50_retinotopic-mapping-where.ipynb`


.. _advanced-topics:

Advanced Topics
+++++++++++++++|

For users working with specialized topics or complex datasets, these notebooks cover edge cases and advanced processing pipelines. Examples include rotation attacks on image recognition systems (important for understanding robustness of biological-inspired architectures), focus-based learning methods for improving accuracy in low-signal regions, and optimization techniques for bounding box estimation:

* :doc:`34_optimize-bbox.ipynb`
* :doc:`40_compute_likelihood_map.ipynb`


.. _data-and-preprocessing:

Data & Preprocessing Tools
++++++++++++++++++++++++++++|

This section covers supporting utilities - how to work with datasets, apply transformations efficiently, and prepare raw neural recordings. These notebooks help build the infrastructure needed before running experiments described elsewhere in this documentation set, from basic utility methods through full multi-modal pipeline processing scripts for image sequences:

* :doc:`08_dataloaders.ipynb`
* :doc:`41_multiple_likelihood_map.ipynb`


.. _full-stack-workflows:
Full Stack Workflows
++++++++++++++++++++|

The final section brings together everything you've learned so long as your system supports all the dependencies we use throughout these notebooks - explore complete end-to-end pipelines that apply retinotopy principles across multiple domains sequentially. These workflows demonstrate production-ready strategies for applying our framework to new problems: 

* :doc:`99_run-all-the-stack.ipynb`


Indices and tables
==================

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
