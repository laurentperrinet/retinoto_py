# Foveated Retinotopy: Giving a biological look to deep learning convolutional networks.

![PyPI version](https://img.shields.io/pypi/v/retinoto_py.svg)[![Documentation Status](https://readthedocs.org/projects/retinoto_py/badge/?version=latest)](https://retinoto_py.readthedocs.io/en/latest/?version=latest)

* PyPI package: https://pypi.org/project/retinoto_py/
* Free software: MIT License
* Documentation: https://retinoto_py.readthedocs.io.

## TODO list in chronological order

* DONE : use Subset instead of n_val_stop.
* DONE : clean for `in_memory` code. fait, bon débarras !
* DONE : pretrain FC pour 20 - pas conclusif
* DONE  : hexagonal tiling for log-polar grid / mettre test_hexagonal_grid dans une notebook / test circular padding - validé, ça marche effectivement mieux
* DONE : regénérer bbox avec `fovea.fixate` = nouveau dataset `focus` / 
* DONE : test different costs: 'BCEWithLogitsLoss' is worst, 'CrossEntropyLoss' is best but 'NegLogitLoss' is pretty good. 
* TODO : test different optimizers: close tie between adam and adamw
* TODO : regarder https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomIoUCrop.html#torchvision.transforms.v2.RandomIoUCrop
* TODO : do visual search with a prompt
* TODO : use ecoset

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.