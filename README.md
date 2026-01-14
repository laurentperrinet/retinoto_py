# Foveated Retinotopy: Giving a biological look to deep learning convolutional networks.

![PyPI version](https://img.shields.io/pypi/v/retinoto_py.svg)[![Documentation Status](https://readthedocs.org/projects/retinoto_py/badge/?version=latest)](https://retinoto_py.readthedocs.io/en/latest/?version=latest)

* PyPI package: https://pypi.org/project/retinoto_py/
* Free software: MIT License
* Documentation: https://retinoto_py.readthedocs.io.

## TODO list in chronological order

* DONE : 2025-12-18 pretrain for faster warmstart FC pour 20 - pas conclusif
* DONE : 2025-12-25 use Subset instead of n_val_stop.
* DONE : 2025-12-27 clean for `in_memory` code. fait, bon débarras !
* DONE : 2026-01-12 hexagonal tiling for log-polar grid / mettre test_hexagonal_grid dans une notebook / 2026-01-12 - validé, ça marche effectivement mieux
* DONE : regénérer bbox avec `fovea.fixate` = nouveau dataset `focus` / 2026-01-14 test sur Jean-Zay, monte à 75% assez vite puis descencd (mais avec  'NegLogitLoss')
* DONE : test different costs: 2026-01-14 'BCEWithLogitsLoss' is worst, 'CrossEntropyLoss' is best but 'NegLogitLoss' is pretty good. 
* TODO : test different optimizers: 2026-01-14  close tie between adam and adamw
* TODO : scheduling
* TODO : remove all resnet testing / learning and focus on convNext ?
* TODO : semi-supervised learning: using YOLO / fasterRCNN method to extract images - regarder https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomIoUCrop.html#torchvision.transforms.v2.RandomIoUCrop
* TODO : use ecoset
* TODO : relancer 1* 2* 3* avant de passer à focus
* TODO : relancer les likelihood maps / ajouter des crops à différentes échelles
* TODO : check if we see center / surround effects
* TODO : test circular padding
* TODO : make a test that the recognition should work on very large images in a better way than a resizing
* TODO: theroretical derivation of the joint encoding / decoding ("how is it to look through an eye")
* TODO: learn the differentiable retinotopic mapping
* TODO: log-polar mapping is a good prior
* TODO: Event-based retinotopy / optic flow detection
* TODO: where network : teacher student / get symmetries to find the best topology 
* TODO : do visual search with a prompt

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.