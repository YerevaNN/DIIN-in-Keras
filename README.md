# Reproducing Densely Interactive Inference Network in Keras

This repository aims to reproduce the results obtained in
[Natural Language Inference over Interaction Space](https://arxiv.org/abs/1709.04348) paper.

We've chosen this paper as a project for 
[reproducibility challenge](http://www.cs.mcgill.ca/~jpineau/ICLR2018-ReproducibilityChallenge.html) organized by ICLR.

DIIN paper in OpenReview: https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-


### Architecture:
`Encoding`, `Interaction`, and also exponentially `DecayingDropout` can be found in `layers/` package.
Feature extractor (in our case [DenseNet](https://arxiv.org/abs/1608.06993)) can be found in `feature_extractors/` package.
Custom optimizers `AdadeltaL2` and `SGDL2` are implemented in `optimizers/`.
![](images/architecture.png "Architecture")


### Instructions

Our code is compatible with both `python3` and `python2` so for all commands listed below python can be substituted
by `python3`.

* Install requirements (`pip3` for `python3`)
```commandline
pip install -r requirements.txt
```

* Preprocess the data
```commandline
python preprocess.py --p 32 --h 32 --chars_per_word 5 --save_dir ./data/ --dataset snli --word_vec_path ./data/word-vectors.npy
```

* Train the model
```commandline
python train.py --batch_size 70 --char_embed_size 100 --char_conv_filters 77 --load_dir ./data --models_dir ./models/ --logdir ./logs --word_vec_path ./data/word-vectors.npy
```

* See the results in TensorBoard
```commandline
tensorboard --logdir=./logs
```

Currently we managed to obtain 84.4% accuracy on `test` set.