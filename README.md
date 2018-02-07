# Reproducing Densely Interactive Inference Network in Keras

This repository aims to reproduce the results obtained in
[Natural Language Inference over Interaction Space](https://arxiv.org/abs/1709.04348) paper.
We've chosen this paper as a project for 
[reproducibility challenge](http://www.cs.mcgill.ca/~jpineau/ICLR2018-ReproducibilityChallenge.html) organized by ICLR.
DIIN paper in OpenReview: https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-


### Problem statement
Given a premise sentence and a hypothesis one needs to determine whether hypothesis is
an entailment of the premise, a contradiction, or a neutral sentence. So given two sentences
we need to classify those between these 3 classes (`entailment`, `contradiction`, `neutral`).

Several samples from MultiNLI dataset are presented below which are copied from DIIN paper.

`Premise`: The FCC has created two tiers of small business for this service with the
approval of the SBA.
`Hypothesis`: The SBA has given the go-ahead for the FCC to divide this service into
two tiers of small business.
`Label` entailment.

`Premise`: He was crying like his mother had just walloped him.
`Hypothesis`: He was crying like his mother hit him with a spoon.
`Label`: Neutral

`Premise`: Later, Tom testified against John so as to avoid the electric chair.
`Hypothesis`: Tom refused to turn on his friend, even though he was slated to be executed.
`Label`: Contradiction


### Architecture
`Encoding`, `Interaction`, and also exponentially `DecayingDropout` can be found in `layers/` package.
Feature extractor (in our case [DenseNet](https://arxiv.org/abs/1608.06993)) can be found in `feature_extractors/` package.
L2 optimizer wrapper can be found in `optimizers/`.
![](images/architecture.png "Architecture")


### Instructions

Our code is compatible with both `python3` and `python2` so for all commands listed below `python` can be substituted
by `python3`.

* Install requirements (`pip3` for `python3`)
```commandline
pip install -r requirements.txt
```

* Preprocess the data
```commandline
python preprocess.py --p 32 --h 32 --chars_per_word 16 --save_dir ./data/ --dataset snli --word_vec_path ./data/word-vectors.npy
```

* Train the model
```commandline
python train.py --batch_size 70 --eval_interval 500 --char_embed_size 8 --char_conv_filters 100 --train_word_embeddings --load_dir ./data --models_dir ./models/ --logdir ./logs --word_vec_path ./data/word-vectors.npy
```

* See the results in TensorBoard
```commandline
tensorboard --logdir=./logs
```

Currently we managed to obtain 87.27% accuracy on `test` set.