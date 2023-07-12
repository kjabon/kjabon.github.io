---
layout: distill
title: Image Regression Lessons Learned (in JAX)
description: An exercise in image processing
giscus_comments: true
date: 2023-07-10
tags: rl 
authors:
  - name: Kenneth Jabon

  

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Future return

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---
## Note from the author:
Please note, this post is incomplete (still needs figures, editing, and completion of content)

To get up to speed with image processing, and doing so in JAX, I decided to try something not typically found in your starter image classification notebook.

# The problem
 - ~100k samples: x: X-ray images of lungs, y: age label (this is a [dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data) found on kaggle)
 - Perform regression from image to age.
(Put a couple image examples here.)

# The process

## Get signs of life, and starter tips

First, you want to make sure your training pipeline actually works and is bug-free. Does your training loss curve reliably decrease and converge to some reasonable (in the context of your dataset) minimum? If it is merely oscillating in place with no sign of training on the dataset, or forever increasing, many things could be going wrong. A bug in your code should always be your first guess. in deep learning, things have a sneaky tendency to fail silently. Without correct code, you could easily spend days chasing barely perceptible gains by tweaking architecture, hyperparameters, trying different optimizers, etc., when it turned out you were computing your loss incorrectly, or simply forgot to apply your gradient. 

Anything is on the table as far as bugs go, and for this reason it's important to just get *something that works* as soon as possible. Start with a relatively low-complexity model, like a basic CNN. If you're at the point where this seems menacing, just start with a basic MLP, or even linear regression. Start simpler and work your way up in complexity later. Prove to yourself that your training loop is computing loss and gradients and applying them correctly. Prove to yourself that your dataset is actually labelled correctly by going through it manually. Better yet, start from a known working dataset online to prove that your training pipeline works, then swap in the dataset of interest.

Even with correct code, your dataset may have little to no real relationships to be learned at all! For this reason it's important to attune yourself to the dataset. Really immerse yourself in the relationships you expect and the mechanisms behind them, if any. Any deep learning practitioner needs some data scientist tools in their belt. Visualize, normalize, and clean your dataset.

Once you are satisfied the code is correct, you may still not be training effectively. Your learning rate could be too low, or too high (this is typically the first thing to check). Your model may not have enough capacity to capture the relationships in the data. This boils down to hyperparameter optimization.

## Overfit on purpose

As with any supervised learning problem, your first goal is to overfit the problem just a bit. I.e., while paying no mind to the validation loss, see that your model is actually capable of learning the dataset, whether or not it is generalizing well or overfitting. Put another way, you make sure you have enough model capacity to accurately map from x to y.

We'll start with a Resnet with as few layers as we can, and start adding blocks until we get a desirable training error. In this case, let's consider ourselves satisfied once our training error is less than 1 (estimating age within +/- 1 years). In theory we could go arbitrarily low, effectively memorizing the dataset.

In the figure below, we've run each of these Resnet models 3 times with different random seeds, with early stopping.  Clearly, increasing the number of blocks (parameters) allows it to more accurately model the dataset, but we see diminishing returns, possibly suggesting that we are nearing the actual relationship that the data actually shows.

[Insert graph here]

This said, it's often a good idea to leave some wiggle room, rather than using a model that is just barely good enough. It helps to allow the model multiple avenues to approximate the dataset, rather than there being one perfect set of trained weights. As it turns out, having a model that is just barely capable of accurately representing the dataset may actually be detrimental to performance, so it's best to give it a little extra capacity once you hit a minimum. Once we start regularizing and adding things like dropout, a little more capacity won't hurt either: e.g. because without we use only a fraction of the network during training. We'll worry about increasing the capacity later if need be.

## Regularize

At this point, it's practically guaranteed we are overfitting the data, particularly as our model size increases.
Let's take a look at the validation error curves from the previous training run.

(Show the curves)

Yes, so we can see that our validation error is quite high. 

The generalization error is the difference between validation and training errors. If generalization error is higher than training error, (as it is here), this indicates we are overfitting and need to regularize our model. Since this is the case without the validation error finding a nadir and then increasing, early stopping will not help us here. So, our next step is to drive down this validation error with regularization and normalization. This will allow our model to generalize to the dataset's distribution well, rather than overfitting to the training data.

### The usual suspects
We have many options here, and rules of thumb are empirical at best. Let's do an ablation:
- Normalize the data (this was done already, so we'll try a run without.)
- Add batch norm in the network
- Add dropout in the network

Validation curve x3 with grey error fill.

Commentary. Did it help?

### Data augmentation

## Speed is the key

While doing all these tasks, your concern should also be speed of training for iteration time. I encountered the following bottlenecks and/or solutions:

- Don’t use python multiprocessing library for data augmentation: inter-process communication is too slow because it writes to disk, and you'll be moving a lot of data around. There are python libraries out there which purport to 
Use tfds!
tfds for data pipeline and dataset creation - speed is key
Balance work on cpu/gpu available resources



RandAugment!
If you’re determining whether a person is x or y, don’t blindly use augmentations that would obscure necessary info to you. Use common sense here.

Once you have a reasonable h-param range and decent performance, optuna is your friend, but NOT BEFORE. You may have bugs in your code, or the model may under/overfit without the right details as above.

Finally, add jax code to GitHub, add relevant snippets.
