---
layout: distill
title: Imitation Learning with PWIL
description: An exercise on the Acrobot Swingup task
giscus_comments: true
date: 2023-07-18
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
Negative IP, takeaways.

Basically, we want to perform the following tasks.

Do RL with D4PG on the task, no imitation. This is our baseline.

Gif of the result.

Collect trajectories from an ilqr controller for swingup, stopping and resetting above the threshold line.
These should be noisy.

Learn PWIL from these trajectories. Stopping and resetting when threshold line hit.


