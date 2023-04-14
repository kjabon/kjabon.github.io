---
layout: distill
title: Some Acme Speedbumps
description: Messing around with parallelization.
giscus_comments: true
date: 2023-03-22
tags: rl 
authors:
  - name: Kenneth Jabon

  

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:

  

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
Check out the companion [github repo](https://github.com/kjabon/AcmeGPUHelper).

## Make Acme work for you!



Parallelizing training of RL agents in Acme required a deeper understanding of the inner workings of Launchpad, DeepMind's framework for distributed platforms, and the intricacies of checkpointing, which after deep diving used quite a bit of code from Tensorflow, took some dives into the code/documentation rabbit hole. This should be enough of a tipoff for you to figure this stuff out yourself. I'll leave the details to the enterprising reader, but there are some things that were particularly obfuscated/necessary to progress.

## Acme uses Tensorflow!

I didn't understand at the outset that so much code from Tensorflow was allocating GPU memory behind the scenes, at the same time as the JAX code, in different processes, which is largely the result of things not being rewritten in JAX in Acme (no judgment here). Resolving cryptic errors, most of which ended up being these same GPU memory allocation issues, took a significant amount of rabbit hole digging and understanding the environment variables related to memory allocation for each library. Eventually I created a "GPU startup" file which solves all these issues without too much thought, which I have ported over to other projects. See [here](https://github.com/kjabon/AcmeGPUHelper/blob/main/gpu.py) for the code.

## Hindsight: Buy Homogenous set of GPUs
Furthermore, JAX specifically does not allow mapping of work to heterogeneous device arrays (different GPUs), which is quite a shame in my opinion, as I currently have a 3080 and a 3080 Ti, which must instead be used for different sets of tasks. The JAX authors (Google) are likely used to using hundreds or thousands of server GPUs of the same type, not to mention TPUs, so I'm not holding my breath for a fix. 

***

Check out more [blog](/blog/) postings on my RL projects!


