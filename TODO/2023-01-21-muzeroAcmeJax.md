---
layout: distill
title: MuZero in Acme
description: An implementation of DeepMind's breakthrough algorithm, with improvements.
giscus_comments: true
tags: rl
date: 2023-01-21

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
  - name: Equations
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Interactive Plots
  - name: Layouts
  - name: Other Typography?

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

## Why MuZero?

Implemented Sampled-MuZero-Reanalyze with sample-efficiency improvements found here, in JAX, in DeepMind's Acme RL framework, synthesizing these implementations. The goal here was my own understanding, and to build it for my own and others' general usage, particularly given its outstanding performance in discrete action spaces. (Full understanding of two source implementations complete; I've just begun to combine them in JAX/Acme.)


Coming Soon!
