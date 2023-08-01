---
layout: distill
title: Imitation Learning with PWIL
description: An exercise on the Acrobot Swingup task
giscus_comments: true
date: 2022-07-18
tags: rl 
authors:
  - name: Kenneth Jabon

  
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
  iframe {
    display: block;
    border-style:none;
  }
---


## Imitation learning

What can we do with imitation learning that we can't do with plain RL? Well, there some situations in which it can help us out. 

- We're unable to adequately define a reward function for the task we have in mind.
- The reward function is known, but too sparse to learn efficiently.
- The reward function is adequate to learn to accomplish the goal, but it accomplishes it in an undesirable way.

We're going to focus on the last situation for most of this post. 

## What do we have without IL?

The learned policies may be idiosyncratic. Perhaps they are too jerky, or do things that look obviously goofy or energy-inefficient. For instance, take a look at the RL-trained humanoid running in the video below. You should get the idea after 30s or so.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/hx_bgoTF7bs?start=88" frameborder="0" allowfullscreen></iframe>​
</center>
<div class="caption">
In this video, we see RL-trained agents successfully navigating environments, with somewhat goofy behavior. More reference links in that video's description.
</div>

The "arm pumping" in the above video is serving some kind of counterbalancing purpose, but we know intuitively that this behavior is not ideal. Suppose we want that not to happen: defining rewards to tamp down on undesired behaviors can become a tedious and never-ending game of whack-a-mole. 

Another example: suppose I was training a robotic hand to grasp and sort objects of various size, material, and shape, certainly I would be able to specify distance-based measures of reward: how far is the object from the target? But the reward for a "stable grasp" of an object is not so easily mathematically defined.

<iframe style='height:300px;width:100%;border:none' frameborder="0" allow="autoplay; fullscreen" allowfullscreen="" src="https://player.vimeo.com/video/365132002?h=689a8eff63&amp;autopause=0&amp;autoplay=1&amp;background=1&amp;loop=1&amp;muted=1&amp;playsinline=1&amp;transparent=1" data-ready="true"></iframe>
<div class="caption">
A robotic hand from OpenAI's "Solving Rubik's Cube with a Robot Hand." 
</div>

### Feature engineering?

Certainly we could get into the realm of feature engineering... perhaps I could define a stable grasp to be three points of contact at which I am applying a normal force, the sum of these forces being as close to zero as possible, all while relying on friction and torque as little as possible. Add in a feature for whether the center of gravity is below all of my contact points, and we're looking pretty good. 

However, this sort of manual feature engineering tends to be object class-specific. The features for grasping a glass of water will look very different from those for grasping a raspberry, a sheet of paper, a block of agar, or a chicken egg. This could also end up needing quite a few more sensors than we have available, rather than the usual image and/or robot state observation. Currently access to this kind of hardware shouldn't be assumed, however, including force sensors in our observation will hopefully become the norm.

<img src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-021-27261-0/MediaObjects/41467_2021_27261_Fig1_HTML.png?as=webp" height="400" alt="hand with force sensors" />
<div class="caption">
An example of a robot hand with force sensors in the fingertips. [https://www.nature.com/articles/s41467-021-27261-0]
</div>

**make that link a footnote**

## Avoid feature and reward engineering!

So, instead, we could have people moving the robot with a handheld controller to perform the desired tasks, record the camera images and control actions, and do imitation learning on these "expert demonstrations." For the humanoid example, perhaps we could have people navigate obstacle courses while wearing motion capture gear. 

There are of course other methods which we could use on this problem, such as [RLHF](https://arxiv.org/abs/1706.03741), and more [advanced methods](https://arxiv.org/abs/2105.12196) which build upon what's discussed here, but let's stick to imitation learning for this post. 

## Solving the Acrobot environment

With all this motivation out of the way, let's get going.

If you're unfamiliar with Acrobot, it's a classic control problem in which the "elbow" joint can be actuated, the "shoulder" joint swings freely, and the goal is (typically) to swing to a vertical, upwards position, starting from a vertical downwards position. Sometimes the goal is to also maintain this upright position, which we'll leave for another time.

This may come as a shock, but there exist pretty good solutions to this problem that don't involve deep reinforcement learning. ILQR, or the Iterative Linear Quadratic Regulator Algorithm, is among the best. See [here](https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/) for a good explanation. Qualitatively, one thing it seems to be good at is swinging up in a natural, expected fashion. In a real robot, we want to avoid high torques, high velocities, and high jerk values, to avoid wear on the robot and stay within operating limits.

The basic exercise of this post is: can we learn a classical ILQR controller's policy for solving acrobot-swingup with imitation learning?

<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/ilqr_swingup.gif" title="ilqr" class="img-fluid rounded" caption="Acrobot swingup with classical ILQR controller." %} 
</div>
 <div class="col">
</div>
</div>




We don't use the standard gym/mujoco environments for this post, because they don't lend themselves as straightforwardly to using classical controllers via access to their dynamics and kinematics (it is possible, I just haven't implemented it). Instead we use [this](https://github.com/dfki-ric-underactuated-lab/double_pendulum) repo.

Before imitation learning, we'll get a baseline with reinforcement learning.

## D4PG Baseline


<iframe src="{{ '/assets/plotly/D4PG_train_curve.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe> 
<div class="caption">
Acrobot training data with D4PG.
</div>

<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/d4pg_swingup.gif" title="D4PG" class="img-fluid rounded" caption="Acrobot run after 10M training steps with D4PG." %} 
</div>
 <div class="col">
</div>
</div>



This uses the D4PG algorithm, and is straightforward as far as RL goes. No tricks here. Before getting into a discussion of results, let's get some results with imitation learning as well.

## Imitation Learning with PWIL

First, we collect trajectories from an ILQR controller for swingup. The environment terminates and resets once the top is reached within some tolerance. Taking inspiration from [DART](https://arxiv.org/abs/1703.09327), since we have the option, we inject the ILQR controller's actions with noise. We use [envlogger](https://github.com/deepmind/envlogger) to record trajectories to a Tensorflow dataset. Another option would be to store to Reverb with the same library, if you're more familiar with that.

Next, we train PWIL on these collected trajectories for 1M time steps. Let's take a look at the results...



<iframe src="{{ '/assets/plotly/pwil_pre-fix.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe> 
<div class="caption">
Acrobot run for 1M timesteps with PWIL.
</div>

By learning to imitate the classical controller's actions given the state, this learned controller shows signs of life in its training curve. Because it isn't training to maximize rewards directly (the loss function doesn't even contain the environment reward), as expected, the final average return isn't quite so high as something that's optimizing for it.

Let's visualize a rollout of the PWIL-learned policy.



<div class="row">
<div class="col">
{% include figure.html path="assets/img/ilqr_swingup.gif" title="ilqr" class="img-fluid rounded"  %} 
</div>
<div class="col">
{% include figure.html path="assets/img/pwil_swingup.gif" title="pwilSwing" class="img-fluid rounded"  %} 
</div>
</div>
<div class="caption">
Left: Acrobot swingup with classical ILQR controller. Right: Rollout of PWIL policy, learned from expert ILQR demonstrations.
</div>

It doesn't mimic the ILQR controller perfectly; the torque actions are not so smooth in time, and the trajectory is not exactly the same. This was trained with two sources of noise: the ILQR controller itself had its actions and observations perturbed, and PWIL additionally perturbs actions. Because of this, the ILQR example may not be perfectly representative of that algorithms behavior. This said, the performance is surprisingly good; training on more demonstrations for more timesteps may mimic the original controller even better.

## Discussion

Now at this point it's worth discussing the reward and termination scheme. 

### Rewards

The environment gives a reward of +10 if the end of the pendulum is within some radius of the target (i.e. where the pendulum is completely vertical and pointing up; the angle of the first joint is $$\pi$$ and the angle of the second joint is 0). The radius threshold is somewhat large: 0.2 (compare to the length of the pendulum: 0.5). In all cases it will receive a penalty proportional to its distance from the target, and penalties proportional to the absolute position and velocity, to avoid wild spinning.

### Termination / reset
Finally, the environment resets either after 10 seconds (5000 time steps each 0.002s long), or if the end of the pendulum is both within the aforementioned radius from the target, and its y position is within 0.05 of the target. This y condition is most of the story, but the radius necessitates it to be nearer to the center.

There is a potential discrepancy in this setup: do you see it? Let's take another look at the D4PG rollout.

<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/d4pg_swingup.gif" title="D4PG" class="img-fluid rounded" caption="Acrobot run after 10M training steps with D4PG (reinserted here for convenience)." %} 
</div>
 <div class="col">
</div>
</div>


The first loop goes to the target as hoped for; the second loop maintains position some distance from the target, below the y position. That's strange... if we look more closely at the first loop, and notice that the position of the middle joint does **not** go to the top, we can see that the termination condition wasn't actually reached! And indeed, looking at the episode length chart from training the D4PG agent, it is more or less always at the full 5000 time steps. Clearly the agent has learned to **not** terminate the episode, but get as many +10's as it can while staying inside the target radius threshold, but below the y position threshold.


<iframe src="{{ '/assets/plotly/d4pg_length.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe> 
<div class="caption">
The D4PG agent learns to not terminate episodes; the episode length is (usually) 5000.
</div>

This is a perfect example of why visualization is so important.

## What to do?
Let's simplify the reward, so there's no conditional +10. Let's re-train both D4PG and PWIL with the fix. This should tamp down on the  "hold near target" behavior, and the max return should be near 0. PWIL doesn't use this reward to train, so we're merely reflecting how the training on the PWIL reward affects progression towards the environment reward. We still expect the D4PG curve to end with a higher return, since it's optimizing for the reward, however it should converge on some value just below 0.


<iframe src="{{ '/assets/plotly/D4PG_train_curve_reward_fix.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe> 
<div class="caption">
Acrobot run after 10M training steps with D4PG (removed 10+ reward; only negative rewards).
</div>

<iframe src="{{ '/assets/plotly/pwil_train_curve_reward_fix.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe> 
<div class="caption">
Acrobot run after 1M training steps with PWIL (removed 10+ reward; only negative rewards).
</div>

My take on these results is that while D4PG appears to have an edge on returns, PWIL is not far behind, and may have caught up with further training. Notice the "pockets" of results higher than -0.5 return.

<div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/850389032?h=dac29fa59b&autoplay=1&loop=1&title=0&byline=0&portrait=0" style="position:absolute;top:0;left:0;width:100%;height:100%;" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

<div style="padding:100% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/850390753?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen style="position:absolute;top:0;left:0;width:100%;height:100%;" title="d4pg_swingup_fix (Copy)"></iframe></div><script src="https://player.vimeo.com/api/player.js"></script>

So what exactly is the benefit of imitation learning? For the most part, D4PG seemed to win out here.



[Robot hand pic]

If human demonstrations are provided, we could then learn from that experience with much less sparse rewards than the end result. That intermediary policy could then be fine-tuned with RL on the sparse reward. If the robot hand has an approximately good policy, we can then improve upon it without so much random exploration.

Can we fine tune on this result with RL? Let's give it a shot.

[D4PG fine tuned from PWIL]

One last topic of discussion. Is the policy enacted by the ILQR controller even learnable by the neural network? In theory it should be, but notice how the ILQR controller swings back and forth several times before building up the momentum to make a final attack. Many of the states in this initial swing-up sequence are pretty similar, and maybe the corresponding actions are not-so-similar. I'm not going to try this here, but if this is the case, we could instead try using the history as the input to our policy instead of its current state to try to resolve this issue. This could either involve frame-stacking, an RNN/LSTM policy and value function, or perhaps even an attention mechanism. 

An easy thing to check is whether the D4PG policy, already represented by the same neural network, is learnable by PWIL. 
Do some roll-outs (they'll probably all be pretty similar; check this first). Learn them with PWIL. 

[Show the curve.]

We expect this to perform quite a bit better.



Discussion/ conclusions, possible applications. Negative IP, takeaways.

**Replace all plots with matplotlib/plotly**
