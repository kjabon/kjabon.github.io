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


The main question of this post is: what can we do with imitation learning that we *can't* do with plain RL? Well, there are two main situations in which it can help us out. First, if we're unable to adequately define a reward function for the task we have in mind, or if the reward function is known, but too sparse to learn efficiently (in which case we have failed to adequately define a reward function, so this is really a special case of the first).

***

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/hx_bgoTF7bs?start=88" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe></center>
<div class="caption">
In this video, we see RL-trained agents successfully navigating environments, with somewhat goofy behavior. More reference links in that video's description.
</div>
***
One symptom of this, as seen in the above video, may be that despite being able to learn a policy which technically achieves the goal, like swingup in acrobot or navigation for humanoid robots, these learned policies may be idiosyncratic. Perhaps they are too jerky, or do things that look obviously goofy or energy-inefficient. Defining rewards to tamp down on undesired behaviors can become a tedious and never-ending game of whack-a-mole. For example, the "arm pumping" in the above video is serving some kind of counterbalancing purpose, but we know intuitively that this behavior is not ideal. Yet, it's not clear how one would go about defining a reward function to tamp down on it.

Another example: suppose I was training a robotic hand to grasp and sort objects of various size, material, and shape, certainly I would be able to specify distance-based measures of reward: how far is the object from the target? But the reward for a "stable grasp" of an object is not so easily mathematically defined. 

<iframe style='height:250px;width:100%;border:none' frameborder="0" allow="autoplay; fullscreen" allowfullscreen="" src="https://player.vimeo.com/video/365132002?h=689a8eff63&amp;autopause=0&amp;autoplay=1&amp;background=1&amp;loop=1&amp;muted=1&amp;playsinline=1&amp;transparent=1" data-ready="true"></iframe>
<div class="caption">
A robotic hand from OpenAI's "Solving Rubik's Cube with a Robot Hand." 
</div>


Certainly we could get into the realm of feature engineering... perhaps I could define a stable grasp to be three points of contact at which I am applying a normal force, the sum of these forces being as close to zero as possible, all while relying on friction and torque as little as possible. Add in a feature for whether the center of gravity is below all of my contact points, and we're looking pretty good. But this sort of feature engineering tends to be object class-specific. The features for grasping a glass of water will look very different from those for grasping a raspberry, a sheet of paper, a block of agar, or a chicken egg. This could also end up needing quite a few sensors than we have available, rather than the usual image and/or robot state observation. (This is [not to say] using 

So, instead, we could have people directly controlling the robot to perform the desired tasks, record the trajectories including control actions by the people, and do imitation learning on that. For the humanoid example, perhaps we could have people navigate obstacle courses while wearing motion capture gear. 

There are of course other methods which we could use on this problem, such as RLHF, and more advanced methods which build upon what's discussed in this post (see google soccer paper), but let's stick to imitation learning for this post. 

If you're unfamiliar with Acrobot, it's a classic control problem in which the "elbow" joint can be actuated, the "shoulder" joint swings freely, and the goal is (typically) to swing to a vertical, upwards position, starting from a vertical downwards position. Sometimes the goal is to also maintain this upright position, which we'll leave for another time.

This may come as a shock, but there exist pretty good solutions to this problem that don't involve deep reinforcement learning. (Brief ILQR explanation.)

The basic question of this post is: can we learn a classical ILQR controller's policy for solving acrobot-swingup with imitation learning?

<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/ilqr_swingup.gif" title="ilqr" class="img-fluid rounded" caption="Acrobot swingup with classical ILQR controller." %} 
</div>
 <div class="col">
</div>
</div>


Replace all plots with matplotlib/plotly

We don't use the standard gym/mujoco environments for this post, because they don't lend themselves as straightforwardly to using classical controllers via access to their dynamics and kinematics (it is possible, I just haven't implemented it).

Basically, we want to perform the following tasks.

Do RL with D4PG on the task, no imitation. This is our baseline. 

{% include figure.html path="assets/img/acrobot_d4pg_10M.png" title="D4PG" class="img-fluid rounded" caption="Episode return vs. time steps; acrobot run with D4PG." %} 


<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/d4pg_swingup.gif" title="D4PG" class="img-fluid rounded" caption="Acrobot run after 10M training steps with D4PG. " %} 
</div>
 <div class="col">
</div>
</div>

During training, upon reaching the top, the episode would end and the environment would reset. In the above figure, after turning this reset off, the policy is still able to swing back up to position, and surprisingly maintains its position.

Collect trajectories from an ilqr controller for swingup, stopping and resetting above the threshold line.
These should be noisy.
Now, we also train PWIL on ILQR data for 1M time steps. Let's take a look at the results...

{% include figure.html path="assets/img/acrobot_pwil_1M.png" title="pwil" class="img-fluid rounded" caption="Episode return vs time steps; acrobot run for 1M timesteps with PWIL." %} 

We can see that PWIL works! By learning to imitate the classical controller's actions given the state, this learned controller shows signs of life in its training curve. Because it isn't training to maximize rewards directly (the loss function doesn't even contain the environment reward), as expected, the final average return isn't quite so high as something that's optimizing for it.

Let's see what happens when we visualize a rollout of the PWIL-learned policy.



<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/pwil_swingup.gif" title="pwilSwing" class="img-fluid rounded" caption="Rollout of PWIL policy, learned from expert ILQR demonstrations." %} 
</div>
 <div class="col">
</div>
</div>

Not exactly the same as the PWIL controller, but pretty close! Keep in mind that this was trained with two sources of noise: the ILQR controller itself had its actions and observations perturbed, and PWIL additionally perturbs actions.

Now at this point it's worth discussing the reward and termination scheme. The environment gives a reward of +10 if the end of the pendulum is within some radius of the target (i.e. where the pendulum is completely vertical and pointing up; the angle of the first joint is $$\pi$$ and the angle of the second joint is 0). The radius threshold is somewhat large: 0.2 (compare to the length of the pendulum: 0.5). In all cases it will receive a penalty proportional to its distance from the target, and penalties proportional to the absolute position and velocity, to avoid wild spinning.

Finally, the environment resets either after 10 seconds (5000 time steps each 0.002s long), or if the end of the pendulum is both within the aforementioned radius from the target, and its y position is within 0.05 of the target. This y condition is most of the story, but the radius necessitates it to be nearer to the center.

Do you see the potential issue? In the D4PG gif above, we see some interesting behavior suggesting an incorrect environment setup.

The first loop goes to the target as hoped for; the second loop maintains position away from the target, below the y position. That's strange... if we look more closely at the first loop, and notice that the position of the middle joint does **not** go to the top, we can see that the reset condition wasn't actually reached! And indeed, looking at the episode length chart from training the D4PG agent, it is more or less always at the full 5000 time steps. Clearly the agent has learned to **not** terminate the episode, but get as many +10's as it can while staying inside the target radius threshold, but below the y position threshold.

{% include figure.html path="assets/img/d4pg_episode_length.png" title="d4pgLen" class="img-fluid rounded" caption="The D4PG agent learns to not terminate episodes; the episode length is 5000." %} 

This is a perfect example of why visualization is so important.

So, what to do? Well, let's first fix the reward, by adding the y condition to be able to get a +10. Let's re-train both D4PG and PWIL with the fix. This should tamp down on the strange "hold near target" behavior, and the max return should be near 0. PWIL doesn't use this reward to train, so we're merely reflecting how the training on the PWIL reward affects progression towards the environment reward. We expect the D4PG curve to end with a higher return, since it's optimizing for the reward. 

[Show D4PG and PWIL training curves].

However, because of the termination condition, it is possible that both policies without resets will get up to the top and then start exhibiting strange behavior.

[Show gifs]

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
