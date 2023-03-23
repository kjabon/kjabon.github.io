---
layout: distill
title: Organizing Your Day with CoachRL
description: Forming good habits is the actual, lasting way to change your life.
giscus_comments: true
date: 2023-01-16
tags: habits rl coachrl
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
  - name: To Summarize the Previous Post
  - name: Enter- CoachRL
  - name: Automation of Fiddling and Edits
  

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


## To Summarize the Previous [Post](/blog/2023/distill/)
Suppose you want to form a lot of habits and do it quickly. You need to get organized. You need to track everything, you need to reward yourself for accomplishing tasks/habits, and you need to do all of this in a well-balanced, proportional way.
But that's hard. 

## Enter- CoachRL

No more tedium of tracking and decision-making. Automate everything to oblivion! Just a "personal coach" that tells you what to do and gives you a treat for doing it.


So what does a typical day look like?


First thing over coffee, you open your spreadsheet on Google Docs to find out what you're going to be doing today. The row for today has already been filled in, based on the moving average of your "performance" for each habit.

{% include figure.html path="assets/img/habitSheet.jpg" title="example" class="img-fluid rounded z-depth-1" %} 
<div class="caption">
In red: today's example activities, suggested by CoachRL. In blue: associated moving average of habit performances.
</div>


Suppose you're trying to work on a personal project for an hour a day. You were a bit overzealous last weekend and worked on it all weekend, sending your average over the one hour mark. The program will probabilistically output actions from a predefined, categorical distribution, so that over the next few days, your average will come down back to the goal. (You can see each habit can output whatever set of values makes sense - like 1-3 hours worked, or 0-4 "chunks of 30 minutes" practicing piano).
If you differ from this suggestion, at the end of the day you should input the actual behavior, so tomorrow has accurate data for its suggestions.

## Automation of Fiddling and Edits

Typically I like to review my suggestions up front to get a handle on what I'll be doing today, and make sure they make sense in aggregate. (Over time as I've smoothed out the bugs, I simply trust the output is correct and carry on with the following step.) E.g., I'd rather not run, go to the gym, play basketball, and do yoga all in one day - one exercise a day, at most, please! If I notice I'm making this kind of correction a lot, I'll add it to the code which fills out the row with some simple logic. This further reduces my daily work and mental load. Ahh, the power of automation.


Now that I have today's numbers in front of me, I open my note taking app on my iPad, and copy my daily schedule template over. I erase the things that have no bearing on today, alter today's work time, and rearrange things according to decreasing difficulty (see [eating the frog](https://todoist.com/productivity-methods/eat-the-frog)), and usually put my exercise in the middle of the day, just before lunch - the perfect midday break from my work day. The template is pre-organized so all of this takes just a minute or two with minimal fiddling. You can see a truncated example of this below. I also like to space out my work with errands/habits, so I'm not forced to work on something for 4 hours straight and get nowhere out of fatigue - but I leave myself the option to keep working indefinitely if I'm in the zone, simply crossing off future work items.

<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/ipadTemplate.jpg" title="example" class="img-fluid rounded z-depth-1" %}
</div>
 <div class="col">
</div>
</div>
<div class="caption">
Example daily template for a to-do list.
</div>


Presto, you're ready to start your day! Perhaps you have a morning routine involving water, spritz of lemon, and a cold shower. Then you can dig straight into your daily habits as the opportunity arises. Time before or after work? Go for a run or plug away at your novel. 

Finally, for each item you complete - say 30 minutes of work, practicing piano for 30 minutes -  you request a reward from Telegram, which will tell you what you've earned for your hard work. A picture of this interface is in the figure below. Rewards are key, and keeps you trucking through the day. 
Pick a reward you can encapsulate and treat yourself with at will, that you find intrinsically rewarding. A YouTube jaunt, good book, or gaming session will do. Enjoy your spoils, and repeat! Add more habits as you feel you can handle the extra load. And that's all there is to it!

<div class="row">
<div class="col">
</div>
<div class="col-8">
{% include figure.html path="assets/img/telegramCoach.jpg" title="example" class="img-fluid rounded z-depth-1" %}
</div>
 <div class="col">
</div>
</div>
<div class="caption">
Telegram interface for getting rewards from CoachRL.
</div>

***

See more for how and why to tune rewards, and snowballing good behaviors, in this previous blog [post](/blog/2023/distill/). 

For the technical details of how the backend works, see the next [post](/blog/2023/CoachRLDetails/).

If there's interest (via Github stars or comments below), I can make a post about setting it up for yourself.




