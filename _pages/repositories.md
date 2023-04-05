---
layout: page
permalink: /repositories/
title: repos
description: See the respective <a href= 'https://kjabon.github.io/blog/'>blog posts</a> for an introductory explanation. I highly recommend setting up an instance of coachRL to form the good habits you never knew you could. Fork my code at your own risk :)
#Edit the `_data/repositories.yml` and change the `github_users` and `github_repos` lists to include your own GitHub profile and repositories.
nav: true
nav_order: 3
---

## GitHub users

{% if site.data.repositories.github_users %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.html username=user %}
  {% endfor %}
</div>
{% endif %}

---

## GitHub Repositories

{% if site.data.repositories.github_repos %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.html repository=repo %}
  {% endfor %}
</div>
{% endif %}
