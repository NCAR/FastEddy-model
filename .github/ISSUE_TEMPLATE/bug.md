---
name: Bug report
about: Fix something that's not working
title: 'Bugfix: '
labels: 'type: bug'
assignees: ''

---

*Replace italics below with details for this issue.*

## Describe the Problem ##
*Provide a clear and concise description of the bug here.*

### Expected Behavior ###
*Provide a clear and concise description of what you expected to happen here.*

### Environment ###
Describe your runtime environment:
*1. Machine: (e.g. HPC name, Linux Workstation, Mac Laptop)*
*2. OS: (e.g. RedHat Linux, MacOS)*
*3. Software version number(s)*

### To Reproduce ###
Describe the steps to reproduce the behavior:
*1. Go to '...'*
*2. Click on '....'*
*3. Scroll down to '....'*
*4. See error*

### Relevant Deadlines ###
*List relevant project deadlines here or state NONE.*

## Define the Metadata ##

### Assignee ###
- [ ] Assign this Task to the appropriate person

### Labels ###
- [ ] Select **pillar**
- [ ] Select **priority**
- [ ] Select **requestor(s)**

## Bugfix Checklist (DETAILS TBD - THE INFO BELOW IS NOT YET DECIDED UPON) ##
- [ ] Complete the issue definition above.
- [ ] Fork this repository or create a branch of **main_\<Version>**.
Branch name: `bugfix_<Issue Number>_main_<Version>_<Description>`
- [ ] Fix the bug and test your changes.
- [ ] Add/update log messages for easier debugging.
- [ ] Add/update unit tests.
- [ ] Add/update documentation.
- [ ] Push local changes to GitHub.
- [ ] Submit a pull request to merge into **main_\<Version>**.
Pull request: `bugfix <Issue Number> main_<Version> <Description>`
- [ ] Define the pull request metadata, as permissions allow.
Select: **Reviewer(s)**
- [ ] Iterate until the reviewer(s) accept and merge your changes.
- [ ] Delete your fork or branch.
- [ ] Complete the steps above to fix the bug on the **develop** branch.
Branch name:  `bugfix_<Issue Number>_develop_<Description>`
Pull request: `bugfix <Issue Number> develop <Description>`
- [ ] Close this issue.

