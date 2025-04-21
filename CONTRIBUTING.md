### Contributing Guide

Please open a new issue or new pull request for bugs, feedback, or new features you would like to see.

See the instructions below for how to contribute to this repository using a forking workflow.

#### Forking Workflow
1. Create a personal fork of the `roman-wfi-reference-pipeline` repository by visiting its location on GitHub and clicking the `Fork` button.  This will create a copy of the `roman-wfi-reference-pipeline` repository under your personal GitHub account (hereby referred to as "personal fork").  Note that this only has to be done once.

2. Make a local copy of your personal fork by cloning the repository (e.g. `git clone https://github.com/username/roman-wfi-reference-pipeline.git`, found by clicking the green "clone or download" button.).  Note that, unless you explicitly delete your clone of the fork, this only has to be done once.

3. Ensure that the personal fork is pointing to the `upstream` `roman-wfi-reference-pipeline` repository with `git remote add upstream https://github.com/spacetelescope/roman-wfi-reference-pipeline.git` (or use the SSH version if you have your SSH keys set up).  Note that, unless you explicitly change the remote location of the repository, this only has to be done once.

4. Create a branch off of the `main` branch on the personal clone to develop software changes on. Branch names should be short but descriptive (e.g. `new-database-table` or `fix-ingest-algorithm`), and not too generic (e.g. `bug-fix`).  Consistent use of hyphens is encouraged.
    1. `git branch <branchname>`
    2. `git checkout <branchname>` - you can use this command to switch back and forth between existing branches.
    3. Perform local software changes using the nominal `git add`/`git commit -m` cycle:
       1. `git status` -  allows you to see which files have changed.
       2. `git add <new or changed files you want to commit>`
       3. `git commit -m 'Explanation of changes you've done with these files'`

5. Remember all changes must have appropriate test and documentation updates.

6. Push the branch to the GitHub repository for the personal fork with `git push origin <branchname>`.

7. In the `roman-wfi-reference-pipeline` repository, create a pull request for the recently pushed branch.  You will want to set the base fork pointing to `roman-wfi-reference-pipeline:main` and the `head` fork pointing to the branch on your personal fork (i.e. `username:branchname`).  Note that if the branch is still under development, you can use the GitHub "Draft" feature (under the "Reviewers" section) to tag the pull request as a draft. Not until the "Ready for review" button at the bottom of the pull request is explicitly pushed is the pull request 'mergeable'.

8. Assign the pull request a reviewer, selecting a maintainer of the `roman-wfi-reference-pipeline` repository.  They will review your pull request and either accept the request and merge, or ask for additional changes.

9. Iterate with your reviewer(s) on additional changes if necessary, addressing any comments on your pull request.  If changes are required, you may end up iterating over steps 4.iii and 5 several times while working with your reviewer.

10. Once the pull request has been accepted and merged, you can delete your local branch with `git branch -d <branchname>`.

#### Keeping your fork updated
If you wish to, you can keep a personal fork up-to-date with the `roman-wfi-reference-pipeline` repository by fetching and rebasing with the `upstream` remote, remember to update your github repo's main after doing this.
1. `git checkout main`
2. `git fetch upstream main`
3. `git rebase upstream/main`
4. `git push origin main`

Alternatively, you can use the `sync fork` button on the main page of your github fork.  Remember once synced you will need to pull your main down to your machine
1. `git checkout main`
2. `git fetch origin main`
3. `git pull origin main`

#### Collaborating on someone else's fork
Users can contribute to another user's personal fork by adding a `remote` that points to their fork and using the nominal forking workflow, e.g.:

1. `git remote add <username> <remote URL>`
2. `git fetch <username>`
3. `git checkout -b <branchname> <username>/<branchname>`
4. Make some changes (i.e. `add/commit` cycle)
5. `git push <username> <branchname>`