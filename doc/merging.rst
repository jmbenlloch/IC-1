How to merge approved PRs
=========================

Requirements
--------------

- A remote called ``upstream`` pointing at the central IC repository.

- A remote configured for the author of the PR.

- After the PR is approved, make sure that it is ready to be merged (asking the authors if they didn't get in touch with you).


Steps
-------

(Detailed execution instructions below)

#. Fetch ``upstream/master``.

#. Reset or rebase your master branch to ``upstream/master`` (make sure your local `master` and `upstream/master` are pointing to the same commit).

#. Fetch the branch of the approved PR.

#. Create and check out a local branch on top of the PR branch.

#. Make sure the branch of the approved PR is rebased onto ``upstream/master``. If not:

   * Rebase the branch onto ``upstream/master``. If there are conflicts, ask the author to resolve them, unless they are obvious.

   * Push to the branch of the PR and wait until the tests finish.

#. Checkout your local ``master``.

#. Merge the PR branch into your local ``master``, making sure that the merge commit conforms to our requirements. Here are the steps needed to make the merge happen:

   * Disallow fast forward merging: we want an explicit merge commit for each PR.

   * Edit the commit message so that it has the following format:

   <PR number>  <PR title>

   <PR url>

   [author: <author's id>]

   <PR description>  (This is usually the whole first comment in the PR)

   [reviewer: <approver's id>]

   <Reviewer comment> (comment the reviewer left on GitHub)


   * Set the reviewer as the author of merge commit.

#. Push the merge commit to ``upstream/master``.

#. Delete the local branch you created in step 4.


Detailed mechanics
--------------------

#. **CLI:** ``git fetch upstream master`` **Magit:** ``f o upstream RET master RET``

#. **CLI:** ``git rebase master`` **Magit:** ``r e master RET``

#. **CLI:** ``git fetch <author's remote> <PR branchname>`` **Magit:** ``f o <author's remote> RET <PR branchname> RET``.

#. **CLI:** ``git checkout -b <PR branchname> <author's remote>/<PR branchname>`` **Magit:** ``b c <author's remote> RET <PR branchname> RET``

#.

   * **CLI:** ``git rebase upstream/master`` **Magit:** ``r e upstream/master RET``

   * **CLI:** ``git push <author's remote> <PR branchname>`` **Magit:** ``P -f e <author's remote>/<PR branchname> RET``

#. **CLI:** ``git checkout master`` **Magit:** ``b b master RET``

#. **CLI:** ``git merge --edit --no-ff <PR branchname>`` **Magit:** ``m -n e <PR branchname> RET``

   <PR number>  <PR title>

   <PR url>

   [author: <author's id>]

   <PR description>  (This is usually the whole first comment in the PR)

   [reviewer: <approver's id>]

   <Reviewer comment> (comment the reviewer left on GitHub)

   ``C-c C-c``

   * **CLI:** ``git commit --amend --author=<reviewer's id>`` **Magit:** ``c -A <reviewer's id> RET w C-c C-c``

#. **CLI:** ``git push upstream master`` **Magit:** ``P -f e upstream/master``

#. **CLI:** ``git branch -d <PR branchname>`` **Magit:** ``b k <PR branchname>``

