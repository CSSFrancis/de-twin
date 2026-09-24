Filing Change Log Entries
=========================

de-twin uses `towncrier <https://towncrier.readthedocs.io/>`_ to manage its
changelog. When you open a pull request that should appear in the next release
notes, add a short news **fragment file** to this directory as part of that PR.

Naming convention
-----------------

Each fragment is a plain ``.rst`` file named::

    {PR_number}.{type}.rst

where ``{PR_number}`` is the GitHub pull-request number and ``{type}`` is one
of the types below.

If a change has no natural PR number (e.g. work batched on a long-lived
feature branch), name the file ``+{slug}.{type}.rst``. The leading ``+`` marks
it as an orphan fragment, so towncrier omits the (otherwise broken) PR link.

=================  ==============================================================
Type               Use when …
=================  ==============================================================
``api_change``     Existing behaviour changed in a way a user has to act on:
                   a signature, a default, or a unit.
``new_feature``    A user-visible capability has been added.
``bugfix``         A bug has been fixed.
``deprecation``    Something is deprecated and will be removed in a future release.
``removal``        A previously deprecated API has been removed.
``doc``            Documentation improved without any code change.
``maintenance``    Internal / infrastructure change invisible to users.
=================  ==============================================================

Content guidelines
------------------

Write for someone upgrading: say what changed and what they have to do, in one
or two sentences of plain reStructuredText. Preview the next release notes with::

    uvx towncrier build --draft --version X.Y.Z
