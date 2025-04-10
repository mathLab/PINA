Contributing to PINA
=====================

First off, thanks for taking the time to contribute to **PINA**! 🎉 Your help makes the project better for everyone. This document outlines the process for contributing, reporting issues, suggesting features, and submitting pull requests.

Table of Contents
------------------------

1. `How to Contribute`_
2. `Reporting Bugs`_
3. `Suggesting Enhancements`_
4. `Pull Request Process`_
5. `Code Style & Guidelines`_
6. `Community Standards`_

How to Contribute
------------------------

You can contribute in several ways:

- Reporting bugs
- Suggesting features/enhancements
- Submitting fixes or improvements via Pull Requests (PRs)
- Improving documentation

We encourage all contributions, big or small!

Reporting Bugs
------------------------

If you find a bug, please open an `issue <https://github.com/mathLab/PINA/issues>`_ and include:

- A clear and descriptive title
- Steps to reproduce the problem
- What you expected to happen
- What actually happened
- Any relevant logs, screenshots, or error messages
- Environment info (OS, Python version, dependencies, etc.)

Suggesting Enhancements
------------------------

We welcome new ideas! If you have an idea to improve PINA:

1. Check the `issue tracker <https://github.com/mathLab/PINA/issues>`_ or the `discussions <https://github.com/mathLab/PINA/discussions>`_ to see if someone has already suggested it.
2. If not, open a new issue describing:
   - The enhancement you'd like
   - Why it would be useful
   - Any ideas on how to implement it (optional but helpful)
3. If you are not sure about (something of) the enhancement, we suggest opening a discussion to collaborate on it with the PINA community.

Pull Request Process
------------------------

Before submitting a PR:

1. Ensure there’s an open issue related to your contribution (or create one).
2. `Fork <https://help.github.com/articles/fork-a-repo>`_ the repository and create a new branch from ``master``:

   .. code-block:: bash

      git checkout -b feature/my-feature

3. Make your changes:
   - Write clear, concise, and well-documented code
   - Add or update tests where appropriate
   - Update documentation if necessary
4. Verify your changes by running tests:

   .. code-block:: bash

      pytest

5. Properly format your code. If you want to save time, simply run:

   .. code-block:: bash

      bash code_formatter.sh

7. Submit a `pull request <https://help.github.com/articles/creating-a-pull-request>`_ with a clear explanation of your changes and reference the related issue if applicable.

Pull Request Checklist

1. Code follows the project’s style guidelines
2. Tests have been added or updated
3. Documentation has been updated if necessary
4. Pull request is linked to an open issue (if applicable)

Code Style & Guidelines
------------------------

- Follow PEP8 for Python code.
- Use descriptive commit messages (e.g. ``Fix parser crash on empty input``).
- Write clear docstrings for public classes, methods, and functions.
- Keep functions small and focused; do one thing and do it well.

Community Standards
------------------------

By participating in this project, you agree to abide by our Code of Conduct. We are committed to maintaining a welcoming and inclusive community.
