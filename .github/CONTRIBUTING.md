# Contribution guide

All contributions are welcome! :)

If you would like to contribute anything, fork the repo and open a Pull Request (PR).

## Recommended git workflow

The best way to ensure that git history is not jumbled up too much is to add an `upstream` remote:

```bash
$ git clone ...your_fastats_fork_url...
$ cd fastats
$ git remote add upstream https://github.com/fastats/fastats
```

Then you can fetch from `upstream` remote and create new features on your fork easily:

```bash
$ git fetch upstream
$ git checkout -b my_awesome_branch upstream/master
```


## Issues

Issues are turned off forever. We prefer Pull Requests for everything.

There's many reasons for this, [this gist][bad_issues] from Ryan Florence details them nicely.

If you have questions about using the library, please feel free to ask
questions on the [fastats mailing list](https://groups.google.com/forum/#!forum/fastats)


## Reporting bugs and requesting changes

- To report a bug, open a PR with a unittest that fails.
- To request an API change/new functionality, open a PR with a failing unittest showing your
  preferred API.
- To submit a fix, open a PR with passing unittests + doctests.

Simples :)


## Installing requirements for development

To ease dependency management, we rely on `setup.py` script to contain the
requirements.

Some of the tests require extra libraries that are not required for normal
installation.

One way to develop `fastats` code is to work in a virtual environment and
install `[dev]` requirements bundle:

```bash
$ pwd
/your/github/checkout/of/fastats
$ python3 -m pip install virtualenv --user
$ python3 -m virtualenv venv
$ . venv/bin/activate
$ pip install -e .[dev]
```

Such install will ensure that all requirements are met, and that the changes
to `fastats` code are immediately visible.

#### IDEs

Advanced IDEs, such as PyCharm, will allow you to create the virtualenv
using GUI and pointing the project interpreter at it.  All you have to do then
is fire up the terminal in the IDE, ensure you're in venv and run
`pip install -e .[dev]`.  This should enable things like
`right-click -> run py.test` etc.


#### windows

If you're on windows, the procedure should be analogous - except
`activate` is a script that can be called directly.

One problem that we've seen on windows is that `statsmodels` won't install
unless `numpy` is installed first.  The solution is to run `pip install numpy`
before `pip install -e .[dev]`.


## Code style

We tend to follow [PEP8][pep8] for Python code style, with a few exceptions:

- All `.py` files should begin with a blank line (blame @dwillmer for this).
- One-letter and upper-case variable names might be acceptable where they make sense, especially in
  linear algebra functions dealing with matrices.

### Before submitting your code for PR:

- Make sure all your `.py` files always end with a blank line (this is good practice,
  also for other file formats).
- Make sure your source files are `UTF-8` encoded.
- Make sure you have tests (test coverage is automatically enforced on every PR).

#### Tests

- Doctests should be minimal, and serve as API docs for the most common use cases.

  - For doctests, you should place this at the bottom of your module (followed by a blank line, ofc):

    ```python
    if __name__ == '__main__':
        import pytest
        pytest.main([__file__])
    ```
      
    This ensures that you can "run" every doctest in the module ad-hoc.
  
  - Note that some doctests require special setup / options that are provided by pytest.
    Because of this, they won't work on their own.
    
    See the `conftest.py` and `pytest.ini` files for more details.

- Unittests should be exhaustive and should reside in the `tests` directory.


#### Running tests

You can use [tox][tox] to run local tests for all supported Python versions and
various versions of libraries such as NumPy.

To run all tests in all environments you can simply execute:

```bash
$ tox
```

To install multiple Python versions so they are available to tox, such as
`python3.5`, `python3.6` etc you can use [pyenv][pyenv]:

```bash
# make sure to install pyenv first: https://github.com/pyenv/pyenv#installation
$ pyenv install 3.5.4
$ pyenv install 3.6.4
$ pyenv local 3.5.4 3.6.4
$ pyenv versions
  system
* 3.5.4 (set by fastats/.python-version)
* 3.6.4 (set by fastats/.python-version)
```

You can also install "in development" versions of Python such as `3.7.0a4`. 

While this is quite useful, it's also a bit cumbersome.  Thankfully we're
running all tests automatically on Travis CI.

Note that tox does not require pyenv on Travis CI, as various Python versions
come pre-installed out of the box.


[bad_issues]: https://gist.github.com/ryanflorence/8a62abea562ca2896dee
[pep8]: https://pep8.org/
[pyenv]: https://github.com/pyenv/pyenv
[tox]: https://tox.readthedocs.io/en/latest/
