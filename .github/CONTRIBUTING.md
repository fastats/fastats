# Contribution guide

All contributions are welcome! :)

If you would like to contribute anything, fork the repo and open a Pull Request (PR).

## Recommended git workflow

The best way to ensure that git history is not jumbled up too much is to add an `upstream` remote:

```
$ git clone ...your_fastats_fork_url...
$ cd fastats
$ git remote add upstream https://github.com/fastats/fastats
```

Then you can fetch from `upstream` remote and create new features on your fork easily:

```
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

- Unittests should be exhaustive and should reside in the `tests` directory.


[bad_issues]: https://gist.github.com/ryanflorence/8a62abea562ca2896dee
[pep8]: https://pep8.org/
