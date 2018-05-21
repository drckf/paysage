# Requesting changes and making pull requests

## Bugs

If you think you have found a bug:

1. Make sure that you have the most recent version of the `master` branch.
2. The `paysage` API is not stable yet. Check the git commit history
to make sure that the function you are using hasn't been updated recently.
3. If you still think you have found a bug, then submit an issue. Tell us:
- What OS / version of python are you using?
- Does the issue occur using both the python and pytorch backends?
- Provide an error message with a full stack trace.
- Try to provide a code example that reproduces the error.

## Issues

You *must* submit an issue describing any major changes in order to have a
pull request approved.

The issue must explain, in detail, *what* you want to do, *why* you want to do
it, and *how* it will improve the results, efficiency, or clarity of the code.

If you want to implement a method from a paper, you must provide a citation.

*Be prepared for discussion!*

## Pull Requests

Pull requests with major changes will not be approved unless the associated
issue has also been approved.

You must write tests for any new functions. If you claim in the issue that
the change will provide better results, you must provide an example that
demonstrates better results than the current version of the code.

Please follow the docstring conventions used throughout the rest of the code.

Please try to name all classes/functions/variables with names that will be
immediately interpretable to anybody the first time they read it.

Tests are run automatically using CircleCI. Your PR will not be accepted unless
all tests pass.

A linter is run automatically using Landscape.io. Your PR will not be accepted
if there is a significant decrease in code quality metrics.

*Be prepared for more discussion!*

### Pull Requests and the Backends

We support 3 different backends that perform the numerical computations.

1) {"backend": "python", "processor": "cpu"}
2) {"backend": "pytorch", "processor": "cpu"}
3) {"backend": "pytorch", "processor": "gpu"}

If you make a change that affects one of the examples, you must post the output
created by that example from *all three backends* to verify that they all run
without errors and that they produce similar results.
