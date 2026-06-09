# Contributing to HCIPy

We welcome contributions of all kinds — bug reports, documentation improvements, new optical elements, and other enhancements.

## Getting started

- For a detailed walkthrough, see the [contributing guide](https://docs.hcipy.org/dev/development/contributing_guide.html).
- Report bugs or request features by [opening an issue](https://github.com/ehpor/hcipy/issues/new/choose).

## Code contributions

1. **Discuss first.** Open an issue to let us know what you'd like to work on so your effort isn't wasted.
2. **Create a feature branch** from `master`.
3. **Commit logically.** Keep commits focused on a single logical change.
4. **Push and open a pull request.** Link any relevant issues in the PR description.
5. **Review.** A maintainer will review your changes.

### Checklist for new code

- Follow the [coding style](https://docs.hcipy.org/dev/development/project_setup.html#coding-style).
- Add docstrings for all public classes and functions (numpydoc format).
- Add tests for new functionality. Coverage should not decrease without good reason.
- Ensure the full test suite passes: `pytest .`
- Make sure the documentation still builds: `cd doc && make html`

## Questions?

Open a discussion or issue on [GitHub](https://github.com/ehpor/hcipy).
