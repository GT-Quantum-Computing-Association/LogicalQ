# Contributing

`LogicalQ` is an open-source software with the core design principle of enabling easier and higher-fidelity research in quantum computation for everyone. As such, we welcome contributions of many forms. This file describes how you can contribute to `LogicalQ` and what guidelines we have in place. Any issues which are not outlined in this file are left to maintainer discretion.

## Installation

If you want to contribute code or general changes to the codebase, the preferred method is to clone `LogicalQ` locally, either in a branch (preferred for internal contributors from QCA) or forks (default for external contributors outside QCA), and then merge through a Pull Request. Branches should be named in all-lowercase with dashes (`-`) separating segments of the name, i.e. they must satisfy the regular expression `^[a-z-]+$`.

The main branch of `LogicalQ` (from which releases are built) can be cloned by running the command
```bash
git clone https://github.com/GT-Quantum-Computing-Association/LogicalQ.git
```

We strongly urge contributors (and users!) to use virtual environments when using `LogicalQ` because we do not always rely on the latest versions of all dependencies. A `virtualenv` stored in the `.logicalq` directory (for example) can be created by running the command
```bash
python -m virtualenv .logicalq
```

The package can then be installed by navigating inside the repository directory and running the command
```bash
python -m pip install -e .
```

## Documentation

Although it is not strictly necessary for all contributions, we request that contributions are accompanied with documentation. Pull requests may be deferred or denied due to lack of documentation, especially if user contributions are highly-nontrivial.

## Testing

Our test suite is a work-in-progress - we welcome contributions to these submodules. With the addition of any features, we require, at a minimum, a demonstration of its accuracy within some relevant usage scope, even if not exhaustive. This typically takes the form of a demo notebook (one which imports `LogicalQ`) or a fragment notebook (one which does not import `LogicalQ`). Test cases are the preferred approach to testing code.

## Development Cycle

The `main` branch is used as the main thread of features which are ready for the next official release (i.e. it is a "nightly" version). The `stable/*` branches are derived from `main` and used to generate official releases. It is only updated upon a release, potentially with minor exceptions.

The development cycle typically takes the following form:

1. Contributor opens Issue or discusses a potential contribution with a Maintainer.
2. Contributor starts a new branch or fork, appropriately-named, for editing.
3. Contributor ideally documents and tests all edits.
4. Contributor opens Pull Request once branch is ready to be merged and requests review from Maintainer.
    - In some cases, the Contibutor may choose to open a Draft Pull Request to discuss development prior to formal review.
5. Maintainer reviews code and provide any necessary feedback.
6. Contributor makes any necessary changes.
7. Maintainer eventually approves Pull Request and merges changes to `main`.
8. Prior to the next applicable release, Maintainer merges changes to `stable/*`.

## Use of AI Tools

`LogicalQ` was initiated as a means for contributors to learn quantum computation and quantum information, particularly quantum error mitigation, detection, and correction and its applications to quantum algorithms. We thus strongly discourage the use of AI in code, but cannot moderate against its use. If you use AI in any way when contributing to `LogicalQ`, please ensure that:

- You understand all generated code and underlying tools and methods in their entirety, such that it can be documented and fully explained during review or in the future.
- You explicitly identify and describe all uses of AI in each relevant contribution.
- Your use of AI does not violate any other licenses, guidelines, policies, or other conditions, within or outside of `LogicalQ`.

Any contribution which appears to be heavily-AI generated with little user input or review is subject to be rejected from merges and removed upon maintainer discretion.

