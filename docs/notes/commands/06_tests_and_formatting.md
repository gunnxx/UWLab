# 6. Tests and formatting

Two commands to run inside the container before a commit.

## Run the tests

```bash
python -m pytest source/uwlab_tasks/test -x
```

Any test file or directory can take the place of the folder. The `-x` flag stops at the first failure, which
is usually what you want while fixing something.

## Format the code

```bash
uwlab --format
```

This runs black and flake8 with the repository's settings, the same checks the pre-commit hooks apply. Run it
once more after fixing anything it reports, until it comes back clean.
