# Code Style

## Docstrings

- Use Google-style docstrings with `Args:` and `Returns:` sections.
- All classes and methods must have docstrings.
- Constructors use `"""Constructor."""` as their docstring.
- For methods that return `None`, skip the `Returns:` part.

## Abstract Classes

- Use `raise NotImplementedError("Subclasses must implement the <method>() method.")` for abstract methods. Do not use `ABC`/`abstractmethod`.

## Logging

- Use `logger = logging.getLogger(__name__)` at module level. Do not use `print` for logging.

## Type Hints

- All function signatures must have type hints for parameters and return types.
- As much as possible, use native type hints but if it does not exist, then use `typing` module and use `from typing import ...`.
- For methods that return `None`, skip the type hints.

## Naming

- Private attributes and methods are prefixed with `_`.
- Class names use PascalCase, methods use snake_case.

## Imports

- Group imports in order: stdlib, third-party, local. Separated by blank lines.

## Code Organization

- Use section comment banners to group related methods within a class:
  ```python
  # =============
  # Section Title
  # =============
  ```
- The public methods should always be grouped at the top, after the constructor if exists.
- For private methods, try to group them based on their functionality. If there is only one private method for a group, then put it on the private methods comment banner.
