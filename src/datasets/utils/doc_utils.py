from typing import Callable


def is_documented_by(function_with_docstring: Callable):
    """Decorator to share docstrings across common functions.

    Args:
        function_with_docstring (`Callable`): Name of the function with the docstring.
    """

    doc = function_with_docstring.__doc__
    
    def wrapper(target_function):
        target_function.__doc__ = doc
        return target_function

    return wrapper
