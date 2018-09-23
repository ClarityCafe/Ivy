# Stdlib
import functools

# Ivy Internals
from framework.ivy import Ivy

__all__ = ("route", "Route")


def route(path, **kwargs):
    """
    Wraps a function to turn it into a `Route`.
    """
    def decorator(func):
        return Route(func, path, **kwargs)

    return decorator


class Route:
    """
    Route class wrapper to register them on the application
    """
    def __init__(self, func, path: str, **kwargs):
        self.func = func
        self.path = path
        self.kwargs = kwargs
        self.parent = None

    def register(self, core: Ivy):
        # Hack around making dynamic routes for flask
        _route = core.route(self.path, **self.kwargs)
        func = functools.wraps(self.func)(
            functools.partial(self.func, self.parent)
        )
        _route(func)

    def set_parent(self, parent):
        self.parent = parent
