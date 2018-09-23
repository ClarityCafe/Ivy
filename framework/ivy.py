# Stdlib
import glob
import importlib

# External Libraries
from flask import Flask

# Ivy Internals
from framework.error_handlers.error_4xx import all_4xx

__all__ = ("Ivy", )


class Ivy(Flask):
    """
    Core application. Wraps Flask to use `super.run` with some default arguments,
    while the user only has to run `run` without providing additional arguments.
    Additionally, we use this class to pass around to register all routes
    """
    def __init__(self):
        self.route_dir = ""
        super().__init__("Ivy")

        for code, func in all_4xx.items():
            self.register_error_handler(code, func)

    def gather(self, route_dir: str):
        for path in glob.glob(f"{route_dir}/**.py"):
            module = importlib.import_module(path.replace('/', '.')[:-3])

            if not hasattr(module, "setup"):
                raise ValueError(
                    f"Module {repr(module.__name__)} does not have a `setup` function!"
                )

            module.setup(self)
            del module

    def run(self, host: str = "localhost", port: int = 4444, *args, **kwargs):  # flake8: noqa pylint: disable=arguments-differ,keyword-arg-before-vararg
        super().run(host, port, *args, **kwargs)
