# Stdlib
from functools import wraps
import json as _json

# External Libraries
from flask import Response, abort

# Ivy Internals
from framework.objects import limiter, auth_service

__all__ = ("json", "auth_has_ratelimit", "auth_only")


def json(func):
    """
    Wraps a function to return a unified JSON format
    """
    @wraps(func)
    def inner(*args, **kwargs):
        response = func(*args, **kwargs)
        text = response.response[0].decode()
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            data = text

        result = _json.dumps({
            "result": data,
            "status": response._status_code,  # flake8: noqa pylint: disable=protected-access
            "success": True if 200 <= response._status_code < 300 else False  # flake8: noqa pylint: disable=protected-access
        })

        return Response(
            response=result,
            headers=response.headers,
            status=response.status,
            content_type="application/json"
        )

    return inner


def auth_has_ratelimit(*rates: list):
    """
    Sets a custom ratelimit when using authorization keys
    """
    rates = rates or ("1 per second", "40 per minute", "2000 per hour")

    def decorator(func):
        func_wrapper = limiter.limit(
            ";".join(rates),
            key_func=auth_service.check_key
        )
        alt_func = func_wrapper(func)

        @wraps(func)
        def inner(*args, **kwargs):
            f = alt_func if auth_service.is_authenticated() else func
            return f(*args, **kwargs)

        return inner

    return decorator


def auth_only(func):
    """
    Only for requests with valid authorization keys
    """
    @wraps(func)
    def inner(*args, **kwargs):
        if auth_service.is_authenticated():
            return func(*args, **kwargs)
        return abort(403)

    return inner
