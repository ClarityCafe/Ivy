# Stdlib
import json

# External Libraries
from flask import Response


# Moved out of response_wrappers.py due to circular imports
def error_handler(func):
    """
    Similar to `json`, but a different format
    """
    def inner(*args, **kwargs):
        response = func(*args, **kwargs)
        text = response.response[0].decode()

        result = json.dumps({
            "error": text,
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
