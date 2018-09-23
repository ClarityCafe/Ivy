# External Libraries
from flask import Response
from werkzeug.exceptions import HTTPException

# Ivy Internals
from framework.error_handlers.error_common import error_handler


@error_handler
def handle_400(err: HTTPException):
    return Response(err.description, 400)


@error_handler
def handle_401(err: HTTPException):
    return Response(err.description, 401)


@error_handler
def handle_403(err: HTTPException):
    return Response(err.description, 403)


@error_handler
def handle_404(err: HTTPException):
    return Response(err.description, 404)


@error_handler
def handle_405(err: HTTPException):
    return Response(err.description, 405)


@error_handler
def handle_429(err: HTTPException):
    return Response(f"Ratelimit for this endpoint: {err.description}", 429)


all_4xx = {
    400: handle_400,
    401: handle_401,
    403: handle_403,
    404: handle_404,
    405: handle_405,
    429: handle_429
}
