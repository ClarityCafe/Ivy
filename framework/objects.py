# Stdlib
import logging

# External Libraries
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Ivy Internals
from framework.authentication import Authenticator
from framework.ivy import Ivy

__all__ = ("ivy_instance", "limiter", "logger", "auth_service")

ivy_instance = Ivy()

limiter = Limiter(
    ivy_instance,
    key_func=get_remote_address,
    default_limits=["1 per 2 seconds", "20 per minute", "1000 per hour"]
)

logger = logging.getLogger("Ivy")
logger.setLevel(logging.INFO)

auth_service = Authenticator("resources/json/keys.json")
auth_service.reload_tokens()
