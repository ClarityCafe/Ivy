# Stdlib
import json

# External Libraries
from flask import request


class Authenticator:
    """
    Service to check API key validity
    """
    def __init__(self, token_path: str):
        self.auth_tokens = []
        self.token_path = token_path

    def check_key(self) -> str:
        return request.headers["API-KEY"]

    def is_authenticated(self) -> bool:
        return "API-KEY" in request.headers and self.valid_key()

    def valid_key(self) -> bool:
        return request.headers["API-KEY"] in self.auth_tokens

    def reload_tokens(self):
        with open(self.token_path) as file:
            self.auth_tokens = json.load(file)
