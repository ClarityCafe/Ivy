import re
from urllib.parse import unquote

from flask import request, abort, jsonify

from ChatLearner.chatbot.botpredictor import BotPredictor
from framework.ivy import Ivy
from framework.route import route
from framework.route_wrappers import json
from framework.routecog import RouteCog

import tensorflow as tf


class Chatbot(RouteCog):
    def __init__(self, core: Ivy):
        super().__init__(core)

        self.sessions = set()

        corp_dir = "resources/chatbot/corpus/"
        knbs_dir = "resources/chatbot/knowledgebase/"
        res_dir = "resources/chatbot/result/"

        with tf.Session() as sess:
            self.bot = BotPredictor(sess, corpus_dir=corp_dir,
                                    knbase_dir=knbs_dir,
                                    result_dir=res_dir,
                                    result_file="basic")

    def new_session(self) -> str:
        token = self.bot.session_data.add_session()
        self.sessions.add(token)
        return token

    @route("/api/chatbot/session")
    @json
    def session(self) -> str:
        session_id = self.new_session()
        return jsonify(session_id)

    @route("/api/chatbot/query", methods=["POST"])
    @json
    def query(self) -> str:
        session_id = request.json.get("session_id") or unquote(request.args.get("session_id"))
        message = request.json.get("message") or unquote(request.args.get("message"))

        if any(x is None for x in (session_id, message)):
            abort(400, "Both 'session_id' and 'message' have to be passed as URL parameters.")

        if session_id not in self.sessions:
            abort(403, f"{session_id} is not a valid session ID.")

        # HACK: JSONifed respose, I don't know if this has the "message" field.
        message = re.sub(r'_nl_|_np_', '\n',
                      self.bot.predict(session_id, message)).strip()

        return jsonify(message)


def setup(core: Ivy):
    Chatbot(core).register()
