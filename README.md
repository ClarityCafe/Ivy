## Rebecca

This is the fallback service for Monika if the Dialogflow service goes down.  
It's a simple Tensorflow chatbot capable of learning from user context.

The chatbot is a Python 3.6 rewrite of [pender/chatbot-rnn](https://github.com/pender/chatbot-rnn) that is modified to run off YAML config files and interface with a HTTP API.
### Dependencies

 - pyyaml
 - numpy
 - tensorflow

### Copyright

Contains code from "pender", licensed under MIT. Copyright (c) 2017.