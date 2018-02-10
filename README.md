# Libitina

This is the next chatbot engine for Monika, replacing the Dialogflow implementation on v1.  
It's a simple Tensorflow chatbot capable of learning from user context.

The chatbot is a Python 3.6 port, and extension of [tensorlayer/seq2seq-chatobbt](https://github.com/tensorlayer/seq2seq-chatbot) that is modified to run off YAML config files and interface with a HTTP API.

# Dependencies

 - pyyaml
 - numpy
 - tensorflow
 - tensorlayer
 - sklearn
 - nltk

# Licenses

Contains code from "zsdonghao", license unspecified.  
All code not from the forked project is licensed under the MIT license.
