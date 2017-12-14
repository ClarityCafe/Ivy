## Rebecca

This is the fallback service for Monika if the Dialogflow service goes down.

It's a simple Tensorflow chatbot capable of learning from user context.

### Requirements

You need the following to run Rebecca : 

- setuptools
- [Tensorflow 1.0 or Later](https://www.tensorflow.org/) **TODO: Makie it compliant to latest Tensorflow
- numpy
- scipy
- six

To prep the bot, go ahead and change the ``seq2seq.ini`` mode field to ``train``. To test the bot, set mode to
``test``.

### Copyright

Contains code from Brain Walker licensed under MIT. Copyright (c) 2017.

