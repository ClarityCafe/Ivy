## Rebecca

This is the fallback service for Monika if the Dialogflow service goes down.

It's a simple Tensorflow chatbot capable of learning from user context.

### Neural network layout

Rebecca uses the LTST Brain layout and the recurrent model layout. Recurrent Layout allows
the bot to learn from context vector encoded by the human and using a long term short term network
allows the brain to remember words or phrases.

### Requirements

You need the following to run Rebecca : 

- setuptools
- [Tensorflow r0.12](https://www.tensorflow.org/versions/r0.12/get_started/os_setup#pip_installation) **TODO: Makie it compliant to latest Tensorflow
- numpy
- scipy
- six

To prep the bot, go ahead and change the ``seq2seq.ini`` mode field to ``train``. To test the bot, set mode to
``test``.

### Copyright

Contains code from the Tensorflow Authors licensed under Apache 2.0. All Rights Reserved.

Contains processed data from the Cornell Movie Dialogue Database. All Rights Reserved.

