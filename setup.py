if "CI" in os.environ:
    import sys
    sys.path.insert(0, "ChatLearner/chatbot/")
    
    from bottrainer import BotTrainer

from ChatLearner.chatbot.bottrainer import BotTrainer

corpus_dir = "resources/chatbot/corpus"
result_dir = "resources/chatbot/result"

trainer = BotTrainer(corpus_dir)
trainer.train(result_dir)

print("Done! Now you can run main.py!")
