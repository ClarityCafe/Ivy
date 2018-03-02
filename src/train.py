import utils
import yaml

from model import Model

with open('config.yaml') as f:
    config = utils.AttrObj(yaml.load(f))

train_model = Model(config, train=True, reuse=False)
inference_model = Model(config)

train_model.load_data(config.data_path)
inference_model.load_data(config.data_path, True)

train_model.load_state()
inference_model.load_state(True)

train_model.init_models()
inference_model.init_models()

for epoch in range(config.epochs):
    train_model.train_epoch(epoch, inference_model)
    train_model.save_state(epoch)
