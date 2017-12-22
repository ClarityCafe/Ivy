import numpy as np
import tensorflow as tf
import os.path as osp

import os
import yaml
import pickle

from utils import TextLoader
from model import Model
from time import time

def train():
    # Load config file.
    with open('./training_config.yaml') as f:
        config = yaml.load(f)

    # Create the data_loader object, which loads up all of our batches, vocab dictionary, etc. from utils.poy (and creates them if they don't already exist).
    # These files go in the data directory.
    data_loader = TextLoader(config['data_dir'], config['batch_size'], config['seq_length'])
    config['vocab_size'] = data_loader.vocab_size
    load_model = False

    if not osp.exists(config['save_dir']):
        print(f'Creating saves directory "{config["save_dir"]}"')
        os.mkdir(config['save_dir'])
    elif osp.exists(osp.join(config['save_dir'], 'config.yaml')):
        # Trained model already exists
        ckpt = tf.train.get_checkpoint_state(config['save_dir'])

        if ckpt and ckpt.model_checkpoint_state:
            with open(osp.join(config['save_dir'], 'config.yaml')) as f:
                saved_args = yaml.load(f)
                config['rnn_size'] = saved_args['rnn_size']
                config['num_layers'] = saved_args['num_layers']
                config['model'] = saved_args['model']

                print('Found a previous checkpoint. Overwriting model description arguments to:')
                print(f' model: {saved_args["model"]}, rnn_size: {saved_args["rnn_size"]}, num_layers: {saved_args["num_layers"]}')

                load_model = True

    # Save all arguments to config.yaml in the save directory -- NOT the data directory.
    with open(osp.join(config['save_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Save a tuple of the characters list and the vocab dictionary to chars_vocab.pkl in the save directory -- NOT the data directory.
    with open(osp.join(config['save_dir'], 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.chars, data_loader.vocab), f)

    # Create the model!
    print('Building the model...')

    model = Model(**config)
    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(model.save_variables_list())

        if load_model:
            print('Loading saved parameters...')
            saver.restore(sess, ckpt.model_checkpoint_path)

        global_epoch_fraction = sess.run(model.global_epoch_fraction)
        global_seconds_elapsed = sess.run(model.global_seconds_elapsed)

        if load_model:
            print(f'Resuming from global epoch fraction {global_epoch_fraction:.3f}, total trained time: {global_seconds_elapsed}, learning rate: {sess.run(model.lr)}')

        data_loader.cue_batch_pointer_to_epoch_fraction(global_epoch_fraction)

        inital_batch_step = int((global_epoch_fraction - int(global_epoch_fraction)) * data_loader.total_batch_count)
        epoch_range = (int(global_epoch_fraction), config['num_epochs'] + int(global_epoch_fraction))
        writer = tf.train.SummaryWriter(config['save_dir'], graph=tf.get_default_graph())
        outputs = [model.cost, model.final_state, model.train_op, model.summary_op]
        is_lstm = config['model'] == 'lstm'
        global_step = epoch_range[0] * data_loader.total_batch_count + inital_batch_step

        try:
            for e in range(*epoch_range):
                # e iterates through the training epochs.
                # Reset the model state, so it does not carry over from the end of the previous epoch.
                state = sess.run(model.inital_state)
                batch_range = (inital_batch_step, data_loader.total_batch_count)
                inital_batch_step = 0

                for b in range(*batch_range):
                    global_step += 1

                    if global_step % config['decay_steps'] == 0:
                        # Set the model.lr element of the model to track the appropriately decayed learning rate.
                        current_learning_rate = sess.run(model.lr)
                        current_learning_rate *= config['decay_rate']

                        sess.run(tf.assign(model.lr, current_learning_rate))
                        print(f'Decayed learning rate to {current_learning_rate}')

                    start = time()

                    # Pull the next batch inputs (x) and targets (y) from the data loader.
                    x, y = data_loader.next_batch()

                    # feed is a dictionary of variable references and respective values for initialization.
                    # Initialize the model's input data and target data from the batch, and initialize the model state to the final state from the previous batch, so that model state is accumulated and carried over between batches.
                    feed = {model.input_data: x, model.targets: y}

                    if is_lstm:
                        for i, (c, h) in enumerate(model.inital_state):
                            feed[c] = state[i].c
                            feed[h] = state[i].h
                    else:
                        for i, c in enumerate(model.initial_state):
                            feed[c] = state[i]

                    # Run the session! Specifically, tell TensorFlow to compute the graph to calculate the values of cost, final state, and the training op.
                    # Cost is used to monitor progress.
                    # Final state is used to carry over the state into the next batch.
                    # Training op is not used, but we want it to be calculated, since that calculation is what updates parameter states (i.e. that is where the training happens).
                    train_loss, state, _, summary = sess.run(outputs, feed)
                    elapsed = time() - start
                    global_seconds_elapsed += elapsed

                    writer.add_summary(summary, e * batch_range[1] + b + 1)
                    print(f'{b}/{batch_range[1]} (epoch {e}/{epoch_range[1]}), loss = {train_loss:.3f}, time/batch = {elapsed:.3f}s')

                    # Every save_every batches, save the model to disk.
                    # By default, only the five most recent checkpoint files are kept.
                    if (e * batch_range[1] + b + 1) % args.save_every == 0 or (e == epoch_range[1] - 1 and b == batch_range[1] - 1):
                        save_model(sess, saver, model, config['save_dir'], global_step, data_loader.total_batch_count, global_seconds_elapsed)
        except KeyboardInterrupt:
            # Intyroduce a line break after ^C is displayed so save message is on its own line.
            print()
        finally:
            global_step = e * data_loader.total_batch_count + b

            writer.flush()
            save_model(sess, saver, model, config['save_dir'], global_step, data_loader.total_batch_count, global_seconds_elapsed)

def save_model(sess, saver, model, save_dir, global_step, steps_per_epoch, global_seconds_elapsed):
    global_epoch_fraction = float(global_step) / float(steps_per_epoch)
    checkpoint_path = osp.join(save_dir, 'model.ckpt')

    print(f'Saving model to {checkpoint_path} (epoch franction {global_epoch_fraction:.3f})')

    sess.run(tf.assign(model.global_epoch_fraction, global_epoch_fraction))
    sess.run(tf.assign(model.global_seconds_elapsed, global_seconds_elapsed))
    saver.save(sess, checkpoint_path, global_step=global_step)

    print('Model saved.')

if __name__ == '__main__':
    train()