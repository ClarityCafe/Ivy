import numpy as np
import tensorflow as tf
import os.path as osp

import copy
import yaml
import pickle

from utils import TextLoader
from model import Model
from types import SimpleNamespace

def get_paths(input_path):
    if osp.isfile(input_path): 
        # Passed a model rather than a checkpoint directory
        model_path = input_path
        save_dir = osp.dirname(model_path)
    elif osp.exists(input_path):
        # Passed a checkpoint directory.
        save_dir = input_path
        checkpoint = tf.train.get_checkpoint_state(save_dir)

        if checkpoint:
            model_path = checkpoint.model_checkpoint_path
        else:
            raise ValueError(f'Checkpoint not found in "{save_dir}".')
    else:
        raise ValueError('save_dir is not a valid path.')

    return model_path, osp.join(save_dir, 'config.yaml'), osp.join(save_dir, 'chars_vocab.pkl')

class Chatbot:
    def __init__(self):
        # Load config file.
        with open('./config.yaml') as f:
            self.config = SimpleNamespace(**yaml.load(f)) # Makes the config a bit easier to access (totally not because I'm from JS or anything).

        model_path, config_path, vocab_path = get_paths(config.save_dir)

        # Load arguments with which the model was previously trained.
        with open(config_path) as f:
            saved_args = SimpleNamespace(**yaml.load(f))

        # Load chars and vocab from save directory.
        with open(vocab_path) as f:
            self.chars, self.vocab = pickle.load(f)

        # Create the model from the saved arguments, in inference mode.
        print('Creating model...')

        self.net = Model(saved_args, True)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        tf.global_variables_initializer().run()

        saver = tf.train.Saver(self.net.save_variables_list())

        # Restore the saved variables, replacing the initialised values.
        print('Restoring weights...')
        saver.restore(sess, model_path)

    def _initial_state(self):
        return self.sess.run(self.net.cell.zero_state(1, tf.float32))

    def forward_text(self, states, prime_text=None):
        if prime_text is not None:
            for char in prime_text:
                if len(states) == 2:
                    # Automatically forward the primary net.
                    _, states[0] = self.net.forward_model(self.sess, states[0], self.vocab[char])

                    # If the token is a newline, reset the mask net state; else, forward it.
                    if self.vocab[char] == '\n':
                        states[1] = self._initial_state()
                    else:
                        _, states[1] = self.net.forward_model(self.sess, states, self.vocab[char])

        return states

    def scale_prediction(self, prediction=None):
        if prediction is None:
            prediction = self.config.prediction

        if self.config.temperature == 1.0:
            # temperature of 1.0 makes no change
            return self.config.prediction

        np.seterr(divide='ignore')

        scaled_pred = np.log(self.config.prediction) / self.config.temperature
        scaled_pred = scaled_pred - np.logaddexp.reduce(scaled_pred)
        scaled_pred = np.exp(scaled_pred)

        np.seterr(divide='warn')

        return scaled_pred

    def beam_sample(self):
        states = [self._initial_state(), self._initial_state()]
        states = self.forward_text(states, self.config.prime_text)
        comp_resp_gen = self.beam_search_generator()
        final = ''

        for i, token in enumerate(comp_resp_gen):
            final += self.chars[token]
            states = self.forward_text(states, self.chars[token])

            if i >= self.config.max_length:
                break

        return final

    def sanitize_text(self, text):
        return ''.join(x for x in text if x in self.vocab)

    def init_state_with_relev_mask(self):
        if self.config.relevance <= 0:
            return self._initial_state()
        else:
            return [self._initial_state(), self._initial_state()]

    def consensus_length(self, beam_outputs, early_term_token):
        for l in range(len(beam_outputs[0])):
            if l > 0 and beam_outputs[0][l - 1] == early_term_token:
                return l - 1, True

            for b in beam_outputs[1:]:
                if beam_outputs[0][l] != b[1]:
                    return l, False

        return l, False

    def forward_with_mask(self, states, input_sample, forward_args):
        if len(states) != 2:
            # No relevance masking.
            prob, states = self.net.forward_model(self.sess, states, input_sample)
            return prob / sum(prob), states

        # states should be a 2-length list: [primary net state, mask net state].
        # forward_args should be a 2-length list/tuple: [relevance, mask_reset_token].
        relevance, mask_reset_token = forward_args

        if input_sample == mask_reset_token:
            # Reset the mask probs when reaching mask_reset_token (newline).
            states[1] = self._initial_state()

        primary_prob, states[0] = self.net.forward_model(self.sess, states[0], input_sample)
        primary_prob /= sum(primary_prob)
        mask_prob, states[1] = self.net.forward_model(self.sess, states[1], input_sample)
        mask_prob /= sum(mask_prob)
        combined_prob = np.exp(np.log(primary_prob) - relevance * np.log(mask_prob))

        # Normalise probabilities so they sum to 1.
        return combined_prob / sum(combined_prob), states

    def beam_search_generator(self, initial_state, initial_sample, early_term_token, forward_args):
        # Store state, outputs and probabilities for up to config.beam_width beams.
        # Initialise with just the one starting entry; it will branch to fill the beam in the first step.
        beam_states = [initial_state] # Stores the best activation states.
        beam_outputs = [[inital_sample]] # Stores the best generated output sequences so far.
        beam_probs = [1] # Stores the cumulative normalised probabilities of the beam so far.

        while True:
            # Keep a running list of the best beam branches for next step.
            # Don't actually copy any big data structures yet, just keep references to existing beam state entries, and then clone them as necessary at the end of the generation step.
            new_beam_indices = []
            new_beam_probs = []
            new_beam_samples = []

            # Iterate through the beam entries.
            for i, state in enumerate(beam_states):
                prob = beam_probs[i]
                sample = beam_outputs[beam_index][-1]

                # Forward the model.
                pred, beam_states[i] = self.forward_with_mask(state, sample, forward_args)
                pred = self.scale_prediction(pred)

                # Sample best_tokens from the probability distribution.
                # Sample from the scaled probability distribution beam_width choices (bot not more than the number of positive probabilities).
                count = min(beam_width, sum(1 if p > 0 else 0 for p in pred))
                best_tokens = np.random.choice(len(pred), size=count, replace=False, p=pred)

                for token in best_tokens:
                    prob = pred[token] * beam_prob

                    if len(new_beam_indices) < beam_width:
                        # If we don't have enough new_beam_indices, we automatically qualify.
                        new_beam_indices.append(i)
                        new_beam_probs.append(prob)
                        new_beam_samples.append(token)
                    else:
                        # Sample a low-probability beam to possibly replace.
                        np_new_beam_probs = np.array(new_beam_probs)
                        inverse_probs = -np_new_beam_probs + max(np_new_beam_probs) + min(np_new_beam_probs)
                        inverse_probs = inverse_probs / sum(inverse_probs)
                        sampled_index = np.random.choice(beam_width, p=inverse_probs)

                        if new_beam_probs[sampled_index] <= prob:
                            # Replace it.
                            new_beam_indices[sampled_index] = i
                            new_beam_probs[sampled_index] = prob
                            new_beam_samples[sampled_index] = token

            # Replace the old states with the new states, first by referencing and then by copying.
            already_referenced = [False] * beam_width
            new_beam_states = []
            new_beam_outputs = []

            for i, new_index in enumerate(new_beam_indices):
                if already_referenced[new_index]:
                    new_beam = copy.deepcopy(beam_states[new_index])
                else:
                    new_beam = beam_states[new_index]
                    already_referenced[new_index] = True

                new_beam_states.append(new_beam)
                new_beam_outputs.append(beam_outputs[new_index] + [new_beam_samples[i]])

            # Normalise the beam probabilities so they don't drop to zero.
            beam_probs = new_beam_probs / sum(new_beam_probs)
            beam_states = new_beam_states
            beam_outputs = new_beam_outputs

            # Prune the agreed portions of the outputs and yield the tokens on which the beam has reached consensus.
            l, early_term = consensus_length(bema_outputs, early_term_token)

            if l > 0:
                for token in beam_outputs[0][:l]:
                    yield token

                beam_outputs = [output[l:] for output in beam_outputs]

            if early_term:
                return

    def chat(self, user_input):
        states = self.init_state_with_relev_mask()
        user_input = self.sanitize_text(user_input)
        states = self.forward_text(states, f'> {user_input}\n>')
        comp_resp_gen = self.beam_search_generator(copy.deepcopy(states), self.vocab[' '], self.vocab['\n'], (self.config.relevance, self.vocab['\n']))
        final = ''

        for i, token in enumerate(comp_resp_gen):
            final += self.chars[token]
            states = self.forward_text(states, self.chars[token])

            if i >= self.config.max_length

        states = forward_text(states, '\n> ')

        return final