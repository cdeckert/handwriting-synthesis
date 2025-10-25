from __future__ import print_function
import os
from typing import Dict

import numpy as np
import tensorflow as tf

import drawing
from data_frame import DataFrame
from rnn_cell import LSTMAttentionCell, LSTMAttentionCellState
from rnn_ops import rnn_free_run
from tf_base_model import TFBaseModel


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = ['x', 'x_len', 'c', 'c_len']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95, random_state=2018)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            mode='train'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            mode='val'
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            mode='test'
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, mode='train'):
        gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode == 'test')
        )
        for batch in gen:
            batch['x_len'] = batch['x_len'] - 1
            max_x_len = np.max(batch['x_len'])
            max_c_len = np.max(batch['c_len'])
            batch['y'] = batch['x'][:, 1:max_x_len + 1, :]
            batch['x'] = batch['x'][:, :max_x_len, :]
            batch['c'] = batch['c'][:, :max_c_len]
            yield batch


class HandwritingNetwork(tf.keras.Model):

    def __init__(self, lstm_size, output_mixture_components, attention_mixture_components):
        super().__init__()
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.attention_mixture_components = attention_mixture_components
        self.output_units = self.output_mixture_components * 6 + 1

        self.cell = LSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            num_output_mixture_components=self.output_mixture_components,
        )
        self.rnn_layer = tf.keras.layers.RNN(
            self.cell,
            return_sequences=True,
            return_state=True,
            name='handwriting_rnn'
        )
        self.output_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.output_units),
            name='gmm_output'
        )

    def call(self, inputs: Dict[str, tf.Tensor], training=False):
        x = inputs['x']
        x_len = inputs['x_len']
        c = inputs['c']
        c_len = inputs['c_len']
        bias = inputs['bias']

        attention_values = tf.one_hot(c, len(drawing.alphabet), dtype=tf.float32)
        mask = tf.sequence_mask(x_len, maxlen=tf.shape(x)[1])
        initial_state = self.cell.get_initial_state(batch_size=tf.shape(x)[0], dtype=tf.float32)

        self.cell.set_constants(attention_values, c_len, bias)
        outputs_and_state = self.rnn_layer(
            x,
            mask=mask,
            initial_state=initial_state,
            training=training,
        )
        outputs = outputs_and_state[0]
        final_state = LSTMAttentionCellState(*outputs_and_state[1:])
        params = self.output_dense(outputs)

        return {
            'params': params,
            'final_state': final_state,
            'attention_values': attention_values,
            'attention_lengths': c_len,
            'bias': bias,
        }


class rnn(TFBaseModel):

    def __init__(
        self,
        lstm_size,
        output_mixture_components,
        attention_mixture_components,
        **kwargs
    ):
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.output_units = self.output_mixture_components*6 + 1
        self.attention_mixture_components = attention_mixture_components
        super(rnn, self).__init__(**kwargs)
        self.metrics = {'sequence_loss': 0.0}

    def build_model(self):
        return HandwritingNetwork(
            lstm_size=self.lstm_size,
            output_mixture_components=self.output_mixture_components,
            attention_mixture_components=self.attention_mixture_components,
        )

    def _prepare_batch(self, batch):
        tensors = {
            'x': tf.convert_to_tensor(batch['x'], dtype=tf.float32),
            'y': tf.convert_to_tensor(batch['y'], dtype=tf.float32),
            'x_len': tf.convert_to_tensor(batch['x_len'], dtype=tf.int32),
            'c': tf.convert_to_tensor(batch['c'], dtype=tf.int32),
            'c_len': tf.convert_to_tensor(batch['c_len'], dtype=tf.int32),
        }
        if 'bias' in batch:
            tensors['bias'] = tf.convert_to_tensor(batch['bias'], dtype=tf.float32)
        else:
            tensors['bias'] = tf.zeros(tf.shape(tensors['x_len']), dtype=tf.float32)
        return tensors

    def calculate_loss(self, batch, training=True):
        tensors = self._prepare_batch(batch)
        network_outputs = self.model(
            {
                'x': tensors['x'],
                'x_len': tensors['x_len'],
                'c': tensors['c'],
                'c_len': tensors['c_len'],
                'bias': tensors['bias'],
            },
            training=training,
        )
        params = network_outputs['params']
        pis, mus, sigmas, rhos, es = self.parse_parameters(params)
        sequence_loss, element_loss = self.NLL(
            tensors['y'], tensors['x_len'], pis, mus, sigmas, rhos, es
        )
        metrics = {'sequence_loss': tf.reduce_mean(sequence_loss)}
        return element_loss, metrics

    def parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            z,
            [
                1*self.output_mixture_components,
                2*self.output_mixture_components,
                1*self.output_mixture_components,
                2*self.output_mixture_components,
                1
            ],
            axis=-1
        )
        pis = tf.nn.softmax(pis, axis=-1)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        return pis, mus, sigmas, rhos, es

    def NLL(self, y, lengths, pis, mus, sigmas, rho, es, eps=1e-8):
        sigma_1, sigma_2 = tf.split(sigmas, 2, axis=2)
        y_1, y_2, y_3 = tf.split(y, 3, axis=2)
        mu_1, mu_2 = tf.split(mus, 2, axis=2)

        norm = 1.0 / (2*np.pi*sigma_1*sigma_2 * tf.sqrt(1 - tf.square(rho)))
        Z = tf.square((y_1 - mu_1) / (sigma_1)) + \
            tf.square((y_2 - mu_2) / (sigma_2)) - \
            2*rho*(y_1 - mu_1)*(y_2 - mu_2) / (sigma_1*sigma_2)

        exp = -1.0*Z / (2*(1 - tf.square(rho)))
        gaussian_likelihoods = tf.exp(exp) * norm
        gmm_likelihood = tf.reduce_sum(pis * gaussian_likelihoods, 2)
        gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

        bernoulli_likelihood = tf.squeeze(tf.where(tf.equal(tf.ones_like(y_3), y_3), es, 1 - es))

        nll = -(tf.math.log(gmm_likelihood) + tf.math.log(bernoulli_likelihood))
        sequence_mask = tf.logical_and(
            tf.sequence_mask(lengths, maxlen=tf.shape(y)[1]),
            tf.logical_not(tf.math.is_nan(nll)),
        )
        nll = tf.where(sequence_mask, nll, tf.zeros_like(nll))
        num_valid = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1)

        sequence_loss = tf.reduce_sum(nll, axis=1) / tf.maximum(num_valid, 1.0)
        element_loss = tf.reduce_sum(nll) / tf.maximum(tf.reduce_sum(num_valid), 1.0)
        return sequence_loss, element_loss

    def sample_sequences(
        self,
        c,
        c_len,
        sample_tsteps,
        num_samples,
        prime=False,
        x_prime=None,
        x_prime_len=None,
        bias=None,
    ):
        tensors = {
            'c': tf.convert_to_tensor(c, dtype=tf.int32),
            'c_len': tf.convert_to_tensor(c_len, dtype=tf.int32),
        }
        if bias is None:
            tensors['bias'] = tf.zeros([num_samples], dtype=tf.float32)
        else:
            tensors['bias'] = tf.convert_to_tensor(bias, dtype=tf.float32)

        attention_values = tf.one_hot(tensors['c'], len(drawing.alphabet), dtype=tf.float32)
        cell = self.model.cell
        cell.set_constants(attention_values, tensors['c_len'], tensors['bias'])

        initial_state = cell.get_initial_state(batch_size=num_samples, dtype=tf.float32)

        if prime and x_prime is not None and x_prime_len is not None:
            x_prime = tf.convert_to_tensor(x_prime, dtype=tf.float32)
            x_prime_len = tf.convert_to_tensor(x_prime_len, dtype=tf.int32)
            mask = tf.sequence_mask(x_prime_len, maxlen=tf.shape(x_prime)[1])
            outputs_and_state = self.model.rnn_layer(
                x_prime,
                mask=mask,
                initial_state=initial_state,
                training=False,
            )
            state = LSTMAttentionCellState(*outputs_and_state[1:])
            initial_input = cell.output_function(state)
        else:
            state = initial_state
            initial_input = tf.concat([
                tf.zeros([num_samples, 2]),
                tf.ones([num_samples, 1]),
            ], axis=1)

        samples, _ = rnn_free_run(
            cell=cell,
            initial_state=state,
            sequence_length=sample_tsteps,
            initial_input=initial_input,
        )
        return samples


if __name__ == '__main__':
    dr = DataReader(data_dir='data/processed/')

    nn = rnn(
        reader=dr,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions',
        learning_rates=[.0001, .00005, .00002],
        batch_sizes=[32, 64, 64],
        patiences=[1500, 1000, 500],
        beta1_decays=[.9, .9, .9],
        validation_batch_size=32,
        optimizer='rms',
        num_training_steps=100000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=2000,
        log_interval=20,
        grad_clip=10,
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10
    )
    nn.fit()
