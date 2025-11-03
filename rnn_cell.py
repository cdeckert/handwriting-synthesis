from collections import namedtuple
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

import drawing


LSTMAttentionCellState = namedtuple(
    "LSTMAttentionCellState",
    ["h1", "c1", "h2", "c2", "h3", "c3", "alpha", "beta", "kappa", "w", "phi"],
)
class LSTMAttentionCell(tf.keras.layers.Layer):
    """Custom RNN cell implementing the attention mechanism from Graves (2013)."""

    def __init__(
        self,
        lstm_size: int,
        num_attn_mixture_components: int,
        num_output_mixture_components: int,
        bias: Optional[tf.Tensor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.num_output_mixture_components = num_output_mixture_components
        self.output_units = 6 * self.num_output_mixture_components + 1
        self.bias = bias
        self._attention_values = None
        self._attention_lengths = None

        self.lstm1 = tf.keras.layers.LSTMCell(self.lstm_size)
        self.lstm2 = tf.keras.layers.LSTMCell(self.lstm_size)
        self.lstm3 = tf.keras.layers.LSTMCell(self.lstm_size)

        self.attention_dense = tf.keras.layers.Dense(
            3 * self.num_attn_mixture_components, activation=tf.nn.softplus, name="attention"
        )
        self.gmm_dense = tf.keras.layers.Dense(self.output_units, name="gmm")

    def build(self, input_shape):
        feature_size = len(drawing.alphabet)
        input_shape = tf.TensorShape(input_shape)
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("LSTMAttentionCell requires a known input dimension during build")
        input_dim = int(input_dim)

        self.lstm1.build((None, feature_size + input_dim))
        lstm_concat_dim = input_dim + feature_size + self.lstm_size
        self.attention_dense.build((None, lstm_concat_dim))
        self.lstm2.build((None, lstm_concat_dim))
        self.lstm3.build((None, lstm_concat_dim))
        self.gmm_dense.build((None, self.lstm_size))

        super().build(input_shape)

    @property
    def state_size(self):
        feature_size = len(drawing.alphabet)
        return LSTMAttentionCellState(
            tf.TensorShape([self.lstm_size]),
            tf.TensorShape([self.lstm_size]),
            tf.TensorShape([self.lstm_size]),
            tf.TensorShape([self.lstm_size]),
            tf.TensorShape([self.lstm_size]),
            tf.TensorShape([self.lstm_size]),
            tf.TensorShape([self.num_attn_mixture_components]),
            tf.TensorShape([self.num_attn_mixture_components]),
            tf.TensorShape([self.num_attn_mixture_components]),
            tf.TensorShape([feature_size]),
            tf.TensorShape([None]),
        )

    @property
    def output_size(self):
        return self.lstm_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        dtype = dtype or tf.float32
        if batch_size is None:
            if inputs is None:
                raise ValueError("inputs or batch_size must be provided")
            batch_size = tf.shape(inputs)[0]
        feature_size = len(drawing.alphabet)

        def zeros(units):
            return tf.zeros([batch_size, units], dtype=dtype)

        return LSTMAttentionCellState(
            zeros(self.lstm_size),
            zeros(self.lstm_size),
            zeros(self.lstm_size),
            zeros(self.lstm_size),
            zeros(self.lstm_size),
            zeros(self.lstm_size),
            zeros(self.num_attn_mixture_components),
            zeros(self.num_attn_mixture_components),
            zeros(self.num_attn_mixture_components),
            zeros(feature_size),
            zeros(1),
        )

    def set_constants(self, attention_values, attention_lengths, bias):
        self._attention_values = attention_values
        self._attention_lengths = attention_lengths
        if bias is not None:
            self.bias = bias

    def call(self, inputs, states: Sequence[tf.Tensor], constants=None, **kwargs):
        del kwargs
        if constants is not None:
            if len(constants) < 3:
                raise ValueError("LSTMAttentionCell requires attention constants")
            attention_values, attention_lengths, bias = constants
            self.set_constants(attention_values, attention_lengths, bias)

        if self._attention_values is None or self._attention_lengths is None:
            raise ValueError("Attention constants must be set before calling the cell")

        attention_values = self._attention_values
        attention_lengths = self._attention_lengths

        batch_size = tf.shape(inputs)[0]
        char_len = tf.shape(attention_values)[1]
        feature_size = tf.shape(attention_values)[2]

        state = LSTMAttentionCellState(*states)

        s1_in = tf.concat([state.w, inputs], axis=1)
        s1_out, [new_h1, new_c1] = self.lstm1(s1_in, [state.h1, state.c1])

        attention_inputs = tf.concat([state.w, inputs, s1_out], axis=1)
        attention_params = self.attention_dense(attention_inputs)
        alpha, beta, kappa = tf.split(attention_params, 3, axis=1)
        kappa = state.kappa + kappa / 25.0
        beta = tf.clip_by_value(beta, 0.01, np.inf)

        kappa_expanded = tf.expand_dims(kappa, 2)
        alpha_expanded = tf.expand_dims(alpha, 2)
        beta_expanded = tf.expand_dims(beta, 2)

        enum = tf.reshape(tf.range(char_len), (1, 1, char_len))
        enum = tf.cast(tf.tile(enum, (batch_size, self.num_attn_mixture_components, 1)), tf.float32)
        phi_flat = tf.reduce_sum(alpha_expanded * tf.exp(-tf.square(kappa_expanded - enum) / beta_expanded), axis=1)

        phi = tf.expand_dims(phi_flat, 2)
        sequence_mask = tf.sequence_mask(attention_lengths, maxlen=char_len, dtype=tf.float32)
        sequence_mask = tf.expand_dims(sequence_mask, 2)
        w = tf.reduce_sum(phi * attention_values * sequence_mask, axis=1)

        s2_in = tf.concat([inputs, s1_out, w], axis=1)
        s2_out, [new_h2, new_c2] = self.lstm2(s2_in, [state.h2, state.c2])

        s3_in = tf.concat([inputs, s2_out, w], axis=1)
        s3_out, [new_h3, new_c3] = self.lstm3(s3_in, [state.h3, state.c3])

        new_state = LSTMAttentionCellState(
            new_h1,
            new_c1,
            new_h2,
            new_c2,
            new_h3,
            new_c3,
            alpha,
            beta,
            kappa,
            w,
            phi_flat,
        )

        return s3_out, new_state

    def output_function(self, state: LSTMAttentionCellState):
        params = self.gmm_dense(state.h3)
        pis, mus, sigmas, rhos, es = self._parse_parameters(params)

        pis_sum = tf.reduce_sum(pis, axis=1, keepdims=True)
        pis_normalized = tf.math.divide_no_nan(pis, pis_sum)
        default_probs = tf.fill(
            [tf.shape(pis)[0], self.num_output_mixture_components],
            1.0 / tf.cast(self.num_output_mixture_components, tf.float32),
        )
        pis = tf.where(pis_sum > 0.0, pis_normalized, default_probs)

        batch_size = tf.shape(pis)[0]
        mus = tf.reshape(
            mus,
            (batch_size, self.num_output_mixture_components, 2),
        )
        sigma_params = tf.reshape(
            sigmas,
            (batch_size, self.num_output_mixture_components, 2),
        )

        component_idx = tf.squeeze(
            tf.random.categorical(
                tf.math.log(pis + 1e-8), num_samples=1, dtype=tf.int32
            ),
            axis=1,
        )

        def gather(param):
            return tf.gather(param, component_idx, batch_dims=1)

        mu_selected = gather(mus)
        sigma_selected = gather(sigma_params)
        rho_selected = gather(rhos)

        mu1_selected = mu_selected[:, 0]
        mu2_selected = mu_selected[:, 1]
        sigma1_selected = sigma_selected[:, 0]
        sigma2_selected = sigma_selected[:, 1]

        z1 = tf.random.normal(tf.shape(mu1_selected), dtype=tf.float32)
        z2 = tf.random.normal(tf.shape(mu1_selected), dtype=tf.float32)

        x = mu1_selected + sigma1_selected * z1
        y = mu2_selected + sigma2_selected * (
            rho_selected * z1
            + tf.sqrt(tf.maximum(1.0 - tf.square(rho_selected), 1e-8)) * z2
        )

        eos_prob = tf.squeeze(es, axis=1)
        sampled_e = tf.cast(
            tf.random.uniform(tf.shape(eos_prob), dtype=tf.float32) < eos_prob,
            tf.float32,
        )

        coords = tf.stack([x, y], axis=1)
        sampled = tf.concat([coords, sampled_e[:, None]], axis=1)
        return sampled

    def termination_condition(
        self,
        state: LSTMAttentionCellState,
        attention_lengths=None,
        output=None,
    ):
        if attention_lengths is None:
            if self._attention_lengths is None:
                raise ValueError("Attention lengths must be set before calling termination_condition")
            attention_lengths = self._attention_lengths
        char_idx = tf.cast(tf.argmax(state.phi, axis=1), tf.int32)
        final_char = char_idx >= attention_lengths - 1
        past_final_char = char_idx >= attention_lengths
        if output is None:
            output = self.output_function(state)
        es = tf.cast(output[:, 2], tf.int32)
        is_eos = tf.equal(es, tf.ones_like(es))
        return tf.logical_or(tf.logical_and(final_char, is_eos), past_final_char)

    def _parse_parameters(self, gmm_params, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            gmm_params,
            [
                1 * self.num_output_mixture_components,
                2 * self.num_output_mixture_components,
                1 * self.num_output_mixture_components,
                2 * self.num_output_mixture_components,
                1,
            ],
            axis=-1,
        )
        if self.bias is not None:
            pis = pis * (1 + tf.expand_dims(self.bias, 1))
            sigmas = sigmas - tf.expand_dims(self.bias, 1)

        pis = tf.nn.softmax(pis, axis=-1)
        pis = tf.where(pis < 0.01, tf.zeros_like(pis), pis)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        es = tf.where(es < 0.01, tf.zeros_like(es), es)

        return pis, mus, sigmas, rhos, es
