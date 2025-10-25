import tensorflow as tf


def rnn_free_run(cell, initial_state, sequence_length, initial_input):
    """Autoregressively samples from ``cell`` for ``sequence_length`` steps."""

    sequence_length = tf.cast(sequence_length, tf.int32)
    outputs = tf.TensorArray(dtype=tf.float32, size=sequence_length)
    state = initial_state
    current_input = initial_input
    finished = tf.zeros([tf.shape(initial_input)[0]], dtype=tf.bool)

    for t in tf.range(sequence_length):
        output, state = cell(current_input, state)
        outputs = outputs.write(t, output)
        next_input = cell.output_function(state)
        termination = cell.termination_condition(state, output=next_input)
        finished = tf.logical_or(finished, termination)
        current_input = tf.where(finished[:, None], tf.zeros_like(next_input), next_input)

    stacked = outputs.stack()
    stacked = tf.transpose(stacked, (1, 0, 2))
    return stacked, state
