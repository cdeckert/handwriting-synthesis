"""Export and verify one handwriting synthesis RNN step as Core ML.

The original inference graph contains a TensorFlow control-flow loop and random
distribution operators. Core ML Tools cannot convert that cyclic graph as one
unit. This exporter isolates the trained recurrent/attention/GMM step, which is
acyclic, and leaves the sampling loop to the native client.
"""

# ruff: noqa: E402, I001 -- repository modules require adding the project root.

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import drawing
from rnn_cell import LSTMAttentionCell, LSTMAttentionCellState
from tf_utils import dense_layer


DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "model-17900"
DEFAULT_OUTPUT = (
    BASE_DIR / "ios" / "HandwritingStudio" / "Resources" / "Models" / "HandwritingStep.mlmodel"
)

STATE_NAMES = ("h1", "c1", "h2", "c2", "h3", "c3")
OUTPUT_NAMES = (
    "gmm_params",
    "next_h1",
    "next_c1",
    "next_h2",
    "next_c2",
    "next_h3",
    "next_c3",
    "next_kappa",
    "next_window",
    "next_phi",
)


def build_and_freeze_step(checkpoint: Path) -> tf.GraphDef:
    """Restore the trained weights and return an acyclic frozen step graph."""

    tf.disable_v2_behavior()
    graph = tf.Graph()

    with graph.as_default():
        stroke = tf.placeholder(tf.float32, [1, 3], name="stroke")
        chars = tf.placeholder(tf.int32, [1, 120], name="chars")
        chars_len = tf.placeholder(tf.int32, [1], name="chars_len")
        state_inputs = {
            name: tf.placeholder(tf.float32, [1, 400], name=name) for name in STATE_NAMES
        }
        kappa = tf.placeholder(tf.float32, [1, 10], name="kappa")
        window = tf.placeholder(tf.float32, [1, len(drawing.alphabet)], name="window")

        state = LSTMAttentionCellState(
            state_inputs["h1"],
            state_inputs["c1"],
            state_inputs["h2"],
            state_inputs["c2"],
            state_inputs["h3"],
            state_inputs["c3"],
            tf.zeros([1, 10]),
            tf.zeros([1, 10]),
            kappa,
            window,
            tf.zeros([1, 120]),
        )
        cell = LSTMAttentionCell(
            lstm_size=400,
            num_attn_mixture_components=10,
            attention_values=tf.one_hot(chars, len(drawing.alphabet)),
            attention_values_lengths=chars_len,
            num_output_mixture_components=20,
            # Bias changes sampling distributions, not the recurrent step.
            bias=tf.zeros([1]),
        )

        with tf.variable_scope("rnn"):
            _, next_state = cell(stroke, state)
            gmm_params = dense_layer(
                next_state.h3,
                121,
                scope="gmm",
                reuse=tf.AUTO_REUSE,
            )

        outputs = (
            tf.identity(gmm_params, name="gmm_params"),
            tf.identity(next_state.h1, name="next_h1"),
            tf.identity(next_state.c1, name="next_c1"),
            tf.identity(next_state.h2, name="next_h2"),
            tf.identity(next_state.c2, name="next_c2"),
            tf.identity(next_state.h3, name="next_h3"),
            tf.identity(next_state.c3, name="next_c3"),
            tf.identity(next_state.kappa, name="next_kappa"),
            tf.identity(next_state.w, name="next_window"),
            tf.identity(next_state.phi, name="next_phi"),
        )
        saver = tf.train.Saver(var_list=tf.global_variables())

    with tf.Session(graph=graph) as session:
        saver.restore(session, str(checkpoint))
        return tf.graph_util.convert_variables_to_constants(
            session,
            graph.as_graph_def(),
            [tensor.op.name for tensor in outputs],
        )


def convert_to_coreml(
    graph_def: tf.GraphDef,
    output: Path,
    checkpoint: Path,
):
    """Convert a frozen TensorFlow GraphDef into a Core ML neural network.

    The neural-network representation intentionally avoids the MLE5 ML Program
    runtime. That runtime aborts while binding this model's output buffers on
    some physical iOS devices. Core ML can still place compatible layers on the
    GPU or Neural Engine when the app loads this model with ``.all`` compute
    units.
    """

    import coremltools as ct

    if output.suffix != ".mlmodel":
        raise ValueError("The neural-network export must use a .mlmodel output path.")

    inputs = [
        ct.TensorType(name="stroke", shape=(1, 3)),
        ct.TensorType(name="chars", shape=(1, 120), dtype=np.int32),
        ct.TensorType(name="chars_len", shape=(1,), dtype=np.int32),
        *[ct.TensorType(name=name, shape=(1, 400)) for name in STATE_NAMES],
        ct.TensorType(name="kappa", shape=(1, 10)),
        ct.TensorType(name="window", shape=(1, len(drawing.alphabet))),
    ]
    outputs = [ct.TensorType(name=name) for name in OUTPUT_NAMES]
    model = ct.convert(
        graph_def,
        source="tensorflow",
        inputs=inputs,
        outputs=outputs,
        convert_to="neuralnetwork",
        minimum_deployment_target=ct.target.iOS14,
    )
    if model.get_spec().WhichOneof("Type") != "neuralNetwork":
        raise RuntimeError("Core ML conversion did not produce a neural-network model.")
    model.author = "Handwriting Synthesis"
    model.short_description = "One recurrent attention/GMM step for offline handwriting synthesis."
    model.version = "1.0"
    model.input_description["stroke"] = "Previous sampled x/y offset and pen state."
    model.input_description["chars"] = "Encoded text padded to 120 characters."
    model.input_description["chars_len"] = "Encoded text length including terminator."
    model.output_description["gmm_params"] = "Raw 20-component GMM parameters."
    model.user_defined_metadata["architecture"] = "3x400 LSTM with Graves attention"
    model.user_defined_metadata["source_checkpoint"] = checkpoint_label(checkpoint)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    model.save(output)
    return model


def checkpoint_label(checkpoint: Path) -> str:
    try:
        return str(checkpoint.relative_to(BASE_DIR))
    except ValueError:
        return checkpoint.name


def verification_inputs() -> dict[str, np.ndarray]:
    """Create deterministic, non-trivial values for parity testing."""

    rng = np.random.default_rng(20260927)
    encoded = np.zeros((1, 120), dtype=np.int32)
    encoded[0, :6] = np.array([24, 48, 55, 55, 58, 0], dtype=np.int32)
    values = {
        "stroke": rng.normal(0, 0.2, (1, 3)).astype(np.float32),
        "chars": encoded,
        "chars_len": np.array([6], dtype=np.int32),
        "kappa": np.abs(rng.normal(0, 0.05, (1, 10))).astype(np.float32),
        "window": rng.normal(0, 0.05, (1, 73)).astype(np.float32),
    }
    values.update({name: rng.normal(0, 0.05, (1, 400)).astype(np.float32) for name in STATE_NAMES})
    return values


def verify_numerical_parity(
    graph_def: tf.GraphDef,
    model_path: Path,
    *,
    tolerance: float = 2e-5,
) -> float:
    """Compare every Core ML output to TensorFlow and return worst max error."""

    import coremltools as ct

    inputs = verification_inputs()
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        with tf.Session(graph=graph) as session:
            expected = session.run(
                [graph.get_tensor_by_name(f"{name}:0") for name in OUTPUT_NAMES],
                feed_dict={
                    graph.get_tensor_by_name(f"{name}:0"): value for name, value in inputs.items()
                },
            )

    # Reloading exercises the saved package and the macOS Core ML runtime.
    reloaded = ct.models.MLModel(
        str(model_path),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    actual = reloaded.predict(inputs)

    worst_error = 0.0
    for name, expected_value in zip(OUTPUT_NAMES, expected, strict=True):
        max_error = float(np.max(np.abs(expected_value - np.asarray(actual[name]))))
        worst_error = max(worst_error, max_error)
        np.testing.assert_allclose(
            actual[name],
            expected_value,
            rtol=tolerance,
            atol=tolerance,
            err_msg=f"Core ML output {name} differs from TensorFlow",
        )
    return worst_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Export without comparing Core ML predictions to TensorFlow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()

    print(f"Restoring {checkpoint}")
    graph_def = build_and_freeze_step(checkpoint)
    print(f"Frozen step graph: {len(graph_def.node)} nodes")
    convert_to_coreml(graph_def, output, checkpoint)
    print(f"Saved Core ML model: {output}")

    if not args.skip_verify:
        worst_error = verify_numerical_parity(graph_def, output)
        print(f"TensorFlow/Core ML parity passed; worst max error: {worst_error:.9g}")


if __name__ == "__main__":
    main()
