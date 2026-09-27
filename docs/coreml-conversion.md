# Core ML conversion

## Result

The trained recurrent cell, Graves attention calculation, and GMM projection convert successfully to a Core ML `mlprogram` targeting iOS 17. The exported package is bundled in the native app as `HandwritingStep.mlpackage`.

The conversion is split at one recurrent step. Its explicit inputs and outputs are:

- Previous stroke: 3 float values
- Six LSTM hidden/cell states: 400 floats each
- Attention state: 10 kappa and 73 window values
- Encoded text: 120 integers plus its length
- Updated states, 120 attention weights, and 121 raw GMM parameters

The 121 GMM values represent 20 mixture weights, 40 standard-deviation logits, 20 correlations, 40 means, and one end-of-stroke probability logit.

The native app now completes the rest of inference in Swift. It loads the bundled style primers, advances the Core ML state for every input and generated point, samples the bivariate GMM, applies the original attention-based termination rule, smooths and aligns strokes, renders them with SwiftUI Canvas, and exports the result as SVG. No Python process or network connection is used by the iPhone app.

## Why the entire graph is not converted

The complete original TensorFlow inference graph was frozen and passed to Core ML Tools 9. The converter accepted the graph and its fixed input shapes, but rejected the graph with `Graph is not a DAG!`. The original generator implements sampling with nested TensorFlow 1 control-flow loops and TensorFlow Probability random-distribution operations, so the full inference graph is cyclic from the converter's perspective.

Separating the acyclic neural step avoids approximating or retraining the network. The Core ML result was executed on the local Core ML runtime with the same deterministic input state as TensorFlow. All ten outputs passed a `2e-5` relative and absolute tolerance; the observed worst maximum absolute error was approximately `2.9e-6`.

## Rebuild and verify

```bash
source .venv/bin/activate
pip install -r requirements-coreml.txt
python tools/export_coreml.py
```

The exporter restores the original checkpoint, freezes only inference weights, converts the step to float32 Core ML, reloads the saved package, and compares every output with TensorFlow.
