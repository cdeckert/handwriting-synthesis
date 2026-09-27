![Handwriting Synthesis](img/banner.svg)

# Handwriting Synthesis

Generate convincing handwriting as SVG from plain text. This project implements the synthesis model from Alex Graves' paper [Generating Sequences with Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) and includes a pretrained checkpoint, 13 writing styles, a Python API, and a responsive web app.

> The neural network intentionally runs through TensorFlow's v1 compatibility layer so the original pretrained checkpoint remains usable. The application and tooling around it use current Python, Flask, React, TypeScript, and Vite conventions.

## Quick start with Docker

Docker builds the React app and serves it with the Python API:

```bash
docker build -t handwriting-synthesis .
docker run --rm -p 5000:5000 handwriting-synthesis
```

Open <http://localhost:5000>. The model loads on the first preview or download request, so that first request takes longer than subsequent ones.

## Local development

Requirements:

- Python 3.13
- Node.js 22.13 or newer
- npm

Create a Python environment and install the backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python web_app.py --host 127.0.0.1 --port 5000
```

In a second terminal, start the front end:

```bash
cd web_app
npm ci
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` requests to Flask on port 5000.

Before opening a pull request, run:

```bash
pytest
cd web_app && npm run check && npm audit --audit-level=high
```

## Python API

```python
from demo import Hand

hand = Hand()
hand.write(
    filename="handwriting.svg",
    lines=["Now this is a story all about how", "My life got flipped upside down"],
    biases=[0.75, 0.75],
    styles=[9, 9],
    stroke_colors=["#172554", "#172554"],
    stroke_widths=[2, 2],
    alignment="left",
)
```

Each line can contain up to 75 supported characters. `biases` control neatness, while `styles` select one of the included samples numbered 0 through 12.

## HTTP API

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | `GET` | Readiness information without loading the model |
| `/api/styles` | `GET` | Available handwriting styles |
| `/api/preview` | `POST` | Generate SVG inside a JSON response |
| `/api/generate` | `POST` | Download generated SVG |
| `/api/render` | `POST` | Generate platform-neutral vector paths for native clients |

Generation requests accept JSON in this shape:

```json
{
  "text": "Hello world",
  "style": 4,
  "alignment": "center"
}
```

## Native iPhone app

[`ios/HandwritingStudio`](ios/HandwritingStudio) contains a fully offline native SwiftUI app for iOS 17 and newer. Core ML runs the converted recurrent/attention network, Swift performs the GMM sampling and stroke layout, SwiftUI Canvas/Core Graphics draws the result, and the native share sheet exports SVG. No server, web view, or JavaScript UI is required.

Open the Xcode project and run it on a simulator or iPhone:

```bash
open ios/HandwritingStudio/HandwritingStudio.xcodeproj
```

The app bundles the Core ML model and all 13 style primers. Its native style selector gives each style a descriptive name and renders a real preview from its primer data. Users can choose common paper sizes (A5 through A3, Letter, and Legal) or screen canvases, switch orientation, and adjust the writing size. Text wraps automatically at word boundaries, and the SVG retains the selected print or pixel dimensions. While generating, the interface shows percentage progress based on completed Core ML inference steps. See [`docs/coreml-conversion.md`](docs/coreml-conversion.md) for conversion and parity details.

## Production notes

- The production container runs as an unprivileged user.
- A single worker processes generation requests because the restored TensorFlow session is stateful and memory-heavy.
- Input size, line count, line length, style, and alignment are validated before model execution.
- The `/api/health` endpoint is suitable for container health checks.

## Training data

The pretrained model is included. To train a new model, follow the instructions in [`data/raw/readme.md`](data/raw/readme.md) and place the IAM dataset in the ignored `data/raw` directories.

## Examples

![Generated handwriting sample](img/usage_demo.svg)

Additional generated samples are available in [`img/`](img/).

## Acknowledgements

This project began as Sean Vasquez's reference implementation of Graves' handwriting-synthesis experiments. Its model structure and bundled checkpoint remain compatible with that work.
