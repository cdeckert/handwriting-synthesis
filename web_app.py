"""Simple Flask web UI for handwriting synthesis."""

from __future__ import annotations

import argparse
import os
import tempfile
from functools import lru_cache
from typing import Iterable, List

from flask import Flask, Response, render_template_string, request

from demo import Hand


app = Flask(__name__)

TEMPLATE = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Handwriting Synthesis</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f7f7f7; }
      header { background: #222; color: #fff; padding: 1.5rem; text-align: center; }
      main { max-width: 720px; margin: 2rem auto; background: #fff; padding: 2rem; box-shadow: 0 2px 6px rgba(0,0,0,.1); border-radius: 8px; }
      textarea { width: 100%; min-height: 240px; font-size: 1rem; padding: .75rem; border: 1px solid #ccc; border-radius: 4px; resize: vertical; }
      button { margin-top: 1rem; background: #0b5ed7; color: #fff; border: none; padding: .75rem 1.5rem; font-size: 1rem; border-radius: 4px; cursor: pointer; }
      button:hover { background: #0a53be; }
      .error { margin-top: 1rem; color: #b00020; font-weight: bold; }
      p.help { color: #555; }
    </style>
  </head>
  <body>
    <header>
      <h1>Handwriting Synthesis</h1>
      <p>Enter text below to generate an SVG rendition.</p>
    </header>
    <main>
      <form method=\"post\">
        <p class=\"help\">Each line may contain up to 75 characters and the supported character set is limited to A-Z, a-z, numbers, and basic punctuation.</p>
        <label for=\"text\" class=\"help\">Text to render</label>
        <textarea id=\"text\" name=\"text\" placeholder=\"Type or paste your message here...\" required>{{ text }}</textarea>
        <button type=\"submit\">Generate SVG</button>
        {% if error %}<p class=\"error\">{{ error }}</p>{% endif %}
      </form>
    </main>
  </body>
</html>"""


@lru_cache(maxsize=1)
def get_hand() -> Hand:
    """Lazily create the Hand model instance."""

    return Hand()


def _generate_svg(lines: Iterable[str]) -> bytes:
    """Generate SVG bytes for the provided lines."""

    hand = get_hand()

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_file:
        temp_name = tmp_file.name

    try:
        hand.write(filename=temp_name, lines=lines)
        with open(temp_name, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the form and handle SVG generation requests."""

    if request.method == "POST":
        text = request.form.get("text", "")
        if not text.strip():
            return render_template_string(TEMPLATE, error="Please enter some text.", text=text)

        lines = [line.rstrip() for line in text.splitlines()]

        try:
            svg_bytes = _generate_svg(lines)
        except ValueError as exc:
            return render_template_string(TEMPLATE, error=str(exc), text=text)

        headers = {"Content-Disposition": "attachment; filename=handwriting.svg"}
        return Response(svg_bytes, mimetype="image/svg+xml", headers=headers)

    return render_template_string(TEMPLATE, error=None, text="")


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the web server entry point."""

    parser = argparse.ArgumentParser(description="Run the handwriting synthesis web UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind the development server to.")
    parser.add_argument(
        "--port",
        default=int(os.environ.get("PORT", 5000)),
        type=int,
        help="Port to listen on (defaults to 5000 or the PORT environment variable).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (disabled by default to avoid the auto reloader).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for running the development server directly."""

    args = _parse_args(argv)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=args.debug)


if __name__ == "__main__":
    main()

