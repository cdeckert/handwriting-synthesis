"""Simple Flask application to render handwriting samples as SVG files."""
import os
import uuid
from pathlib import Path

from flask import Flask, flash, render_template, request, send_from_directory, url_for

from demo import Hand

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "handwriting-secret")

OUTPUT_DIR = Path("web_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_hand_instance = None


def get_hand():
    """Lazily construct the Hand model to avoid loading it multiple times."""
    global _hand_instance
    if _hand_instance is None:
        _hand_instance = Hand()
    return _hand_instance


@app.route("/", methods=["GET", "POST"])
def index():
    svg_url = None
    text = request.form.get("text", "") if request.method == "POST" else ""
    if request.method == "POST":
        normalized_text = text.replace("\r", "")
        if not normalized_text.strip():
            flash("Please enter some text to render.")
        else:
            lines = normalized_text.split("\n")
            filename = OUTPUT_DIR / f"{uuid.uuid4().hex}.svg"
            try:
                get_hand().write(str(filename), lines)
            except ValueError as exc:
                flash(str(exc))
            else:
                svg_url = url_for("generated_svg", filename=filename.name)
    return render_template("index.html", svg_url=svg_url, text=text)


@app.route("/generated/<path:filename>")
def generated_svg(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(debug=True)
