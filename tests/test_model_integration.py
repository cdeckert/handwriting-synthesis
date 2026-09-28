from __future__ import annotations

from xml.etree import ElementTree

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow is required for the model smoke test")

from demo import Hand


def test_layout_scales_coordinates_and_stroke_width_with_font_size():
    offsets = np.array(
        [
            [1.0, 0.2, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, -0.1, 0.0],
            [1.0, 0.3, 0.0],
            [1.0, -0.2, 0.0],
            [1.0, 0.1, 0.0],
            [1.0, 0.2, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    hand = Hand.__new__(Hand)

    default = hand._layout([offsets], ["Scale"], font_size=36)
    large = hand._layout([offsets], ["Scale"], font_size=72)

    default_points = default["paths"][0]["points"]
    large_points = large["paths"][0]["points"]
    default_delta_x = default_points[-1]["x"] - default_points[0]["x"]
    large_delta_x = large_points[-1]["x"] - large_points[0]["x"]

    assert large["height"] == default["height"] * 2
    assert large["paths"][0]["lineWidth"] == default["paths"][0]["lineWidth"] * 2
    assert large_delta_x == pytest.approx(default_delta_x * 2)


def test_pretrained_checkpoint_generates_valid_svg(tmp_path):
    output = tmp_path / "smoke-test.svg"
    hand = Hand()

    hand.write(
        filename=output,
        lines=["Hello"],
        biases=[0.75],
        styles=[0],
    )

    root = ElementTree.parse(output).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert output.stat().st_size > 1_000

    native_document = hand.render(lines=["Native"], biases=[0.75], styles=[0])
    assert native_document["width"] == 1000.0
    assert native_document["height"] == 120.0
    assert len(native_document["paths"]) == 1
    assert len(native_document["paths"][0]["points"]) > 100
