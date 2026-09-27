from __future__ import annotations

from xml.etree import ElementTree

import pytest

pytest.importorskip("tensorflow", reason="TensorFlow is required for the model smoke test")

from demo import Hand


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
