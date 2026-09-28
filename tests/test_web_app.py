from __future__ import annotations

import pytest

import web_app


@pytest.fixture()
def client():
    web_app.app.config.update(TESTING=True)
    with web_app.app.test_client() as test_client:
        yield test_client


def test_health_does_not_load_the_model(client):
    web_app.get_hand.cache_clear()

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json == {"model_loaded": False, "status": "ok", "styles": 13}
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_styles_are_discovered_in_numeric_order(client):
    response = client.get("/api/styles")

    assert response.status_code == 200
    assert [style["id"] for style in response.json["styles"]] == list(range(13))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"text": ""}, "Please enter some text."),
        ({"text": "hello", "style": "unknown"}, "Invalid style selected."),
        ({"text": "hello", "style": 99}, "Selected style is not available."),
        ({"text": "hello", "alignment": "right"}, "Invalid alignment selected."),
        ({"text": "hello", "fontSize": 17}, "Font size must be a number between 18 and 72."),
        ({"text": "hello", "fontSize": 73}, "Font size must be a number between 18 and 72."),
        ({"text": "hello", "fontSize": "large"}, "Font size must be a number between 18 and 72."),
        ({"text": "hello", "fontSize": True}, "Font size must be a number between 18 and 72."),
        ({"text": "x" * 76}, "the limit is 75"),
        ({"text": "\n".join(["x"] * 13)}, "no more than 12 lines"),
    ],
)
def test_preview_rejects_invalid_input(client, payload, message):
    response = client.post("/api/preview", json=payload)

    assert response.status_code == 400
    assert message in response.json["error"]


def test_generate_returns_an_svg_download(client, monkeypatch):
    generated = b'<svg xmlns="http://www.w3.org/2000/svg" />'
    calls = []

    def fake_generate(lines, *, style=None, alignment="center", font_size=36):
        calls.append((lines, style, alignment, font_size))
        return generated

    monkeypatch.setattr(web_app, "_generate_svg", fake_generate)

    response = client.post(
        "/api/generate",
        json={"text": "Hello\nworld", "style": 2, "alignment": "left", "fontSize": 48},
    )

    assert response.status_code == 200
    assert response.data == generated
    assert response.mimetype == "image/svg+xml"
    assert response.headers["Content-Disposition"] == "attachment; filename=handwriting.svg"
    assert calls == [(["Hello", "world"], 2, "left", 48.0)]


def test_render_returns_native_vector_document(client, monkeypatch):
    document = {
        "width": 1000.0,
        "height": 120.0,
        "backgroundColor": "#FFFFFF",
        "paths": [
            {
                "strokeColor": "black",
                "lineWidth": 2.0,
                "points": [{"x": 12.0, "y": 34.0, "move": True}],
            }
        ],
    }
    calls = []

    def fake_render(lines, *, style=None, alignment="center", font_size=36):
        calls.append((lines, style, alignment, font_size))
        return document

    monkeypatch.setattr(web_app, "_render_document", fake_render)

    response = client.post(
        "/api/render",
        json={"text": "Native", "style": 7, "alignment": "center"},
    )

    assert response.status_code == 200
    assert response.json == document
    assert response.headers["Cache-Control"] == "no-store"
    assert calls == [(["Native"], 7, "center", 36.0)]
