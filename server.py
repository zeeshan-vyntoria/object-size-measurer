"""
Local API for the Object Size Measurer test UI.

Usage:
    python server.py
Then open frontend/index.html in a browser.
"""

import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from measure import measure_objects, measure_shirt_auto

app = Flask(__name__)
CORS(app)


def _round(v, digits=2):
    """
    Round a numeric value for the JSON response, unwrapping numpy scalar
    types first (e.g. np.float32/np.int64 from OpenCV/numpy computations) —
    Python's json module can serialize a plain float or np.float64 (which
    subclasses float) but raises TypeError on other numpy scalar types like
    np.float32, which previously crashed these endpoints with a 500 instead
    of returning a real error.
    """
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(v, digits)
    return v


@app.post("/api/measure")
def measure():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    try:
        ref_width = float(request.form.get("ref_width", 3.37))
        min_area = float(request.form.get("min_area", 800))
    except ValueError:
        return jsonify({"error": "ref_width and min_area must be numbers"}), 400
    unit = request.form.get("unit", "in")

    suffix = Path(image_file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        results, output = measure_objects(tmp_path, ref_width, unit, min_area)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    ok, buf = cv2.imencode(".jpg", output)
    annotated_b64 = base64.b64encode(buf).decode("ascii") if ok else None

    return jsonify(
        {
            "results": [
                {"label": r["label"], "width": _round(r["width"]), "height": _round(r["height"])}
                for r in results
            ],
            "unit": unit,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
        }
    )


@app.post("/api/measure-clothing")
def measure_clothing_endpoint():
    """
    Unified clothing endpoint: automatically measures accurately when a
    reference object is also in the photo, and falls back to a rough
    no-reference size-chart guess when it's just the shirt alone. The
    response's "accuracy" field tells the caller which path was used.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    try:
        ref_width = float(request.form.get("ref_width", 3.37))
        min_area = float(request.form.get("min_area", 800))
        shoulder_pct = float(request.form.get("shoulder_pct", 0.16))
        chest_pct = float(request.form.get("chest_pct", 0.35))
        waist_pct = float(request.form.get("waist_pct", 0.7))
    except ValueError:
        return jsonify({"error": "ref_width, min_area, shoulder_pct, chest_pct, waist_pct must be numbers"}), 400
    unit = request.form.get("unit", "in")

    suffix = Path(image_file.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result, output = measure_shirt_auto(
            tmp_path, ref_width, unit, min_area, shoulder_pct, chest_pct, waist_pct
        )
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    ok, buf = cv2.imencode(".jpg", output)
    annotated_b64 = base64.b64encode(buf).decode("ascii") if ok else None

    return jsonify(
        {
            **{k: _round(v) for k, v in result.items()},
            "unit": unit,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
        }
    )


if __name__ == "__main__":
    app.run(port=5050, debug=True, use_reloader=False)
