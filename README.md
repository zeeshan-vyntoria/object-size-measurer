# Object Size Measurer

Estimates real-world dimensions (width x height) of objects in a photo using a
reference object of known size — no depth sensor or special camera required.

## How it works

1. Take a photo top-down (bird's-eye view works best) of your objects on a
   plain, contrasting background (e.g. white paper / dark table), with good,
   even lighting and no shadows crossing objects.
2. Place a reference object of **known width** as the **leftmost** object in
   the frame — a credit/ID card (3.37in / 8.56cm wide) is convenient.
3. Run the script; it finds object outlines, calibrates pixel-to-unit scale
   from the reference object, and prints/draws the size of every other object.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
python measure.py --image photo.jpg --ref-width 3.37 --unit in
```

- `--ref-width`: known width of the reference (leftmost) object, in whatever
  unit you want output in (inches, cm, mm — just be consistent).
- `--unit`: label only, for display (e.g. `in`, `cm`).
- `--min-area`: raise this if small background specks are being picked up as
  objects (default 800 px²).

Output: dimensions printed to the console, plus an annotated copy of the image
saved as `<name>_measured.<ext>` with bounding boxes and measurements drawn on.

## Accuracy notes

- Works best top-down with the camera roughly parallel to the surface;
  perspective distortion (angled shots) will skew results.
- All objects should be roughly the same distance from the camera (same
  plane) since there's no depth information — an object closer to the camera
  will appear larger than its true size relative to the reference.
- If detection fails ("no object contours found"), improve contrast between
  objects and background, or lower `--min-area`.

## Limitations / next steps

This first version assumes a controlled, top-down photo. If you need it to
work with arbitrary casual photos (angled shots, cluttered backgrounds, no
visible reference object), that requires a different approach — e.g. a
depth-estimation model or a vision-LLM rough estimate — with less accuracy.
Let me know if you want that mode added.

## Test UI

A small web UI wraps `measure.py` for interactive testing: a Flask API
(`server.py`) plus a React frontend (`frontend/`, built with Vite).

### Backend

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```

Runs on `http://127.0.0.1:5050`.

The included MediaPipe pose model (~5.5MB) detects people wearing shirts so
those photos can be handled differently from flat-laid garments.

### Frontend

Install Node.js, then from the repository root run:

```powershell
cd frontend
npm ci
npm run dev
```

Opens on `http://localhost:5173`. Two modes are available: **Clothing**
(auto-routes between three photo types — flat-laid with a reference object
gets real calibrated measurements; flat-laid alone, or worn by a model,
both fall back to a rough size-chart guess, the worn case using detected
body pose landmarks since there's no flat garment shape to trace) and
**Generic object** (bounding-box width/height for every contour found).
