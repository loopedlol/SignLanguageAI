# SignLanguageAI / Visual assets

[← Project](../README.md) · [Workflow guide](WORKFLOW_GUIDE.md)

The hero uses a schematic hand with repeated outlines to suggest time; it is not a detected pose or model output. The system diagram reflects the code's landmark, normalization, temporal-window, training, and inference boundaries. Both have light/dark variants.

## Replace the demo placeholder

The paired `assets/portfolio/demo-placeholder-{light,dark}.svg` files deliberately label an unfilled capture slot. No webcam recording, trained checkpoint, or recognition result is fabricated.

Capture a real still or short clip from `src/predict_webcam.py` with:

- a consenting signer and a neutral background without identifying information;
- a visible landmark overlay, predicted label, and confidence;
- the tested vocabulary, model/checkpoint provenance, and recording conditions in the caption;
- successes and representative limitations clearly distinguished.

Use a 1200 × 675 still, or a similarly framed short recording. A real image can replace both placeholder variants via a normal `<img>` element. It need not be recolored for dark mode. Keep explanatory text in Markdown as well as the image caption.

## Social card

`assets/portfolio/social-preview.png` is a 1280 × 640 PNG derived from the adjacent editable SVG. It represents project identity, not demonstrated model performance.
