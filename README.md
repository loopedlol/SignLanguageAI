<a id="signlanguageai"></a>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/banner-dark.svg">
  <img src="docs/assets/readme/banner-light.svg" alt="SignLanguageAI — Recognizing individual signs over time." width="100%">
</picture>

I'm working on recognizing individual Korean Sign Language signs from webcam video. A sign involves movement, so the model looks at a sequence of frames instead of trying to classify one still image.

I've trained and tried a classifier locally, and the project is still ongoing. The training recordings and trained classifier are not included here, so cloning the repository doesn't give you a ready-to-use recognizer.

> [PLACEHOLDER — 10–15 second webcam recording showing a trained sign, the tracked landmarks, and the predicted label. Include an incorrect or uncertain prediction to show the current limits.]

<a id="pipeline"></a>
## How it works

MediaPipe tracks points on the hands, face, and body. I center those coordinates between the shoulders and scale them by shoulder width before training. This is meant to reduce differences caused by where someone stands in the frame.

A temporal convolutional network then looks for patterns across 30 frames and predicts a sign label. The surrounding scripts handle recording examples, inspecting missing detections, normalizing the data, training, and webcam predictions.

**Built with:** Python, MediaPipe, PyTorch, and OpenCV.

<a id="evaluation"></a>
## What still needs checking

This recognizes isolated signs rather than translating full conversations. I don't have an independently verified accuracy result to report.

The training script splits individual recordings into training and validation sets. The evaluation script uses the full normalized dataset by default, which can include training examples. A stronger test needs separate recordings from new people or sessions, rather than treating that default score as evidence of generalization.

<a id="start"></a>
## Try it

<details>
<summary>Setup requirements</summary>

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\activate` instead of the `source` command.

You also need a compatible MediaPipe Holistic Landmarker model at `models/holistic_landmarker.task` and a MediaPipe version that supports its Holistic Tasks API. Neither the detector model nor a trained sign classifier is included.

Check the detector setup with:

```bash
python src/holistic_health_check.py
```

The [workflow guide](docs/WORKFLOW_GUIDE.md) explains how to record examples for multiple sign labels, train the classifier, and run webcam predictions. It also covers evaluation settings and troubleshooting.

</details>
