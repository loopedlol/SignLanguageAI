# SignLanguageAI

I'm experimenting with recognizing individual Korean Sign Language signs from webcam video. Instead of looking at one still image, the program tracks how the hands, face, and body move over a short sequence.

<a id="pipeline"></a>
## How it works

MediaPipe tracks points on the person in each frame. A neural network then uses those movement sequences to predict a sign label.

The repository includes scripts to record examples, prepare the data, train a model, check its predictions, and try it with a webcam. It is an isolated-sign recognition project, not a translator for full conversations.

**Built with:** Python, MediaPipe, PyTorch, and OpenCV.

<a id="evaluation"></a>
## Current limits

Training recordings and trained sign-recognition models are not included, so this is not a ready-to-use demo. I don't have an independently verified accuracy result to report here.

The evaluation script uses the full normalized dataset by default, which can include examples used in training. Testing on new people or recording sessions requires a separate test dataset.

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
