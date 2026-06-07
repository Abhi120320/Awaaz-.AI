# Hand Gesture → Voice

Real-time hand gesture recognition with spoken output. Uses [MediaPipe](https://developers.google.com/mediapipe) for hand tracking and custom finger-geometry checks for reliable detection.

## Requirements

- Python 3.9+
- Webcam
- macOS (uses built-in `say` for speech) or Linux/Windows (speech requires macOS currently)

## Quick start

```bash
git clone <your-repo-url>
cd HandGestureVoice
chmod +x run.sh
./run.sh
```

On first run, `run.sh` creates a virtual environment, installs dependencies, and downloads the MediaPipe gesture model (~7 MB).

### Camera permission (macOS)

Grant camera access to Terminal or Cursor: **System Settings → Privacy & Security → Camera**.

## Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **H** | Toggle on-screen help |
| **Space** | Speak current gesture immediately |

Hold each gesture steady for about **1 second** before it speaks.

## Gestures

| Gesture | Says |
|---------|------|
| Thumbs up | Yes |
| Thumbs down | No |
| Fist | Help me |
| Peace sign | Peace |
| I love you | I love you |
| OK sign | OK |
| Crossed fingers | Good luck |
| Spread hand (right) | Hello |
| Spread hand (left) | Goodbye |
| Point up | Attention |
| Point at ear (side of face) | I can't hear |
| Point at mouth | I'm hungry |
| Heart hands (two hands) | Thank you |

**Spread hand** = fingers fanned extra wide (jazz-hands), not a normal flat palm.

## Project structure

```
HandGestureVoice/
├── main.py              # Camera loop, UI, speech
├── gesture_engine.py    # MediaPipe + detection pipeline
├── gesture_classifier.py # Finger geometry & gesture labels
├── hand_visualizer.py   # Animated hand overlay
├── run.sh               # Setup & launch script
└── requirements.txt
```

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## License

MIT
