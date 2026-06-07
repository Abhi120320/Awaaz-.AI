"""
Hand Gesture → Voice

Hold a gesture steady ~1 second to speak. Keys: Q quit, H help, SPACE speak now.
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from collections import deque
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from gesture_classifier import ACTIVE_GESTURES, GESTURE_LABELS
from gesture_engine import GestureEngine
from hand_visualizer import HandVisualizer

IS_MACOS = platform.system() == "Darwin"

COOLDOWN = 1.0
STABLE_WINDOW = 10
STABLE_NEED = 5
MIN_CONF = 0.48

_say_proc: Optional[subprocess.Popen] = None
_show_help = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 220, 130)
GRAY = (130, 130, 130)
RED = (80, 80, 255)
YELLOW = (80, 220, 255)
CYAN = (255, 220, 80)

HELP_LINES = [
    "Thumbs UP=Yes  Fist=Help  Peace=Peace  I-love-you",
    "OK sign=OK  Crossed fingers=Good luck  Spread R=Hello  Spread L=Goodbye",
    "Point UP=Attention  |  Point EAR (side)=Can't hear  MOUTH=Hungry",
    "Heart hands (two hands)=Thank you  |  Hold ~1 sec",
]


def speak(text: str) -> None:
    global _say_proc
    if not text:
        return
    if _say_proc is not None and _say_proc.poll() is None:
        _say_proc.terminate()
        _say_proc = None

    def _run() -> None:
        global _say_proc
        try:
            if IS_MACOS:
                _say_proc = subprocess.Popen(["say", "-r", "165", text])
                _say_proc.wait()
        finally:
            _say_proc = None

    threading.Thread(target=_run, daemon=True).start()


def stable_gesture(buf: deque[Optional[str]]) -> Optional[str]:
    if len(buf) < STABLE_NEED:
        return None
    tail = list(buf)[-STABLE_NEED:]
    if not all(g is not None and g == tail[0] for g in tail):
        return None
    return tail[0]


def draw_ui(
    frame,
    hand_ok: bool,
    stable: Optional[str],
    last_spoken: str,
    status: str,
    cooldown_frac: float,
    show_help: bool,
    conf: float,
    raw: str,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 72), (15, 15, 15), -1)
    cv2.putText(frame, "Q=quit  H=help  SPACE=speak", (10, 22), FONT, 0.48, GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, status, (10, 46), FONT, 0.45, GREEN if hand_ok else RED, 1, cv2.LINE_AA)
    if raw:
        cv2.putText(frame, f"Detect: {raw} {int(conf * 100)}%", (10, 66), FONT, 0.38, CYAN, 1, cv2.LINE_AA)

    if show_help:
        cv2.rectangle(frame, (0, 72), (w, 72 + len(HELP_LINES) * 22 + 10), (20, 20, 20), -1)
        for i, line in enumerate(HELP_LINES):
            cv2.putText(frame, line, (10, 94 + i * 22), FONT, 0.42, YELLOW, 1, cv2.LINE_AA)

    if not hand_ok:
        cv2.putText(
            frame, "Show hand to camera (two hands for heart)",
            (10, h - 30), FONT, 0.65, RED, 2, cv2.LINE_AA,
        )
        return

    if stable and stable in GESTURE_LABELS:
        label, phrase = GESTURE_LABELS[stable]
        cv2.rectangle(frame, (0, h - 90), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, label, (16, h - 50), FONT, 0.95, GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, f'Will say: "{phrase}"', (16, h - 22), FONT, 0.55, GRAY, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Hold gesture steady...", (16, h - 30), FONT, 0.65, GRAY, 1, cv2.LINE_AA)

    if last_spoken:
        cv2.putText(
            frame, f'Said: "{last_spoken}"',
            (w - 340, h - 30), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

    bar = int(w * cooldown_frac)
    cv2.rectangle(frame, (0, h - 4), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, h - 4), (bar, h), GREEN, -1)


def open_camera() -> cv2.VideoCapture:
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if IS_MACOS else [cv2.CAP_ANY]
    for idx in (0, 1, 2):
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(idx)
            if cap.isOpened() and cap.read()[0]:
                print(f"Camera OK (index {idx})")
                return cap
            cap.release()
    return cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION if IS_MACOS else cv2.CAP_ANY)


def main() -> None:
    global _show_help

    cap = open_camera()
    if not cap.isOpened():
        print("\nEnable Camera for Terminal/Cursor in System Settings → Privacy → Camera\n")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    engine = GestureEngine()
    visualizer = HandVisualizer()

    buf: deque[Optional[str]] = deque(maxlen=STABLE_WINDOW)
    last_time = 0.0
    last_spoken = ""
    last_gesture_spoken: Optional[str] = None
    prev_detected: Optional[str] = None
    status = "Waiting for hand..."

    print("\n=== Supported gestures ===")
    for g in ACTIVE_GESTURES:
        label, phrase = GESTURE_LABELS[g]
        print(f"  {label:22} → {phrase}")
    print("Hold ~1 second. Spread: right hand=Hello, left hand=Goodbye.\n")

    def do_speak(text: str, gesture: str) -> None:
        nonlocal last_spoken, last_time, last_gesture_spoken, status
        if gesture == last_gesture_spoken:
            return
        speak(text)
        last_spoken = text
        last_gesture_spoken = gesture
        last_time = time.time()
        status = f"Spoke: {text}"
        print(f"✓ {GESTURE_LABELS[gesture][0]} → \"{text}\"")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fr = engine.process(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        hand_ok = bool(fr.landmarks)
        detected = fr.gesture if fr.confidence >= MIN_CONF else None

        if detected != prev_detected:
            if detected is not None:
                buf.clear()
            elif not hand_ok:
                buf.clear()
                last_gesture_spoken = None
            prev_detected = detected

        if hand_ok and detected:
            label = GESTURE_LABELS[detected][0]
            status = f"Seeing: {label}  ({int(fr.confidence * 100)}%) [{fr.source}]"
            buf.append(detected)
        else:
            if not hand_ok:
                status = "No hand — show palm to camera"
                buf.clear()
                last_gesture_spoken = None
                prev_detected = None
            elif fr.gesture and fr.confidence < MIN_CONF:
                status = f"Uncertain ({int(fr.confidence * 100)}%) — hold steadier"
            else:
                status = "No match — try thumbs up, fist, peace, spread hand"
            buf.append(None)

        stable = stable_gesture(buf)

        for hl in fr.landmarks:
            visualizer.draw(frame, hl, is_stable=bool(stable), gesture=stable)

        now = time.time()
        if stable and stable != last_gesture_spoken and (now - last_time) >= COOLDOWN:
            phrase = GESTURE_LABELS[stable][1]
            if not phrase.endswith("."):
                phrase += "."
            do_speak(phrase, stable)

        draw_ui(
            frame, hand_ok, stable, last_spoken, status,
            min(1.0, (now - last_time) / COOLDOWN), _show_help,
            fr.confidence, fr.raw_label,
        )
        cv2.imshow("Hand Gesture -> Voice", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("h"):
            _show_help = not _show_help
        if key == ord(" ") and stable:
            last_gesture_spoken = None
            phrase = GESTURE_LABELS[stable][1]
            if not phrase.endswith("."):
                phrase += "."
            do_speak(phrase, stable)

    cap.release()
    cv2.destroyAllWindows()
    engine.close()


if __name__ == "__main__":
    main()
