"""MediaPipe gesture recognition with custom finger-geometry fallbacks."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.processors import classifier_options

from gesture_classifier import (
    classify_static,
    detect_body_point_gestures,
    detect_extra_gestures,
    detect_heart_two_hands,
    is_core_landmark_gesture,
    verify_gesture,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(_DIR, "gesture_recognizer.task")
GESTURE_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

MP_MAP: dict[str, str] = {
    "Thumb_Up": "thumbs_up",
    "Thumb_Down": "thumbs_down",
    "Closed_Fist": "fist",
    "Pointing_Up": "pointing",
    "Victory": "peace",
    "ILoveYou": "love_you",
}

STANDARD_GESTURES = frozenset({
    "thumbs_up", "thumbs_down", "fist", "peace", "love_you",
})

MIN_SCORE = 0.48


@dataclass
class _Pt:
    x: float
    y: float
    z: float = 0.0


@dataclass
class FrameResult:
    landmarks: list[list[Any]]
    handedness: list[str]
    gesture: Optional[str]
    confidence: float
    source: str
    raw_label: str = ""


def ensure_gesture_model() -> str:
    if not os.path.exists(GESTURE_MODEL):
        print("Downloading gesture model (~7MB)...")
        urllib.request.urlretrieve(GESTURE_URL, GESTURE_MODEL)
        print("Done.\n")
    return GESTURE_MODEL


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.55) -> None:
        self._alpha = alpha
        self._cache: dict[int, list[tuple[float, float, float]]] = {}

    def reset(self) -> None:
        self._cache.clear()

    def apply(self, hand_idx: int, landmarks: list[Any]) -> list[_Pt]:
        pts = [(l.x, l.y, getattr(l, "z", 0.0)) for l in landmarks]
        if hand_idx not in self._cache:
            self._cache[hand_idx] = pts
            return [_Pt(x=p[0], y=p[1], z=p[2]) for p in pts]

        prev = self._cache[hand_idx]
        a = self._alpha
        out = [
            _Pt(
                x=prev[i][0] * (1 - a) + pts[i][0] * a,
                y=prev[i][1] * (1 - a) + pts[i][1] * a,
                z=prev[i][2] * (1 - a) + pts[i][2] * a,
            )
            for i in range(len(pts))
        ]
        self._cache[hand_idx] = [(p.x, p.y, p.z) for p in out]
        return out


class GestureEngine:
    def __init__(self) -> None:
        model = ensure_gesture_model()
        self._recognizer = mp_vision.GestureRecognizer.create_from_options(
            mp_vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                canned_gesture_classifier_options=classifier_options.ClassifierOptions(
                    score_threshold=0.35,
                ),
            )
        )
        self._smoother = LandmarkSmoother()
        self._ts_ms = 0

    def close(self) -> None:
        self._recognizer.close()

    def _pick_mp(
        self, result: Any, hand_idx: int, lm: list[Any], handedness: str
    ) -> tuple[Optional[str], float, str]:
        if not result.gestures or hand_idx >= len(result.gestures):
            return None, 0.0, ""
        cats = sorted(result.gestures[hand_idx], key=lambda c: c.score, reverse=True)
        for cat in cats:
            if cat.category_name in ("None", "") or cat.score < MIN_SCORE:
                continue
            mapped = MP_MAP.get(cat.category_name)
            if not mapped:
                continue
            if verify_gesture(mapped, lm, handedness):
                return mapped, cat.score, cat.category_name
        return None, 0.0, ""

    def process(self, mp_image: mp.Image) -> FrameResult:
        self._ts_ms += 33
        result = self._recognizer.recognize_for_video(mp_image, self._ts_ms)

        if not result.hand_landmarks:
            self._smoother.reset()
            return FrameResult([], [], None, 0.0, "none")

        smoothed = [
            self._smoother.apply(i, hl) for i, hl in enumerate(result.hand_landmarks)
        ]
        handedness = [
            (
                result.handedness[i][0].category_name
                if result.handedness and i < len(result.handedness) and result.handedness[i]
                else "Right"
            )
            for i in range(len(smoothed))
        ]

        best: tuple[Optional[str], float, str] = (None, 0.0, "")

        if len(smoothed) >= 2:
            heart = detect_heart_two_hands(smoothed)
            if heart and heart.name:
                return FrameResult(
                    smoothed, handedness, heart.name, heart.confidence, "two_hand", "Heart"
                )

        for i, lm in enumerate(smoothed):
            extra = detect_extra_gestures(lm, handedness[i])
            if extra and extra.name:
                return FrameResult(
                    smoothed, handedness, extra.name, extra.confidence, "custom", extra.raw_pose or ""
                )

        for i, lm in enumerate(smoothed):
            mp_name, mp_score, raw = self._pick_mp(result, i, lm, handedness[i])
            if mp_name in STANDARD_GESTURES and mp_score >= MIN_SCORE:
                return FrameResult(
                    smoothed, handedness, mp_name, mp_score, "mediapipe", raw
                )
            if mp_name and mp_score > best[1]:
                best = (mp_name, mp_score, raw)

        for i, lm in enumerate(smoothed):
            if not is_core_landmark_gesture(lm, handedness[i]):
                continue
            custom = classify_static(lm, handedness[i])
            if custom.name in STANDARD_GESTURES and custom.confidence >= MIN_SCORE:
                return FrameResult(
                    smoothed, handedness, custom.name, custom.confidence, "custom", custom.name or ""
                )

        for i, lm in enumerate(smoothed):
            body = detect_body_point_gestures(lm, handedness[i])
            if body and body.name:
                return FrameResult(
                    smoothed, handedness, body.name, body.confidence, "body", body.name
                )

        if best[0]:
            return FrameResult(smoothed, handedness, best[0], best[1], "mediapipe", best[2])

        for i, lm in enumerate(smoothed):
            custom = classify_static(lm, handedness[i])
            if custom.name and custom.confidence > best[1]:
                best = (custom.name, custom.confidence, custom.name or "")

        if best[0]:
            return FrameResult(smoothed, handedness, best[0], best[1], "custom", best[2])

        return FrameResult(smoothed, handedness, None, 0.0, "none", "")
