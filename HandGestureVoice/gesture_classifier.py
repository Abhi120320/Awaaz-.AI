"""Finger-geometry gesture classification and labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

GESTURE_LABELS: dict[str, tuple[str, str]] = {
    "thumbs_up": ("Thumbs Up", "Yes"),
    "thumbs_down": ("Thumbs Down", "No"),
    "fist": ("Fist", "Help me"),
    "peace": ("Peace Sign", "Peace"),
    "love_you": ("Love You", "I love you"),
    "ok": ("OK Sign", "OK"),
    "crossed_fingers": ("Crossed Fingers", "Good luck"),
    "spread_hello": ("Spread Hand (right)", "Hello"),
    "spread_goodbye": ("Spread Hand (left)", "Goodbye"),
    "pointing": ("Pointing", "Attention"),
    "point_ear": ("Point to Ear", "I can't hear"),
    "hungry": ("Hungry", "I'm hungry"),
    "heart_hands": ("Heart Hands", "Thank you"),
}

ACTIVE_GESTURES: tuple[str, ...] = tuple(GESTURE_LABELS.keys())

FINGERS = (
    (8, 6, 5),
    (12, 10, 9),
    (16, 14, 13),
    (20, 18, 17),
)


@dataclass
class GestureDetection:
    name: Optional[str]
    confidence: float
    raw_pose: Optional[str] = None


def _dist(a: Any, b: Any) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _hand_scale(lm: list[Any]) -> float:
    return max(_dist(lm[0], lm[9]), 0.05)


def _palm_center(lm: list[Any]) -> tuple[float, float]:
    xs = [lm[i].x for i in (0, 5, 9, 13, 17)]
    ys = [lm[i].y for i in (0, 5, 9, 13, 17)]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _finger_extended(lm: list[Any], tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
    wrist, tip, pip, mcp = lm[0], lm[tip_idx], lm[pip_idx], lm[mcp_idx]
    scale = _hand_scale(lm)
    height = tip.y < pip.y - scale * 0.06
    wrist_far = _dist(tip, wrist) > _dist(pip, wrist) * 1.03
    bone_long = _dist(tip, mcp) > _dist(pip, mcp) * 1.05
    return sum((height, wrist_far, bone_long)) >= 2


def _thumb_state(lm: list[Any], handedness: str) -> tuple[bool, bool, bool]:
    tip, ip, mcp = lm[4], lm[3], lm[2]
    wrist = lm[0]
    px, py = _palm_center(lm)
    scale = _hand_scale(lm)

    if tip.y > wrist.y + scale * 0.06 and tip.y > ip.y:
        return False, True, True

    tip_palm = ((tip.x - px) ** 2 + (tip.y - py) ** 2) ** 0.5
    ip_palm = ((ip.x - px) ** 2 + (ip.y - py) ** 2) ** 0.5
    mcp_palm = ((mcp.x - px) ** 2 + (mcp.y - py) ** 2) ** 0.5

    extended = tip_palm > max(ip_palm * 1.12, mcp_palm * 1.05, scale * 0.55)
    tucked = tip_palm < scale * 0.45 and not extended
    down = False

    if not extended and not tucked and scale > 0.06:
        if handedness == "Right":
            lateral_ext = tip.x > ip.x + scale * 0.12
            lateral_in = tip.x < ip.x - scale * 0.08
        else:
            lateral_ext = tip.x < ip.x - scale * 0.12
            lateral_in = tip.x > ip.x + scale * 0.08
        if lateral_ext and not down:
            extended = True
        if lateral_in:
            tucked = True

    return extended, tucked, down


def _finger_states(lm: list[Any], handedness: str) -> tuple[list[bool], bool, bool, bool]:
    fu = [_finger_extended(lm, tip, pip, mcp) for tip, pip, mcp in FINGERS]
    thumb_ext, thumb_tucked, thumb_down = _thumb_state(lm, handedness)
    return fu, thumb_ext, thumb_tucked, thumb_down


def _is_duplicate_hand(lm_a: list[Any], lm_b: list[Any]) -> bool:
    scale = min(_hand_scale(lm_a), _hand_scale(lm_b))
    if _dist(lm_a[0], lm_b[0]) < scale * 0.55:
        return True
    if _dist(lm_a[9], lm_b[9]) < scale * 0.45:
        return True
    c_a, c_b = _palm_center(lm_a), _palm_center(lm_b)
    palm_dist = ((c_a[0] - c_b[0]) ** 2 + (c_a[1] - c_b[1]) ** 2) ** 0.5
    return palm_dist < scale * 0.45


def detect_heart_two_hands(landmarks_list: list[list[Any]]) -> Optional[GestureDetection]:
    if len(landmarks_list) < 2:
        return None
    for i in range(len(landmarks_list)):
        for j in range(i + 1, len(landmarks_list)):
            a, b = landmarks_list[i], landmarks_list[j]
            if _is_duplicate_hand(a, b):
                continue
            scale = min(_hand_scale(a), _hand_scale(b))
            if _dist(a[8], b[8]) < scale * 0.60 and _dist(a[4], b[4]) < scale * 0.90:
                return GestureDetection("heart_hands", 0.90, "heart_hands")
    return None


def _is_pointing_pose(lm: list[Any], handedness: str) -> bool:
    fu, _, _, _ = _finger_states(lm, handedness)
    index, middle, ring, pinky = fu
    return index and not middle and not ring and not pinky


def _index_extended(lm: list[Any]) -> bool:
    scale = _hand_scale(lm)
    tip, pip, mcp, wrist = lm[8], lm[6], lm[5], lm[0]
    if _dist(tip, mcp) > _dist(pip, mcp) * 1.02:
        return True
    if _dist(tip, wrist) > _dist(pip, wrist) * 1.01:
        return True
    if tip.y < pip.y - scale * 0.04:
        return True
    if abs(tip.x - mcp.x) > scale * 0.22:
        return True
    return False


def is_core_landmark_gesture(lm: list[Any], handedness: str = "Right") -> bool:
    """True when pose is not a single-finger ear/mouth point."""
    fu, thumb_ext, _, _ = _finger_states(lm, handedness)
    n = sum(fu)
    index, middle, ring, pinky = fu
    if n >= 3 or n == 0:
        return True
    if index and middle and not ring and not pinky:
        return True
    if thumb_ext and pinky and not middle and not ring:
        return True
    if index and pinky and not middle and not ring:
        return True
    return False


def _body_point_scores(lm: list[Any], handedness: str = "Right") -> tuple[float, float]:
    if not _index_extended(lm):
        return 0.0, 0.0

    fu, _, _, _ = _finger_states(lm, handedness)
    if fu[0] and fu[1] and not fu[2] and not fu[3]:
        return 0.0, 0.0

    scale = _hand_scale(lm)
    px, py = _palm_center(lm)
    tip = lm[8]
    palm_off = abs(px - 0.5)
    tip_off = abs(tip.x - 0.5)
    side_reach = abs(tip.x - px)

    ear = 0.0
    mouth = 0.0

    if palm_off > 0.06:
        ear += min(0.35, palm_off * 2.5)
    if tip_off > 0.10:
        ear += min(0.25, (tip_off - 0.06) * 2.0)
    if side_reach > scale * 0.10:
        ear += 0.20
    if (px - 0.5) * (tip.x - 0.5) > 0.002:
        ear += 0.20
    if abs(tip.x - 0.5) > abs(px - 0.5) + 0.03:
        ear += 0.15

    if palm_off < 0.14:
        mouth += 0.30
    if tip_off < 0.12:
        mouth += 0.30
    if _dist(tip, lm[4]) < scale * 0.52:
        mouth += 0.35
    if side_reach < scale * 0.28:
        mouth += 0.20
    if abs(tip.x - px) < scale * 0.30:
        mouth += 0.15
    if 0.40 < tip.y < 0.58:
        mouth += 0.10
    if tip.y < py - scale * 0.18:
        mouth = max(0.0, mouth - 0.45)
    if palm_off > 0.12:
        mouth = max(0.0, mouth - 0.35)
    if palm_off < 0.10:
        ear = max(0.0, ear - 0.40)
    if side_reach < scale * 0.12:
        ear = max(0.0, ear - 0.30)

    return min(1.0, ear), min(1.0, mouth)


def _is_point_ear(lm: list[Any], handedness: str = "Right") -> bool:
    ear, mouth = _body_point_scores(lm, handedness)
    return ear >= 0.55 and ear > mouth + 0.18


def _is_point_mouth(lm: list[Any], handedness: str = "Right") -> bool:
    ear, mouth = _body_point_scores(lm, handedness)
    return mouth >= 0.55 and mouth > ear + 0.18


def resolve_gesture(name: Optional[str], handedness: str = "Right") -> Optional[str]:
    if name == "goodbye_spread":
        return "spread_hello" if handedness == "Right" else "spread_goodbye"
    return name


def detect_extra_gestures(lm: list[Any], handedness: str = "Right") -> Optional[GestureDetection]:
    if verify_gesture("ok", lm, handedness):
        return GestureDetection("ok", 0.90, "ok")
    if verify_gesture("crossed_fingers", lm, handedness):
        return GestureDetection("crossed_fingers", 0.90, "crossed_fingers")
    if verify_gesture("goodbye_spread", lm, handedness):
        resolved = resolve_gesture("goodbye_spread", handedness)
        return GestureDetection(resolved, 0.91, "goodbye_spread")
    return None


def detect_body_point_gestures(lm: list[Any], handedness: str = "Right") -> Optional[GestureDetection]:
    if not _is_pointing_pose(lm, handedness) or is_core_landmark_gesture(lm, handedness):
        return None
    ear, mouth = _body_point_scores(lm, handedness)
    if ear >= 0.55 and ear > mouth + 0.18:
        return GestureDetection("point_ear", ear, "point_ear")
    if mouth >= 0.55 and mouth > ear + 0.18:
        return GestureDetection("hungry", mouth, "hungry")
    return None


def verify_gesture(name: str, lm: list[Any], handedness: str = "Right") -> bool:
    fu, thumb_ext, _thumb_tucked, thumb_down = _finger_states(lm, handedness)
    index, middle, ring, pinky = fu
    n = sum(fu)
    scale = _hand_scale(lm)
    index_mid_dist = _dist(lm[8], lm[12])
    mcp_spread = abs(lm[5].x - lm[17].x)
    tip_spread = abs(lm[8].x - lm[20].x)

    checks: dict[str, bool] = {
        "thumbs_up": n == 0 and thumb_ext and not thumb_down,
        "thumbs_down": thumb_down and n <= 1,
        "fist": n == 0 and not thumb_ext and not thumb_down,
        "peace": index and middle and not ring and not pinky and index_mid_dist > scale * 0.35,
        "pointing": (
            index and not middle and not ring and not pinky
            and not _is_point_ear(lm, handedness)
            and not _is_point_mouth(lm, handedness)
        ),
        "point_ear": _is_point_ear(lm, handedness),
        "hungry": _is_point_mouth(lm, handedness),
        "love_you": index and pinky and thumb_ext and not middle and not ring,
        "ok": (
            _dist(lm[4], lm[8]) < scale * 0.50
            and middle
            and ring
            and not index
            and n >= 2
        ),
        "crossed_fingers": (
            index and middle and not ring and not pinky
            and index_mid_dist < scale * 0.35
        ),
        "goodbye_spread": (
            n == 4
            and mcp_spread > scale * 0.35
            and tip_spread > mcp_spread * 1.55
        ),
    }
    return checks.get(name, True)


def _score_static(lm: list[Any], handedness: str) -> list[tuple[str, float]]:
    fu, thumb_ext, thumb_tucked, thumb_down = _finger_states(lm, handedness)
    index, middle, ring, pinky = fu
    n = sum(fu)
    scale = _hand_scale(lm)
    scores: list[tuple[str, float]] = []

    index_mid_dist = _dist(lm[8], lm[12])
    index_mid_y = abs(lm[8].y - lm[12].y)
    mcp_spread = abs(lm[5].x - lm[17].x)
    tip_spread = abs(lm[8].x - lm[20].x)

    if n == 0 and not thumb_ext and not thumb_down:
        scores.append(("fist", 0.94 if thumb_tucked else 0.88))
    if thumb_down and n <= 1:
        scores.append(("thumbs_down", 0.96))
    if n == 0 and thumb_ext and not thumb_down:
        scores.append(("thumbs_up", 0.93))

    if n == 4 and tip_spread > mcp_spread * 1.55 and mcp_spread > scale * 0.35:
        scores.append(("goodbye_spread", 0.91))

    if index and middle and not ring and not pinky:
        if index_mid_dist < scale * 0.35 and index_mid_y > scale * 0.12:
            scores.append(("crossed_fingers", 0.89))
        elif index_mid_dist > scale * 0.45:
            scores.append(("peace", 0.92))
        else:
            scores.append(("peace", 0.80))

    if index and not middle and not ring and not pinky:
        body = detect_body_point_gestures(lm, handedness)
        if body and body.name:
            scores.append((body.name, body.confidence))
        else:
            scores.append(("pointing", 0.88))

    if index and not middle and not ring and pinky and thumb_ext:
        scores.append(("love_you", 0.93))

    thumb_index_dist = _dist(lm[4], lm[8])
    if thumb_index_dist < scale * 0.50 and middle and ring and not index and n >= 2:
        scores.append(("ok", 0.90))

    return scores


def classify_static(lm: list[Any], handedness: str = "Right") -> GestureDetection:
    scores = _score_static(lm, handedness)
    if not scores:
        return GestureDetection(None, 0.0, None)

    merged: dict[str, float] = {}
    for name, score in scores:
        merged[name] = max(merged.get(name, 0.0), score)

    best_name = max(merged, key=merged.get)
    best_score = merged[best_name]

    sorted_scores = sorted(merged.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 0.08:
        for p in ("ok", "crossed_fingers", "fist", "peace", "pointing", "thumbs_up", "thumbs_down"):
            if p in merged and merged[p] >= sorted_scores[0] - 0.05:
                best_name = p
                best_score = merged[p]
                break

    if best_score < 0.75 or not verify_gesture(best_name, lm, handedness):
        return GestureDetection(None, 0.0, None)

    if best_name == "goodbye_spread":
        best_name = resolve_gesture(best_name, handedness) or best_name

    return GestureDetection(best_name, best_score, best_name)
