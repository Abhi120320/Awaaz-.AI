"""Neon aurora hand skeleton — animated wireframe with pulse waves and orbit particles."""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Each finger chain: wrist → tip (for color + pulse direction)
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 5, 9, 10, 11, 12],
    [0, 5, 9, 13, 14, 15, 16],
    [0, 5, 9, 13, 17, 18, 19, 20],
]

# BGR neon palette per finger
FINGER_PALETTE = [
    (80, 140, 255),   # thumb — warm coral
    (255, 220, 60),   # index — electric cyan
    (255, 80, 220),   # middle — magenta
    (60, 255, 180),   # ring — mint
    (180, 120, 255),  # pinky — violet
]

STABLE_RING_COLOR = (100, 255, 160)
IDLE_RING_COLOR = (255, 200, 80)


def _chain_for_connection(a: int, b: int) -> int:
    for i, chain in enumerate(FINGER_CHAINS):
        if a in chain and b in chain:
            return i
    return 0


def _dist_along_chain(chain: list[int], idx: int) -> float:
    if idx not in chain:
        return 0.0
    pos = chain.index(idx)
    return pos / max(len(chain) - 1, 1)


class HandVisualizer:
    """Draws an animated neon skeleton with traveling pulses and orbit particles."""

    def __init__(self, particle_count: int = 14) -> None:
        self._t0 = time.time()
        self._particles = [
            {"angle": (i / particle_count) * math.tau, "speed": 0.9 + (i % 5) * 0.15, "r": 28 + (i % 4) * 12}
            for i in range(particle_count)
        ]

    def _phase(self) -> float:
        return (time.time() - self._t0) * 3.2

    def _landmark_points(self, landmarks: list[Any], w: int, h: int) -> list[tuple[int, int]]:
        return [(int(l.x * w), int(l.y * h)) for l in landmarks]

    def _palm_center(self, pts: list[tuple[int, int]]) -> tuple[int, int]:
        cx = int((pts[0][0] + pts[5][0] + pts[9][0] + pts[13][0] + pts[17][0]) / 5)
        cy = int((pts[0][1] + pts[5][1] + pts[9][1] + pts[13][1] + pts[17][1]) / 5)
        return cx, cy

    def _draw_glow_line(
        self,
        overlay: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        pulse: float,
    ) -> None:
        brightness = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(pulse))
        c = tuple(int(ch * brightness) for ch in color)
        cv2.line(overlay, p1, p2, c, 6, cv2.LINE_AA)
        cv2.line(overlay, p1, p2, tuple(min(255, ch + 40) for ch in c), 2, cv2.LINE_AA)

    def _draw_pulse_dot(
        self,
        overlay: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        phase: float,
        offset: float,
    ) -> None:
        t = (math.sin(phase + offset) + 1) * 0.5
        x = int(p1[0] + (p2[0] - p1[0]) * t)
        y = int(p1[1] + (p2[1] - p1[1]) * t)
        glow = tuple(min(255, int(c * 1.2)) for c in color)
        cv2.circle(overlay, (x, y), 9, glow, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_orbit_particles(
        self,
        overlay: np.ndarray,
        center: tuple[int, int],
        phase: float,
        is_stable: bool,
    ) -> None:
        cx, cy = center
        for i, p in enumerate(self._particles):
            ang = p["angle"] + phase * p["speed"] * 0.08
            wobble = 6 * math.sin(phase * 0.7 + i)
            r = p["r"] + wobble + (8 if is_stable else 0)
            x = int(cx + math.cos(ang) * r)
            y = int(cy + math.sin(ang) * r * 0.65)
            hue_shift = int(40 * math.sin(phase * 0.5 + i))
            base = STABLE_RING_COLOR if is_stable else FINGER_PALETTE[i % len(FINGER_PALETTE)]
            col = tuple(max(0, min(255, c + hue_shift)) for c in base)
            cv2.circle(overlay, (x, y), 3, col, -1, cv2.LINE_AA)
            if i % 3 == 0:
                cv2.line(overlay, center, (x, y), tuple(int(c * 0.35) for c in col), 1, cv2.LINE_AA)

    def _draw_scan_ring(
        self,
        overlay: np.ndarray,
        center: tuple[int, int],
        phase: float,
        is_stable: bool,
    ) -> None:
        cx, cy = center
        base_r = 55 + int(8 * math.sin(phase * 0.6))
        color = STABLE_RING_COLOR if is_stable else IDLE_RING_COLOR
        n_segments = 24
        for i in range(n_segments):
            seg_phase = phase * 1.4 + i * (math.tau / n_segments)
            alpha = 0.25 + 0.75 * max(0, math.sin(seg_phase))
            if alpha < 0.2:
                continue
            a1 = (i / n_segments) * math.tau
            a2 = ((i + 1) / n_segments) * math.tau
            p1 = (int(cx + math.cos(a1) * base_r), int(cy + math.sin(a1) * base_r * 0.75))
            p2 = (int(cx + math.cos(a2) * base_r), int(cy + math.sin(a2) * base_r * 0.75))
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(overlay, p1, p2, c, 2, cv2.LINE_AA)

        # Expanding ripple when gesture is locked
        if is_stable:
            ripple = (math.sin(phase * 2.0) + 1) * 0.5
            rr = int(base_r + 20 + ripple * 25)
            cv2.ellipse(overlay, center, (rr, int(rr * 0.75)), 0, 0, 360, color, 1, cv2.LINE_AA)

    def _draw_joints(
        self,
        overlay: np.ndarray,
        pts: list[tuple[int, int]],
        phase: float,
        is_stable: bool,
    ) -> None:
        for chain_idx, chain in enumerate(FINGER_CHAINS):
            color = FINGER_PALETTE[chain_idx]
            for j, idx in enumerate(chain):
                if idx >= len(pts):
                    continue
                pt = pts[idx]
                dist = _dist_along_chain(chain, idx)
                pulse = phase * 1.8 - dist * 2.5
                size = 5 + int(3 * (0.5 + 0.5 * math.sin(pulse)))
                if idx in (4, 8, 12, 16, 20):
                    size += 2
                glow = tuple(min(255, int(c * (0.7 + 0.3 * math.sin(pulse)))) for c in color)
                cv2.circle(overlay, pt, size + 3, glow, -1, cv2.LINE_AA)
                core = (255, 255, 255) if is_stable and j == len(chain) - 1 else glow
                cv2.circle(overlay, pt, size, core, -1, cv2.LINE_AA)

    def draw(
        self,
        frame: np.ndarray,
        landmarks: list[Any],
        *,
        is_stable: bool = False,
        gesture: Optional[str] = None,
    ) -> None:
        h, w = frame.shape[:2]
        pts = self._landmark_points(landmarks, w, h)
        phase = self._phase()
        center = self._palm_center(pts)

        overlay = frame.copy()

        self._draw_scan_ring(overlay, center, phase, is_stable)
        self._draw_orbit_particles(overlay, center, phase, is_stable)

        for a, b in HAND_CONNECTIONS:
            chain_idx = _chain_for_connection(a, b)
            color = FINGER_PALETTE[chain_idx]
            dist = (_dist_along_chain(FINGER_CHAINS[chain_idx], a) + _dist_along_chain(FINGER_CHAINS[chain_idx], b)) / 2
            pulse = phase * 2.0 - dist * 3.0
            self._draw_glow_line(overlay, pts[a], pts[b], color, pulse)
            if (a + b) % 3 == 0:
                self._draw_pulse_dot(overlay, pts[a], pts[b], color, phase, a * 0.4)

        self._draw_joints(overlay, pts, phase, is_stable)

        # Gesture label halo near palm
        if gesture and is_stable:
            cv2.putText(
                overlay,
                gesture.replace("_", " ").upper(),
                (center[0] - 40, center[1] - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                STABLE_RING_COLOR,
                1,
                cv2.LINE_AA,
            )

        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
