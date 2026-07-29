"""Timed shot lists: what a generated video does over the length of an audio track.

Shared by any node that needs to turn a timeline of prompts into per-segment
decisions. Nothing here is specific to music.
"""
import re

SHOT_RE = re.compile(r"^\[\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*(cut|back)?\s*\]\s*$")


def parse_shots(text):
    """Parse a shot list into [{at, cut, back, prompt}], earliest first.

        [0:00]
        The opening shot.

        [0:32 cut]
        A real scene change, generated fresh.

        [0:48 back]
        Return to the earlier subject, seeded from before the cutaway.

    A bare marker keeps the frame chain, so the change reads as a move within
    one continuous shot. `cut` breaks the chain. `back` resumes from the
    anchor, the last frame before the most recent cutaway, so a subject
    returns looking like itself.

    Lines starting with # are comments. Text runs until the next marker.
    """
    shots, cur = [], None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        m = SHOT_RE.match(stripped)
        if m:
            at = int(m.group(1) or 0) * 60 + float(m.group(2))
            kind = m.group(3) or ""
            cur = {"at": at, "cut": kind == "cut", "back": kind == "back",
                   "lines": []}
            shots.append(cur)
        elif cur is not None:
            cur["lines"].append(raw.rstrip())
    if not shots:
        raise ValueError("shot list has no [mm:ss] markers")
    for s in shots:
        s["prompt"] = " ".join(x.strip() for x in s["lines"] if x.strip())
        if not s["prompt"]:
            raise ValueError(f"the shot at {s['at']:g}s has no prompt text")
    shots.sort(key=lambda s: s["at"])
    if shots[0]["at"] > 0:
        raise ValueError("the first shot must start at [0:00], got "
                         f"{shots[0]['at']:g}s")
    return shots


def shot_at(shots, t):
    """Index of the shot in force at time t."""
    idx = 0
    for i, s in enumerate(shots):
        if s["at"] <= t + 1e-6:
            idx = i
    return idx


def plan(shots, segments, seg_secs):
    """Per-segment decisions: (shot_index, is_new_shot, source).

    source is "fresh" (nothing carried in), "anchor" (resume from before the
    cutaway) or "previous" (continue the preceding segment).
    """
    out = []
    for i in range(segments):
        start = i * seg_secs
        si = shot_at(shots, start)
        new_shot = i > 0 and si != shot_at(shots, start - seg_secs)
        if i == 0 or (new_shot and shots[si]["cut"]):
            src = "fresh"
        elif new_shot and shots[si]["back"]:
            src = "anchor"
        else:
            src = "previous"
        out.append((si, new_shot, src))
    return out
