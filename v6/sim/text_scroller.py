"""Text scroller: manage Hamlet lines drifting upward, word positions, and eaten state."""
from __future__ import annotations
from dataclasses import dataclass

WORLD_W = 1600.0
WORLD_H = 1000.0

SCROLL_SPEED = 15.0       # world units / second (upward = decreasing y)
LINE_SPACING = 75.0       # world units between lines at spawn
SPAWN_Y = WORLD_H - 20.0
KILL_Y = -80.0
SPAWN_INTERVAL = 4.5      # seconds between new lines
CHAR_W = 11.0             # world units per character (monospace approx)
WORD_GAP = 70.0           # world units gap between words
CENTER_X = WORLD_W / 2.0


@dataclass
class WordState:
    text: str
    line_id: int
    word_idx: int
    x: float
    y: float
    alive: bool = True
    edible: bool = True  # False for set-dressing words (names, cues, directions)


class TextScroller:
    def __init__(self, sentences: list[list[str]], loop: bool = True,
                 edible_flags: list[bool] | None = None):
        """If loop=False, stop spawning new lines after one full pass; the
        `corpus_exhausted` property goes True once the last line has scrolled
        off the screen. Generation mode passes loop=False.

        `edible_flags` is a per-sentence list (same indexing as `sentences`):
        False makes every word in that sentence inert (non-edible,
        non-smellable) — used to exclude speaker names / scene cues / stage
        directions. Omitted → all words edible (legacy behavior)."""
        self.sentences = sentences
        self.edible_flags = edible_flags or []
        self.loop = loop
        self._sent_idx = 0
        self._line_id = 0
        self._active: list[list[WordState]] = []
        self._dead: set[tuple[int, int]] = set()
        self._elapsed = 0.0
        self._next_spawn = 0.0

    @property
    def corpus_exhausted(self) -> bool:
        """True once all sentences have been spawned AND all spawned lines
        have either been eaten through or scrolled off the top. Only ever
        True in one-pass mode (loop=False)."""
        return (not self.loop
                and self._sent_idx >= len(self.sentences)
                and not self._active)

    def step(self, dt: float) -> None:
        """Advance time, scroll words upward, spawn new lines, remove off-screen lines."""
        self._elapsed += dt

        # Scroll all words upward (decreasing y)
        for line in self._active:
            for w in line:
                w.y -= SCROLL_SPEED * dt

        # Remove lines entirely off-screen
        self._active = [
            line for line in self._active
            if any(w.y > KILL_Y for w in line)
        ]

        # Spawn next line if it's time and we have more to spawn
        if self._elapsed >= self._next_spawn:
            if self.loop or self._sent_idx < len(self.sentences):
                self._spawn()
            self._next_spawn = self._elapsed + SPAWN_INTERVAL

    def _spawn(self) -> None:
        """Create a new line of words from the next sentence."""
        if self.loop:
            si = self._sent_idx % len(self.sentences)
        else:
            if self._sent_idx >= len(self.sentences):
                return  # nothing left; corpus_exhausted will go True after last line scrolls off
            si = self._sent_idx
        tokens = self.sentences[si]
        edible = self.edible_flags[si] if si < len(self.edible_flags) else True
        self._sent_idx += 1
        lid = self._line_id
        self._line_id += 1

        # Horizontal layout centered on world center
        total_w = sum(len(t) * CHAR_W for t in tokens) + WORD_GAP * (len(tokens) - 1)
        x = CENTER_X - total_w / 2
        line: list[WordState] = []

        for idx, tok in enumerate(tokens):
            w = WordState(
                text=tok,
                line_id=lid,
                word_idx=idx,
                x=x + len(tok) * CHAR_W / 2,  # center of word
                y=SPAWN_Y,
                edible=edible,
            )
            line.append(w)
            x += len(tok) * CHAR_W + WORD_GAP

        self._active.append(line)

    def mark_eaten(self, line_id: int, word_idx: int) -> None:
        """Mark a word as eaten; it won't be returned by alive_words()."""
        self._dead.add((line_id, word_idx))

    def alive_words(self) -> list[WordState]:
        """Words still alive (not eaten, on screen)."""
        return [
            w
            for line in self._active
            for w in line
            if (w.line_id, w.word_idx) not in self._dead
        ]

    def all_words(self) -> list[WordState]:
        """All words (alive + eaten) for rendering eaten words as faded."""
        return [w for line in self._active for w in line]

    # ------------------------------------------------------------------
    # Mid-generation checkpoint support.
    #
    # The scroll position (_sent_idx), the eaten set (_dead), and the words
    # currently on screen (_active) are the entirety of a worm's *corpus*
    # progress within a generation. Snapshotting them lets a restart resume
    # mid-sentence/mid-chew instead of restarting the generation from word 0.
    # The worm's physical body + brain are deliberately NOT captured (they
    # re-settle within ~1s); see docs spec. `sentences`/`edible_flags`/`loop`
    # are not serialized — they're fixed by the corpus + run mode and are
    # reconstructed when the scroller is built.
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Serializable corpus-progress state. All plain JSON scalars."""
        return {
            "sent_idx": self._sent_idx,
            "line_id": self._line_id,
            "elapsed": self._elapsed,
            "next_spawn": self._next_spawn,
            "dead": [list(k) for k in self._dead],
            "active": [
                [
                    {"text": w.text, "line_id": w.line_id, "word_idx": w.word_idx,
                     "x": w.x, "y": w.y, "alive": w.alive, "edible": w.edible}
                    for w in line
                ]
                for line in self._active
            ],
        }

    def restore(self, state: dict) -> None:
        """Overwrite scroll state from a snapshot() dict. Safe to call right
        after construction; replaces all mutable progress fields in place."""
        self._sent_idx = int(state["sent_idx"])
        self._line_id = int(state["line_id"])
        self._elapsed = float(state["elapsed"])
        self._next_spawn = float(state["next_spawn"])
        self._dead = {(int(a), int(b)) for a, b in state.get("dead", [])}
        self._active = [
            [
                WordState(text=w["text"], line_id=w["line_id"], word_idx=w["word_idx"],
                          x=w["x"], y=w["y"], alive=w.get("alive", True),
                          edible=w.get("edible", True))
                for w in line
            ]
            for line in state.get("active", [])
        ]
