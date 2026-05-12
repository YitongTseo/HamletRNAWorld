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


class TextScroller:
    def __init__(self, sentences: list[list[str]]):
        self.sentences = sentences
        self._sent_idx = 0
        self._line_id = 0
        self._active: list[list[WordState]] = []   # list of lines (each = list of words)
        self._dead: set[tuple[int, int]] = set()   # (line_id, word_idx) eaten by worm
        self._elapsed = 0.0
        self._next_spawn = 0.0

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

        # Spawn next line if it's time
        if self._elapsed >= self._next_spawn:
            self._spawn()
            self._next_spawn = self._elapsed + SPAWN_INTERVAL

    def _spawn(self) -> None:
        """Create a new line of words from the next sentence."""
        tokens = self.sentences[self._sent_idx % len(self.sentences)]
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
