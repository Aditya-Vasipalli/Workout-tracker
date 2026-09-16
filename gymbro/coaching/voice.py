"""Live spoken coaching.

Three things make or break this, and the naive version gets all three wrong:

  * It must not block. `pyttsx3.runAndWait()` blocks until the phrase finishes;
    calling it from the capture loop stalls pose estimation for the whole
    utterance. Speech runs on its own thread here, and the loop never waits.
  * It must not nag. Repeating "engage your core" every frame -- or even every
    two seconds -- is noise you stop hearing. Each cue has a cooldown, and a
    fault you are already fixing goes quiet.
  * It must say the most useful thing, not everything. A safety correction
    interrupts a rep count; a refinement waits for a gap.

Stale cues are dropped rather than queued: advice about a position you left two
seconds ago is worse than silence, because you will try to act on it.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from queue import Empty


class Priority(IntEnum):
    """Lower speaks first."""

    SAFETY = 0        # stop-the-set corrections
    REP_COUNT = 1     # the count itself
    FORM = 2          # technique corrections
    TEMPO = 3         # pace and holds
    ENCOURAGE = 4     # confirmation, praise


@dataclass(order=True)
class Cue:
    priority: int
    sequence: int = field(compare=True)
    key: str = field(compare=False, default="")
    text: str = field(compare=False, default="")
    created: float = field(compare=False, default_factory=time.monotonic)
    # Past this age the cue is no longer worth saying.
    ttl_s: float = field(compare=False, default=2.5)
    cooldown_s: float = field(compare=False, default=6.0)

    def stale(self, now: float) -> bool:
        return (now - self.created) > self.ttl_s


class Speaker:
    """Text to speech backend. Subclass to swap engines or capture for tests."""

    def speak(self, text: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class Pyttsx3Speaker(Speaker):
    """Local offline TTS. The engine is created on the worker thread.

    pyttsx3 is not thread-safe and misbehaves if initialised on one thread and
    driven from another, so construction is deferred to the first `speak` call,
    which happens on the worker.
    """

    def __init__(self, rate: int = 185, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._engine = None

    def _ensure(self):
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
        return self._engine

    def speak(self, text: str) -> None:
        engine = self._ensure()
        engine.say(text)
        engine.runAndWait()

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


class CommandSpeaker(Speaker):
    """Uses a system binary (`say` on macOS, `espeak` on Linux)."""

    def __init__(self, command: list[str]):
        self.command = command

    def speak(self, text: str) -> None:
        import subprocess

        subprocess.run([*self.command, text], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class NullSpeaker(Speaker):
    """Records instead of speaking. Used by tests and `--quiet`."""

    def __init__(self):
        self.spoken: list[tuple[float, str]] = []

    def speak(self, text: str) -> None:
        self.spoken.append((time.monotonic(), text))


def auto_speaker() -> Speaker:
    """Pick whatever actually works on this machine."""
    import shutil
    import sys

    try:
        import pyttsx3  # noqa: F401

        return Pyttsx3Speaker()
    except ImportError:
        pass

    if sys.platform == "darwin" and shutil.which("say"):
        return CommandSpeaker(["say"])
    for binary in ("espeak-ng", "espeak"):
        if shutil.which(binary):
            return CommandSpeaker([binary])
    return NullSpeaker()


class VoiceCoach:
    """Priority-queued, rate-limited, non-blocking speech."""

    def __init__(
        self,
        speaker: Speaker | None = None,
        default_cooldown_s: float = 6.0,
        min_gap_s: float = 0.35,
        enabled: bool = True,
    ):
        self.speaker = speaker if speaker is not None else auto_speaker()
        self.default_cooldown_s = default_cooldown_s
        self.min_gap_s = min_gap_s
        self.enabled = enabled

        self._heap: list[Cue] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._counter = itertools.count()
        self._last_spoken: dict[str, float] = {}
        self._last_utterance = 0.0
        self._thread: threading.Thread | None = None

        self.dropped_stale = 0
        self.suppressed_cooldown = 0

    # -------------------------------------------------------------- lifecycle

    def start(self) -> "VoiceCoach":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self, drain: bool = False) -> None:
        if not drain:
            with self._lock:
                self._heap.clear()
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.speaker.close()

    def __enter__(self) -> "VoiceCoach":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ input

    def say(
        self,
        text: str,
        priority: Priority = Priority.FORM,
        key: str | None = None,
        cooldown_s: float | None = None,
        ttl_s: float = 2.5,
    ) -> bool:
        """Queue a cue. Returns False if it was suppressed.

        `key` is the dedup identity: two cues with the same key inside the
        cooldown collapse into one, so a fault that persists across twenty
        frames is spoken once.
        """
        if not self.enabled:
            return False

        key = key or text
        cooldown = self.default_cooldown_s if cooldown_s is None else cooldown_s
        now = time.monotonic()

        with self._lock:
            last = self._last_spoken.get(key)
            if last is not None and (now - last) < cooldown:
                self.suppressed_cooldown += 1
                return False

            # Already queued under the same key: don't stack duplicates.
            if any(c.key == key for c in self._heap):
                return False

            # Reserve the slot now so a burst of identical frames collapses,
            # rather than all passing the check before the first is spoken.
            self._last_spoken[key] = now
            heapq.heappush(self._heap, Cue(
                priority=int(priority), sequence=next(self._counter), key=key,
                text=text, created=now, ttl_s=ttl_s, cooldown_s=cooldown,
            ))

        self._wake.set()
        return True

    def interrupt(self, text: str, key: str | None = None) -> bool:
        """Say something immediately, clearing anything queued behind it."""
        with self._lock:
            self._heap.clear()
        return self.say(text, Priority.SAFETY, key=key, cooldown_s=3.0, ttl_s=1.5)

    def clear(self) -> None:
        with self._lock:
            self._heap.clear()

    def forget(self, key: str) -> None:
        """Reset a cue's cooldown, so it can be said again right away."""
        with self._lock:
            self._last_spoken.pop(key, None)

    # ------------------------------------------------------------------ worker

    def _loop(self) -> None:
        while not self._stop.is_set():
            cue = self._next_cue()
            if cue is None:
                self._wake.wait(timeout=0.1)
                self._wake.clear()
                continue

            gap = time.monotonic() - self._last_utterance
            if gap < self.min_gap_s:
                time.sleep(self.min_gap_s - gap)

            try:
                self.speaker.speak(cue.text)
            except Exception:
                # A broken voice engine must never take the workout down.
                pass
            self._last_utterance = time.monotonic()

    def _next_cue(self) -> Cue | None:
        now = time.monotonic()
        with self._lock:
            while self._heap:
                cue = heapq.heappop(self._heap)
                if cue.stale(now):
                    self.dropped_stale += 1
                    continue
                return cue
        return None

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._heap)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until the queue drains. For tests and end-of-session."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.pending == 0:
                time.sleep(0.05)
                if self.pending == 0:
                    return True
            time.sleep(0.02)
        return False
