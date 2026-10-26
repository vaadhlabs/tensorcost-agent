"""``application`` / ``tags`` on observations — held back in 1.5.0.

The server's ObservationDto reads both, but byoc-v0.1.40 (what design
partners run) validates observations with ``forbidNonWhitelisted`` and has
neither field, so an observation carrying them is rejected with a 400 and
dropped. The fields are built here and sent only when ``SEND`` is True,
which a later release turns on alongside the server release that accepts them.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

#: Internal. Off in 1.5.0; tests flip it to pin the later behaviour.
SEND = False


def observation_scope(application: Optional[str], tags: Optional[Iterable[str]]) -> dict[str, Any]:
    if not SEND:
        return {}
    tag_list = [t for t in (tags or []) if t]
    return {
        **({"application": application} if application else {}),
        **({"tags": tag_list} if tag_list else {}),
    }
