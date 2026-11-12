"""``application`` / ``tags`` on observations — sent since 1.6.0.

The server's ObservationDto reads both and the spend event takes the
application, so calls an SDK makes directly are attributed to the
application they were wrapped with (1.5.x held them back and every such
call landed as "sdk:tier2"). byoc-v0.1.40 validates observations with
``forbidNonWhitelisted`` and has neither field, so it rejects them: an
install on that server must upgrade with 1.6.0.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

#: Internal. On since 1.6.0; tests may flip it.
SEND = True


def observation_scope(application: Optional[str], tags: Optional[Iterable[str]]) -> dict[str, Any]:
    if not SEND:
        return {}
    tag_list = [t for t in (tags or []) if t]
    return {
        **({"application": application} if application else {}),
        **({"tags": tag_list} if tag_list else {}),
    }
