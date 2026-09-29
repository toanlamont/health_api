"""Keep the raw colour means of recent captures on disk.

A rejected capture only reports one number, and that number cannot say whether
the pulse is buried under drift, erased by the camera's own noise reduction, or
simply not there. Keeping the arrays makes that answerable offline with
tools/inspect_capture.py instead of by guessing from a screenshot.

Local diagnostic only: colour means of skin, timestamps, and nothing that
identifies anyone. Set HEARTBEAT_SAVE_CAPTURES=0 to turn it off.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("heartbeat.captures")

DIRECTORY = Path(__file__).resolve().parent.parent / "captures"
# Enough to compare a handful of attempts against each other without the folder
# growing without bound.
KEEP = 50


def enabled() -> bool:
    return os.environ.get("HEARTBEAT_SAVE_CAPTURES", "1") != "0"


def save(source: str, times: list[float], region_count: int,
         values: list[float], outcome: dict) -> Path | None:
    if not enabled():
        return None
    try:
        DIRECTORY.mkdir(exist_ok=True)
        # Milliseconds, because several captures a second apart would otherwise
        # overwrite each other.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = DIRECTORY / f"{stamp}-{source}.json"
        path.write_text(json.dumps({
            "source": source,
            "saved_at": stamp,
            "region_count": region_count,
            "times": times,
            "values": values,
            "outcome": outcome,
        }), encoding="utf-8")

        existing = sorted(DIRECTORY.glob("*.json"))
        for stale in existing[:-KEEP]:
            stale.unlink(missing_ok=True)
        return path
    except OSError as error:
        # A capture that cannot be archived must not fail the measurement.
        log.warning("could not save capture: %s", error)
        return None
