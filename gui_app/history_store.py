from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class HistoryStore:
    def __init__(self, app_home: Path):
        self.app_home = app_home
        self.app_home.mkdir(parents=True, exist_ok=True)
        self.path = self.app_home / "history.json"

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save(self, rows: List[Dict]) -> None:
        self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, row: Dict) -> None:
        rows = self.load()
        rows.insert(0, row)
        self.save(rows[:200])


def build_history_row(
    title: str,
    source_url: str,
    output_language: str,
    status: str,
    output_dir: str,
    audio_path: str = "",
    video_path: str = "",
    error: str = "",
) -> Dict:
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "source_url": source_url,
        "output_language": output_language,
        "status": status,
        "output_dir": output_dir,
        "audio_path": audio_path,
        "video_path": video_path,
        "error": error,
    }
