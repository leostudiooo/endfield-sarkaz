from __future__ import annotations

from pathlib import Path
from typing import Iterable

SARKAZ_TABLE = "gkamztlbdqiyfucxbhsjoprnweygtjmevchdxsanqolkrvwiypjzquhe"


def convert_chars_to_skz(text: str) -> str:
    """Convert plain text into Sarkaz cipher letters."""
    return "".join(SARKAZ_TABLE[ord(char) % 56] for char in text)


def read_non_empty_lines(path: str | Path) -> list[str]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_lines_from_many(paths: Iterable[str | Path]) -> list[str]:
    result: list[str] = []
    for item in paths:
        result.extend(read_non_empty_lines(item))
    return result
