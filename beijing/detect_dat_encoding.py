from __future__ import annotations

from pathlib import Path


ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "gb2312",
    "big5",
]

EXPECTED_WORDS = (
    "\u53f0\u98ce",  # tai feng
    "\u8def\u5f84",  # lu jing
    "\u53f7",        # hao
    "diamond",
)

MOJIBAKE_MARKERS = (
    "\u93c3",
    "\u7490",
    "\u9359",
    "\u95c3",
    "\u6d5c",
    "\u00c2",
    "\u00c3",
)


def score_text(text: str) -> int:
    lines = text.splitlines()
    first_line = lines[0] if lines else text
    score = 0

    for keyword in EXPECTED_WORDS:
        if keyword in first_line:
            score += 10

    score -= first_line.count("\ufffd") * 20
    score -= sum(first_line.count(marker) * 8 for marker in MOJIBAKE_MARKERS)

    chinese_count = sum("\u4e00" <= ch <= "\u9fff" for ch in first_line)
    printable_count = sum(ch.isprintable() or ch in "\r\n\t" for ch in first_line)
    score += chinese_count * 2 + printable_count
    return score


def detect_encoding(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    candidates: list[tuple[int, str, str]] = []

    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        candidates.append((score_text(text), encoding, text))

    if not candidates:
        text = data.decode("utf-8", errors="replace")
        return "unknown", text

    candidates.sort(reverse=True, key=lambda item: item[0])
    _, encoding, text = candidates[0]
    return encoding, text


def main() -> None:
    dat_dir = Path(__file__).resolve().parent
    dat_files = sorted(dat_dir.glob("*.dat"))

    if not dat_files:
        print(f"No .dat files found in: {dat_dir}")
        return

    for index, path in enumerate(dat_files):
        encoding, text = detect_encoding(path)
        lines = text.splitlines()

        if index:
            print()
        print("=" * 80)
        print(f"file: {path}")
        print(f"encoding: {encoding}")
        print(f"first line: {lines[0] if lines else ''}")
        print("preview:")
        print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
