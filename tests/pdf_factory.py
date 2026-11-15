"""Build tiny but genuinely valid PDFs for tests, with no external dependency."""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Return a minimal PDF 1.4 document with one page per list of text lines."""
    if not pages:
        raise ValueError("need at least one page")

    streams: list[str] = []
    for lines in pages:
        ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj")
            ops.append("T*")
        ops.append("ET")
        streams.append("\n".join(ops))

    count = len(pages)
    page_ids = [4 + i for i in range(count)]
    content_ids = [4 + count + i for i in range(count)]

    objects: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [{}] /Count {} >>".format(
            " ".join(f"{pid} 0 R" for pid in page_ids), count
        ),
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    for index, page_id in enumerate(page_ids):
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_ids[index]} 0 R >>"
        )
    for index, content_id in enumerate(content_ids):
        stream = streams[index]
        objects[content_id] = (
            f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n{objects[number]}\nendobj\n".encode("latin-1")

    xref_at = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for number in range(1, size):
        if number in offsets:
            out += f"{offsets[number]:010d} 00000 n \n".encode()
        else:
            out += b"0000000000 65535 f \n"
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)
