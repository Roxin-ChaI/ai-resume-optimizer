"""Shared pytest configuration and deterministic temporary-file factories."""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path

import pytest
from docx import Document
from docx.document import Document as DocumentObject


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_stream(text: str) -> bytes:
    commands = ["BT", "/F1 12 Tf", "72 720 Td"]
    for line_number, line in enumerate(text.splitlines() or [text]):
        if line_number:
            commands.append("0 -18 Td")
        if line:
            commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1")


def _build_pdf(pages: list[str]) -> bytes:
    font_object_number = 3 + (2 * len(pages))
    page_object_numbers = [3 + (2 * index) for index in range(len(pages))]
    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
    ]
    for index, text in enumerate(pages):
        page_number = page_object_numbers[index]
        content_number = page_number + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
                f"/Contents {content_number} 0 R >>"
            ).encode()
        )
        stream = _pdf_stream(text)
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately if a test attempts to open an external connection."""

    def blocked_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access is not allowed during tests.")

    monkeypatch.setattr(socket, "create_connection", blocked_connection)


@pytest.fixture
def pdf_factory(tmp_path: Path) -> Callable[[list[str], str], Path]:
    """Create a deterministic, minimal text-layer PDF under tmp_path."""

    def create(pages: list[str], filename: str = "resume.pdf") -> Path:
        path = tmp_path / filename
        path.write_bytes(_build_pdf(pages))
        return path

    return create


@pytest.fixture
def docx_factory(
    tmp_path: Path,
) -> Callable[[Callable[[DocumentObject], None] | None, str], Path]:
    """Create a deterministic DOCX under tmp_path."""

    def create(
        builder: Callable[[DocumentObject], None] | None = None,
        filename: str = "resume.docx",
    ) -> Path:
        document = Document()
        if builder is not None:
            builder(document)
        path = tmp_path / filename
        document.save(path)
        return path

    return create


@pytest.fixture
def text_file_factory(tmp_path: Path) -> Callable[[str | bytes, str], Path]:
    """Create a text or byte fixture under tmp_path."""

    def create(content: str | bytes, filename: str = "job_description.txt") -> Path:
        path = tmp_path / filename
        data = content if isinstance(content, bytes) else content.encode()
        path.write_bytes(data)
        return path

    return create
