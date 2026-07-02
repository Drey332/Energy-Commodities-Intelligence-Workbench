"""Export generated outputs, evidence, and provenance manifests."""

from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from devfinintel.models import GeneratedOutput
from devfinintel.utils import file_sha256, slugify, utc_now_iso


@dataclass(frozen=True)
class ExportPaths:
    """Paths created for a generated output."""

    markdown_path: Path
    csv_path: Path
    pdf_path: Path
    evidence_json_path: Path
    manifest_json_path: Path
    package_path: Path


class OutputExporter:
    """Write reviewable files for a generated brief."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, output: GeneratedOutput, file_manifests: list[dict] | None = None) -> ExportPaths:
        """Export one generated output plus its audit materials."""

        timestamp = utc_now_iso().replace(":", "-")
        slug = slugify(output.title)
        base = self.output_dir / f"{timestamp}-{slug}"
        markdown_path = base.with_suffix(".md")
        csv_path = base.with_suffix(".csv")
        pdf_path = base.with_suffix(".pdf")
        evidence_json_path = base.with_suffix(".evidence.json")
        manifest_json_path = base.with_suffix(".manifest.json")
        package_path = base.with_suffix(".zip")

        markdown_path.write_text(output.body_markdown, encoding="utf-8")
        self._write_records_csv(csv_path, output)
        self._write_pdf(pdf_path, output.body_markdown)
        self._write_evidence_json(evidence_json_path, output)

        paths = ExportPaths(
            markdown_path=markdown_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            evidence_json_path=evidence_json_path,
            manifest_json_path=manifest_json_path,
            package_path=package_path,
        )
        self._write_manifest_json(manifest_json_path, output, paths, file_manifests or [])
        self._write_package_zip(package_path, paths)
        return paths

    def _write_records_csv(self, path: Path, output: GeneratedOutput) -> None:
        """Flatten structured records to CSV for spreadsheet review."""

        rows = []
        for record in output.records:
            row = {
                "record_id": record.record_id,
                "record_type": record.record_type,
                "title": record.title,
                "confidence": record.confidence,
                "review_status": record.review_status,
                "evidence_chunk_ids": "; ".join(record.evidence_chunk_ids),
            }
            for key, value in record.fields.items():
                row[key] = export_cell_value(value)
            rows.append(row)

        fieldnames = sorted({key for row in rows for key in row}) if rows else ["message"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            if rows:
                writer.writerows(rows)
            else:
                writer.writerow({"message": "No structured records were produced."})

    def _write_evidence_json(self, path: Path, output: GeneratedOutput) -> None:
        """Write the exact evidence pack and verification findings as JSON.

        This file is the machine-readable audit view. It lets a reviewer inspect
        what was retrieved, what was extracted, and which verification checks ran
        without reading generated prose first.
        """

        data = {
            "schema": "devfinintel.evidence_export.v1",
            "created_at": utc_now_iso(),
            "task_type": output.task_type,
            "title": output.title,
            "metrics": output.metrics,
            "evidence_pack": output.evidence_pack.as_dict() if output.evidence_pack else None,
            "verification_findings": [
                {
                    "level": finding.level,
                    "message": finding.message,
                    "evidence_reference": finding.evidence_reference,
                }
                for finding in output.verification_findings
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _write_manifest_json(
        self,
        path: Path,
        output: GeneratedOutput,
        paths: ExportPaths,
        file_manifests: list[dict],
    ) -> None:
        """Write a provenance-oriented run manifest.

        The manifest borrows ideas from W3C PROV and Croissant without claiming
        full compliance. It gives a non-technical reviewer the important facts:
        inputs, outputs, checksums, analysis activity, and source-grounding
        policy.
        """

        output_paths = [
            paths.markdown_path,
            paths.csv_path,
            paths.pdf_path,
            paths.evidence_json_path,
        ]
        output_files = [
            {
                "path": str(file_path),
                "filename": file_path.name,
                "sha256": file_sha256(file_path),
                "size_bytes": file_path.stat().st_size,
            }
            for file_path in output_paths
        ]
        manifest = {
            "schema": "devfinintel.run_manifest.v1",
            "created_at": utc_now_iso(),
            "task_type": output.task_type,
            "title": output.title,
            "metrics": output.metrics,
            "source_file_manifests": file_manifests,
            "output_files": output_files,
            "provenance": build_prov_like_graph(output, output_files, file_manifests),
            "croissant_style_metadata": build_croissant_style_metadata(output, file_manifests),
            "design_notes": [
                "Facts are extracted into structured records before prose is drafted.",
                "Local LLM follow-up receives only the bounded evidence pack, not unrestricted file access.",
                "Numbers, citations, claim support, schema validity, and record traceability are checked before export.",
            ],
        }
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _write_package_zip(self, path: Path, paths: ExportPaths) -> None:
        """Bundle all output files into one review package."""

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in (
                paths.markdown_path,
                paths.csv_path,
                paths.pdf_path,
                paths.evidence_json_path,
                paths.manifest_json_path,
            ):
                archive.write(file_path, arcname=file_path.name)

    def _write_pdf(self, path: Path, markdown: str) -> None:
        """Write a simple PDF.

        ReportLab is used when installed. A tiny fallback PDF writer is included
        so the export contract still works in minimal environments.
        """

        try:
            from reportlab.lib.pagesizes import letter  # type: ignore
            from reportlab.pdfgen import canvas  # type: ignore

            pdf = canvas.Canvas(str(path), pagesize=letter)
            width, height = letter
            y = height - 48
            pdf.setFont("Helvetica", 10)
            for raw_line in markdown_to_plain_lines(markdown):
                for line in wrap_line(raw_line, max_chars=95):
                    if y < 48:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 10)
                        y = height - 48
                    pdf.drawString(48, y, line)
                    y -= 14
            pdf.save()
            return
        except Exception:
            self._write_minimal_pdf(path, markdown)

    def _write_minimal_pdf(self, path: Path, markdown: str) -> None:
        """Write a minimal one-page PDF without external dependencies."""

        lines = []
        for raw_line in markdown_to_plain_lines(markdown):
            lines.extend(wrap_line(raw_line, max_chars=90))
        lines = lines[:48]
        stream_lines = ["BT", "/F1 10 Tf", "50 750 Td"]
        for index, line in enumerate(lines):
            safe = pdf_escape(line)
            if index == 0:
                stream_lines.append(f"({safe}) Tj")
            else:
                stream_lines.append(f"0 -14 Td ({safe}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")

        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]

        content = bytearray(b"%PDF-1.4\n")
        offsets = []
        for i, obj in enumerate(objects, start=1):
            offsets.append(len(content))
            content.extend(f"{i} 0 obj\n".encode("ascii"))
            content.extend(obj)
            content.extend(b"\nendobj\n")
        xref_offset = len(content)
        content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        content.extend(b"0000000000 65535 f \n")
        for offset in offsets:
            content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        content.extend(
            (
                f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(bytes(content))


def markdown_to_plain_lines(markdown: str) -> list[str]:
    """Convert Markdown to plain lines for simple PDF output."""

    lines = []
    for line in markdown.splitlines():
        plain = re.sub(r"^#+\s*", "", line)
        plain = plain.replace("**", "")
        lines.append(plain)
    return lines


def wrap_line(line: str, max_chars: int) -> list[str]:
    """Wrap a line without needing a rich text layout engine."""

    if len(line) <= max_chars:
        return [line]
    words = line.split()
    wrapped: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            wrapped.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        wrapped.append(current)
    return wrapped or [""]


def pdf_escape(text: str) -> str:
    """Escape characters that have special meaning inside PDF text strings."""

    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def export_cell_value(value) -> str:
    """Convert nested structured values into safe CSV cell text."""

    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return "; ".join(str(item) for item in value)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def build_prov_like_graph(
    output: GeneratedOutput,
    output_files: list[dict],
    file_manifests: list[dict],
) -> dict:
    """Build a compact W3C PROV-inspired graph for the run manifest."""

    pack_id = output.evidence_pack.pack_id if output.evidence_pack else "no-evidence-pack"
    entities = {
        f"source:{manifest['document_id']}": {
            "prov:type": "SourceFile",
            "title": manifest.get("title"),
            "path": manifest.get("source_path"),
            "sha256": manifest.get("file_sha256"),
        }
        for manifest in file_manifests
    }
    entities.update(
        {
            f"output:{file_info['filename']}": {
                "prov:type": "GeneratedFile",
                "path": file_info["path"],
                "sha256": file_info["sha256"],
            }
            for file_info in output_files
        }
    )
    return {
        "@context": {
            "prov": "http://www.w3.org/ns/prov#",
            "devfinintel": "https://example.local/devfinintel#",
        },
        "entity": entities,
        "activity": {
            f"analysis:{pack_id}": {
                "prov:type": "EvidenceGroundedAnalysis",
                "task_type": output.task_type,
                "title": output.title,
                "evidence_items": int(output.metrics.get("evidence_items", 0)),
                "structured_records": int(output.metrics.get("structured_records", 0)),
            }
        },
        "agent": {
            "agent:local-user": {
                "prov:type": "Person",
                "name": "local-user",
            },
            "agent:devfinintel": {
                "prov:type": "SoftwareAgent",
                "name": "Development Finance Intelligence Workbench",
            },
        },
    }


def build_croissant_style_metadata(output: GeneratedOutput, file_manifests: list[dict]) -> dict:
    """Return dataset-style metadata inspired by Croissant.

    It is intentionally labelled as "style" metadata because the project has not
    implemented the complete Croissant JSON-LD specification yet.
    """

    return {
        "@context": "https://mlcommons.org/croissant/context.jsonld",
        "@type": "sc:Dataset",
        "name": output.title,
        "description": "Evidence-grounded analysis package generated from local uploaded files.",
        "distribution": [
            {
                "name": manifest.get("title"),
                "contentUrl": manifest.get("source_path"),
                "sha256": manifest.get("file_sha256"),
                "encodingFormat": manifest.get("source_type"),
            }
            for manifest in file_manifests
        ],
        "recordSet": [
            {
                "name": record.record_type,
                "description": record.title,
                "sourceEvidence": record.evidence_chunk_ids,
            }
            for record in output.records
        ],
    }
