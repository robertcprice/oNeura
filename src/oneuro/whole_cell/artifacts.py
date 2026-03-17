"""Artifact ingestion for external whole-cell runtimes."""

from __future__ import annotations

import csv
import json
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class WholeCellArtifactSummary:
    """Standardized summary of a whole-cell result bundle."""

    source_path: str
    source_kind: str
    created_at_epoch_s: int
    file_count: int
    total_bytes: int
    extension_counts: Dict[str, int]
    sample_members: List[str]
    csv_summaries: List[Dict[str, Any]]


class WholeCellArtifactIngestor:
    """Summarize MC4D-style outputs into oNeura JSON artifacts."""

    def __init__(
        self,
        artifact_root: Path | str = Path("experiments") / "results",
    ) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve()

    def ingest(
        self,
        source: Path | str,
        name: str | None = None,
        artifact_subdir: Path | str | None = None,
    ) -> WholeCellArtifactSummary:
        """Ingest a directory, tarball, zip, or file and write a JSON summary."""

        path = Path(source).expanduser().resolve()
        timestamp = int(time.time())
        summary = self._summarize(path, timestamp)

        artifact_dir = (
            self.artifact_root
            if artifact_subdir is None
            else self.artifact_root / Path(artifact_subdir)
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stem = name or path.stem.replace(".", "_")
        out_path = artifact_dir / f"{stem}_whole_cell_artifact_{timestamp}.json"
        out_path.write_text(json.dumps(asdict(summary), indent=2), encoding="ascii")
        return summary

    def _summarize(self, path: Path, timestamp: int) -> WholeCellArtifactSummary:
        if path.is_dir():
            return self._summarize_directory(path, timestamp)
        if path.suffix == ".zip":
            return self._summarize_zip(path, timestamp)
        if path.suffixes[-2:] == [".tar", ".gz"]:
            return self._summarize_tar(path, timestamp)
        return self._summarize_file(path, timestamp)

    def _summarize_directory(self, path: Path, timestamp: int) -> WholeCellArtifactSummary:
        members = [p for p in sorted(path.rglob("*")) if p.is_file()]
        return self._build_summary(
            source_path=path,
            source_kind="directory",
            timestamp=timestamp,
            members=[str(member.relative_to(path)) for member in members],
            file_sizes=[member.stat().st_size for member in members],
            csv_summaries=self._collect_directory_csv_summaries(path, members),
        )

    def _summarize_tar(self, path: Path, timestamp: int) -> WholeCellArtifactSummary:
        with tarfile.open(path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            names = [member.name for member in members]
            sizes = [member.size for member in members]
            csv_summaries = self._collect_archive_csv_summaries(
                names,
                lambda name: archive.extractfile(name),
            )
        return self._build_summary(
            source_path=path,
            source_kind="tar.gz",
            timestamp=timestamp,
            members=names,
            file_sizes=sizes,
            csv_summaries=csv_summaries,
        )

    def _summarize_zip(self, path: Path, timestamp: int) -> WholeCellArtifactSummary:
        with zipfile.ZipFile(path, "r") as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            names = [info.filename for info in members]
            sizes = [info.file_size for info in members]
            csv_summaries = self._collect_archive_csv_summaries(
                names,
                lambda name: archive.open(name, "r"),
            )
        return self._build_summary(
            source_path=path,
            source_kind="zip",
            timestamp=timestamp,
            members=names,
            file_sizes=sizes,
            csv_summaries=csv_summaries,
        )

    def _summarize_file(self, path: Path, timestamp: int) -> WholeCellArtifactSummary:
        csv_summaries: List[Dict[str, Any]] = []
        if path.suffix.lower() == ".csv":
            csv_summaries.append(self._read_csv_summary(path.name, path.open("r", encoding="utf-8")))
        return self._build_summary(
            source_path=path,
            source_kind="file",
            timestamp=timestamp,
            members=[path.name],
            file_sizes=[path.stat().st_size],
            csv_summaries=csv_summaries,
        )

    def _build_summary(
        self,
        source_path: Path,
        source_kind: str,
        timestamp: int,
        members: List[str],
        file_sizes: List[int],
        csv_summaries: List[Dict[str, Any]],
    ) -> WholeCellArtifactSummary:
        extension_counts: Dict[str, int] = {}
        for member in members:
            suffix = "".join(Path(member).suffixes) or "<no_ext>"
            extension_counts[suffix] = extension_counts.get(suffix, 0) + 1

        return WholeCellArtifactSummary(
            source_path=str(source_path),
            source_kind=source_kind,
            created_at_epoch_s=timestamp,
            file_count=len(members),
            total_bytes=sum(file_sizes),
            extension_counts=extension_counts,
            sample_members=members[:10],
            csv_summaries=csv_summaries[:10],
        )

    def _collect_directory_csv_summaries(
        self,
        root: Path,
        members: Iterable[Path],
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for member in members:
            if member.suffix.lower() != ".csv":
                continue
            with member.open("r", encoding="utf-8") as handle:
                summaries.append(
                    self._read_csv_summary(str(member.relative_to(root)), handle)
                )
        return summaries

    def _collect_archive_csv_summaries(
        self,
        members: Iterable[str],
        opener,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for member in members:
            if Path(member).suffix.lower() != ".csv":
                continue
            raw_handle = opener(member)
            if raw_handle is None:
                continue
            with raw_handle as handle:
                text = (line.decode("utf-8") for line in handle)
                summaries.append(self._read_csv_summary(member, text))
        return summaries

    def _read_csv_summary(self, name: str, handle) -> Dict[str, Any]:
        reader = csv.reader(handle)
        header = next(reader, [])
        row_count = sum(1 for _ in reader)
        return {
            "name": name,
            "column_count": len(header),
            "columns": header[:20],
            "row_count": row_count,
        }
