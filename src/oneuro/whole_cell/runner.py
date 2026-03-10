"""Job planning and execution support for external whole-cell runtimes."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .adapters import MC4DAdapter, MC4DDependencyStatus, MC4DRunConfig


@dataclass(frozen=True)
class WholeCellJobPlan:
    """Serializable launch plan for an external whole-cell job."""

    job_name: str
    created_at_epoch_s: int
    runtime: str
    repo_root: str
    command: list[str]
    environment: Dict[str, str]
    repo_validation: list[Dict[str, Any]]
    environment_validation: list[Dict[str, Any]]
    output_directory: str
    manifest_path: str


@dataclass(frozen=True)
class WholeCellLaunchResult:
    """Outcome of a launched whole-cell job."""

    plan: WholeCellJobPlan
    returncode: int
    stdout_path: str
    stderr_path: str


class MC4DRunner:
    """Prepare and optionally launch MC4D runs under oNeuro-managed artifacts."""

    def __init__(
        self,
        adapter: MC4DAdapter,
        artifact_root: Path | str = Path("experiments") / "results",
    ) -> None:
        self.adapter = adapter
        self.artifact_root = Path(artifact_root).expanduser().resolve()

    def plan_run(
        self,
        run: MC4DRunConfig,
        env: Mapping[str, str] | None = None,
        job_name: str | None = None,
        artifact_subdir: Path | str | None = None,
    ) -> WholeCellJobPlan:
        """Create a JSON-serializable launch plan and write it to disk."""

        cfg = run.normalized()
        timestamp = int(time.time())
        resolved_job_name = job_name or f"mc4d_{cfg.output_dir}_{timestamp}"
        artifact_dir = (
            self.artifact_root
            if artifact_subdir is None
            else self.artifact_root / Path(artifact_subdir)
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        command = self.adapter.build_command(cfg)
        launch_env = self.adapter.build_environment(base_env=env)
        repo_validation = self.adapter.validate_repo()
        env_validation = self.adapter.validate_environment(env=launch_env)

        manifest_path = artifact_dir / f"{resolved_job_name}_launch_manifest.json"
        plan = WholeCellJobPlan(
            job_name=resolved_job_name,
            created_at_epoch_s=timestamp,
            runtime="mc4d",
            repo_root=str(self.adapter.repo_root),
            command=command,
            environment=launch_env,
            repo_validation=[asdict(status) for status in repo_validation],
            environment_validation=[asdict(status) for status in env_validation],
            output_directory=cfg.output_dir,
            manifest_path=str(manifest_path),
        )
        manifest_path.write_text(json.dumps(asdict(plan), indent=2), encoding="ascii")
        return plan

    def launch(
        self,
        run: MC4DRunConfig,
        env: Mapping[str, str] | None = None,
        job_name: str | None = None,
        artifact_subdir: Path | str | None = None,
        check: bool = False,
    ) -> WholeCellLaunchResult:
        """Launch an external MC4D job and capture stdout/stderr in artifacts."""

        plan = self.plan_run(
            run=run,
            env=env,
            job_name=job_name,
            artifact_subdir=artifact_subdir,
        )
        manifest_path = Path(plan.manifest_path)
        stdout_path = manifest_path.with_suffix(".stdout.log")
        stderr_path = manifest_path.with_suffix(".stderr.log")

        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                plan.command,
                cwd=self.adapter.repo_root,
                env=plan.environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )

        result = WholeCellLaunchResult(
            plan=plan,
            returncode=completed.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                plan.command,
            )
        return result
