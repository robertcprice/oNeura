"""Adapter layer for the published Minimal_Cell_4DWCM runtime."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from oneuro.whole_cell.manifest import WholeCellRuntimeManifest, syn3a_reference_manifest


@dataclass(frozen=True)
class MC4DDependencyStatus:
    """Validation result for one dependency or expected path."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class MC4DRunConfig:
    """Arguments for the MC4D main or restart entrypoint."""

    output_dir: str
    sim_time_seconds: int
    cuda_device: int
    dna_rng_seed: int
    dna_software_directory: Path | str
    working_directory: Optional[Path | str] = None
    python_executable: str = "python3"
    restart: bool = False
    extra_args: Dict[str, str] = field(default_factory=dict)

    def normalized(self) -> "MC4DRunConfig":
        """Return a normalized copy with resolved filesystem paths."""

        return MC4DRunConfig(
            output_dir=self.output_dir,
            sim_time_seconds=int(self.sim_time_seconds),
            cuda_device=int(self.cuda_device),
            dna_rng_seed=int(self.dna_rng_seed),
            dna_software_directory=Path(self.dna_software_directory).expanduser().resolve(),
            working_directory=(
                None
                if self.working_directory is None
                else Path(self.working_directory).expanduser().resolve()
            ),
            python_executable=self.python_executable,
            restart=self.restart,
            extra_args=dict(self.extra_args),
        )


class MC4DAdapter:
    """Bridge object for an external MC4D checkout."""

    def __init__(
        self,
        repo_root: Path | str,
        manifest: WholeCellRuntimeManifest | None = None,
        dna_software_directory: Path | str | None = None,
        conda_env_name: str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.manifest = manifest or syn3a_reference_manifest()
        self.dna_software_directory = (
            None
            if dna_software_directory is None
            else Path(dna_software_directory).expanduser().resolve()
        )
        self.conda_env_name = conda_env_name

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str] | None = None,
        repo_root: Path | None = None,
    ) -> "MC4DAdapter":
        """Construct the adapter from the current process environment."""

        env_map = dict(os.environ if env is None else env)
        manifest = syn3a_reference_manifest()

        if repo_root is None:
            repo_value = env_map.get(manifest.repository_env_var or "")
            if not repo_value:
                raise ValueError(
                    f"Set {manifest.repository_env_var} or pass repo_root explicitly."
                )
            repo_root = Path(repo_value)

        dna_dir_value = env_map.get("ONEURO_MC4D_DNA_SOFTWARE_DIR")
        conda_env_name = env_map.get("ONEURO_MC4D_CONDA_ENV")
        return cls(
            repo_root=repo_root,
            manifest=manifest,
            dna_software_directory=None if not dna_dir_value else Path(dna_dir_value),
            conda_env_name=conda_env_name,
        )

    @property
    def entrypoint_path(self) -> Path:
        """Path to the main MC4D executable."""

        return self.repo_root / self.manifest.entrypoint

    @property
    def restart_entrypoint_path(self) -> Path:
        """Path to the restart MC4D executable."""

        return self.repo_root / self.manifest.restart_entrypoint

    def validate_repo(self) -> List[MC4DDependencyStatus]:
        """Check that the external checkout looks like an MC4D repository."""

        statuses: List[MC4DDependencyStatus] = []
        for rel_path in self.manifest.expected_repo_paths:
            abs_path = self.repo_root / rel_path
            statuses.append(
                MC4DDependencyStatus(
                    name=str(rel_path),
                    ok=abs_path.exists(),
                    detail=f"expected at {abs_path}",
                )
            )
        return statuses

    def validate_environment(
        self,
        env: Mapping[str, str] | None = None,
    ) -> List[MC4DDependencyStatus]:
        """Validate external tool availability and configured directories."""

        env_map = dict(os.environ if env is None else env)
        statuses: List[MC4DDependencyStatus] = []

        for dependency in self.manifest.dependencies:
            if dependency.executable:
                resolved = shutil.which(dependency.executable)
                statuses.append(
                    MC4DDependencyStatus(
                        name=dependency.tool.value,
                        ok=resolved is not None or not dependency.required,
                        detail=(
                            f"resolved executable {resolved}"
                            if resolved is not None
                            else f"missing executable {dependency.executable}"
                        ),
                    )
                )
                continue

            if dependency.env_var and dependency.repo_subdir:
                root_value = env_map.get(dependency.env_var)
                if root_value:
                    subdir = Path(root_value).expanduser().resolve() / dependency.repo_subdir
                    ok = subdir.exists()
                    detail = f"expected at {subdir}"
                else:
                    ok = not dependency.required
                    detail = f"missing environment variable {dependency.env_var}"
                statuses.append(
                    MC4DDependencyStatus(
                        name=dependency.tool.value,
                        ok=ok,
                        detail=detail,
                    )
                )
                continue

            statuses.append(
                MC4DDependencyStatus(
                    name=dependency.tool.value,
                    ok=True,
                    detail=dependency.notes or "manual validation required",
                )
            )

        return statuses

    def build_command(self, run: MC4DRunConfig) -> List[str]:
        """Build a subprocess argv list for the MC4D main or restart entrypoint."""

        cfg = run.normalized()
        script = self.restart_entrypoint_path if cfg.restart else self.entrypoint_path
        command: List[str] = [
            cfg.python_executable,
            str(script),
            "-od",
            cfg.output_dir,
            "-t",
            str(cfg.sim_time_seconds),
            "-cd",
            str(cfg.cuda_device),
            "-drs",
            str(cfg.dna_rng_seed),
            "-dsd",
            str(cfg.dna_software_directory),
        ]

        if cfg.working_directory is not None:
            command.extend(["-wd", str(cfg.working_directory)])

        for key, value in sorted(cfg.extra_args.items()):
            command.extend([key, value])

        return command

    def build_environment(self, base_env: Mapping[str, str] | None = None) -> Dict[str, str]:
        """Build environment variables used to launch an external MC4D run."""

        env = dict(os.environ if base_env is None else base_env)
        env.setdefault("ONEURO_MC4D_REPO", str(self.repo_root))
        if self.dna_software_directory is not None:
            env.setdefault("ONEURO_MC4D_DNA_SOFTWARE_DIR", str(self.dna_software_directory))
        if self.conda_env_name:
            env.setdefault("ONEURO_MC4D_CONDA_ENV", self.conda_env_name)
        return env
