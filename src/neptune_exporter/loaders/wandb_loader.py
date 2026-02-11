#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import logging
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Generator, Optional, Any, Union
import pandas as pd
import pyarrow as pa
import wandb

from neptune_exporter.types import ProjectId, TargetRunId, TargetExperimentId
from neptune_exporter.loaders.loader import DataLoader

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
HTML_EXTENSIONS = {".html", ".htm"}


def _is_image(filename: Union[str, Path]) -> bool:
    """Check if a file is an image based on its extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def _is_html(filename: Union[str, Path]) -> bool:
    """Check if a file is HTML based on its extension."""
    return Path(filename).suffix.lower() in HTML_EXTENSIONS


def _extract_param_value(row: pd.Series) -> Any:
    """Extract the parameter value from a parquet row based on its attribute_type."""
    attr_type = row["attribute_type"]
    if attr_type == "float" and pd.notna(row["float_value"]):
        return row["float_value"]
    elif attr_type == "int" and pd.notna(row["int_value"]):
        return int(row["int_value"])
    elif attr_type == "string" and pd.notna(row["string_value"]):
        return row["string_value"]
    elif attr_type == "bool" and pd.notna(row["bool_value"]):
        return bool(row["bool_value"])
    elif attr_type == "datetime" and pd.notna(row["datetime_value"]):
        return str(row["datetime_value"])
    elif attr_type == "string_set" and row["string_set_value"] is not None:
        return list(row["string_set_value"])
    return None


def _deep_set(d: dict, keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys.

    If an intermediate key already exists as a non-dict (scalar), it is overwritten
    with a dict to support the deeper path.
    """
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


_NEPTUNE_SYSTEM_PREFIXES = ("sys/", "monitoring/", "metaflow/", "source_code/")

CHECKPOINT_PATH_PREFIXES = (
    "model/checkpoints/",
    "checkpoint/files/",
    "checkpoint/",
    "checkpoints/",
)

CHECKPOINT_NAME_RE = re.compile(r"epoch[_=](\d+)(?:[-_]step[_=](\d+))?")


def _is_checkpoint_path(cleaned_path: str) -> bool:
    """Check if a cleaned attribute path refers to a checkpoint file."""
    return any(cleaned_path.startswith(prefix) for prefix in CHECKPOINT_PATH_PREFIXES)


def _is_onnx_path(cleaned_path: str) -> bool:
    """Check if a cleaned attribute path refers to an ONNX file."""
    return "onnx" in cleaned_path.lower()


def _parse_checkpoint_stem(stem: str) -> tuple[int | None, int | None]:
    """Parse epoch and step from a checkpoint filename stem.

    Handles both old (=) and new (_) conventions:
    - epoch=0035 -> (35, None)
    - epoch=16-step=34000 -> (16, 34000)
    - model_step=002756 -> (None, 2756)
    - epoch_00010-step_00001000 -> (10, 1000)
    - last -> (None, None)
    """
    match = CHECKPOINT_NAME_RE.search(stem)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2)) if match.group(2) else None
        return epoch, step
    # Handle model_step=XXXX (GIM convention)
    step_match = re.search(r"(?:model_)?step[_=](\d+)", stem)
    if step_match:
        return None, int(step_match.group(1))
    return None, None


SUNSTONE_CHECKPOINT_FMT = "epoch_{epoch:05d}-step_{step:08d}"


def _to_sunstone_checkpoint_name(stem: str) -> str:
    """Convert a Neptune checkpoint stem to Sunstone's naming convention.

    Sunstone expects: epoch_XXXXX-step_XXXXXXXX
    Neptune uses: epoch=XXXX, epoch=XX-step=XXXXX, model_step=XXXXX

    Names that don't match any known pattern (e.g., "last") are returned as-is.
    Names already in Sunstone format are returned unchanged.
    """
    epoch, step = _parse_checkpoint_stem(stem)
    if epoch is None and step is None:
        return stem
    return SUNSTONE_CHECKPOINT_FMT.format(epoch=epoch or 0, step=step or 0)


def _ckpt_sort_key(item: tuple[str, Path]) -> tuple[float, float]:
    """Sort key for checkpoint files by (epoch, step).

    epoch=None maps to 0 so step-only checkpoints (e.g. GIM's model_step=2756) sort
    before all epoch-based checkpoints. This is acceptable because no single run mixes
    both conventions.
    """
    cleaned_path, _file_path = item
    stem = Path(cleaned_path).name
    if stem == "last":
        return (float("inf"), float("inf"))
    epoch, step = _parse_checkpoint_stem(stem)
    return (epoch if epoch is not None else 0, step if step is not None else 0)


class WandBLoader(DataLoader):
    """Loads Neptune data from parquet files into Weights & Biases."""

    def __init__(
        self,
        entity: str,
        api_key: Optional[str] = None,
        name_prefix: Optional[str] = None,
        show_client_logs: bool = False,
    ):
        """
        Initialize W&B loader.

        Args:
            entity: W&B entity (organization/username)
            api_key: Optional W&B API key for authentication
            name_prefix: Optional prefix for project and run names
            show_client_logs: Enable W&B client logging
        """
        self.entity = entity
        self.name_prefix = name_prefix
        self._logger = logging.getLogger(__name__)
        self._active_run: Optional[wandb.Run] = None
        self._current_run_name: Optional[str] = None
        self._pending_tags: set[str] = set()
        self._pending_checkpoints: list[tuple[str, Path]] = []
        self._pending_onnx: list[tuple[str, Path]] = []

        # Authenticate with W&B
        if api_key:
            wandb.login(key=api_key)

        # Configure W&B logging
        if not show_client_logs:
            os.environ["WANDB_SILENT"] = "true"

    def _sanitize_attribute_name(self, attribute_path: str) -> str:
        """Sanitize Neptune attribute path to W&B-compatible metric/config key.

        Preserves "/" separators since W&B uses them for hierarchical section
        grouping in the UI (e.g. "training/val/loss" renders in a "training" section).
        """
        # Replace invalid characters with underscores, but preserve "/"
        sanitized = re.sub(r"[^a-zA-Z0-9_/]", "_", attribute_path)

        # Clean up slashes: strip leading/trailing, collapse doubles
        sanitized = sanitized.strip("/")
        sanitized = re.sub(r"/+", "/", sanitized)

        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized

        # Handle empty result
        if not sanitized:
            sanitized = "_attribute"

        return sanitized

    def _clean_attribute_path(self, path: str) -> str:
        """Strip Lightning's NeptuneLogger 'training/' prefix from attribute paths.

        Lightning's NeptuneLogger defaults to prefix='training', which prepends
        'training/' to all logged metrics, hyperparams, and checkpoints. Stripping
        this makes migrated runs comparable with natively-logged W&B runs.
        """
        if path.startswith("training/"):
            return path[len("training/") :]
        return path

    def _make_artifact_name(self, attribute_path: str, run_name: str, suffix: str = "") -> str:
        """Create a W&B artifact name from Neptune attribute path and run name.

        Artifact names cannot contain "/" so we use "__" to preserve hierarchy
        while avoiding collisions (e.g. "foo/bar" -> "foo__bar", "foo-bar" stays).
        Names are scoped per run to prevent cross-run collisions.
        """
        # Replace "/" with "__" to preserve hierarchy without collisions
        sanitized = re.sub(r"/+", "__", attribute_path)
        # Replace remaining invalid characters
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", sanitized)
        sanitized = sanitized.strip("_.-")

        name = f"{sanitized}-{run_name}"
        if suffix:
            name = f"{name}-{suffix}"

        # W&B artifact names have a 128-char limit
        if len(name) > 128:
            # Truncate the path part, keep run_name and suffix intact
            tail = f"-{run_name}"
            if suffix:
                tail = f"-{run_name}-{suffix}"
            max_path_len = 128 - len(tail)
            name = f"{sanitized[:max_path_len]}{tail}"

        return name

    def _get_project_name(self, project_id: str) -> str:
        """Get W&B project name from Neptune project ID.

        Strips the Neptune org prefix (e.g. "veo-ai/my-project" -> "my-project")
        since the W&B entity (org) is passed separately via --wandb-entity.
        """
        # Strip Neptune org prefix — W&B entity is set separately
        name = project_id.split("/")[-1] if "/" in project_id else project_id

        if self.name_prefix:
            name = f"{self.name_prefix}_{name}"

        # Sanitize for W&B project name (alphanumeric, hyphens, underscores)
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        return name

    def _convert_step_to_int(self, step: Decimal, step_multiplier: int) -> int:
        """Convert Neptune decimal step to W&B integer step."""
        if step is None:
            return 0
        return int(float(step) * step_multiplier)

    def create_experiment(self, project_id: str, experiment_name: str) -> TargetExperimentId:
        """
        Neptune experiment_name maps to W&B group (set in create_run).
        We return the experiment name as the group name to use.
        """
        return TargetExperimentId(experiment_name)

    def find_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId],
    ) -> Optional[TargetRunId]:
        """Find a run by name in a W&B project.

        Args:
            run_name: Name of the run to find
            experiment_id: W&B group name (experiment name from Neptune)
            project_id: Neptune project ID (used to construct W&B project name)

        Returns:
            W&B run ID if found, None otherwise
        """
        sanitized_project = self._get_project_name(project_id)

        try:
            # Use wandb.Api() to search for runs
            api = wandb.Api()
            project_path = f"{self.entity}/{sanitized_project}"

            # Search for runs with matching name and group
            filters = {"display_name": run_name}
            if experiment_id:
                filters["group"] = experiment_id

            runs = api.runs(project_path, filters=filters, per_page=1)

            # Get the first matching run
            for run in runs:
                return TargetRunId(run.id)

            return None
        except Exception:
            self._logger.error(
                f"Error finding project {project_id}, run '{run_name}'",
                exc_info=True,
            )
            return None

    def create_run(
        self,
        project_id: ProjectId,
        run_name: str,
        experiment_id: Optional[TargetExperimentId] = None,
        parent_run_id: Optional[TargetRunId] = None,
        fork_step: Optional[float] = None,
        step_multiplier: Optional[int] = None,
    ) -> TargetRunId:
        """Create W&B run, with support for forked runs.

        Args:
            fork_step: Fork step as float (decimal). Will be converted to int using step_multiplier.
            step_multiplier: Step multiplier for converting decimal steps to integers.
                If provided, will be used for fork_step conversion. If not provided,
                will calculate from fork_step alone as fallback.
        """
        sanitized_project = self._get_project_name(project_id)

        try:
            # Prepare init arguments
            init_kwargs: dict[str, Any] = {
                "entity": self.entity,
                "project": sanitized_project,
                "group": experiment_id,
                "name": run_name,
            }

            # Handle forking if parent exists
            if parent_run_id:
                # Convert fork_step to int using provided step_multiplier
                # step_multiplier should always be provided when fork_step is set
                if fork_step is not None:
                    if step_multiplier is None:
                        raise ValueError("step_multiplier must be provided when fork_step is set")
                    step_int = self._convert_step_to_int(Decimal(str(fork_step)), step_multiplier)
                else:
                    step_int = 0

                # W&B fork format: run_id?_step=step
                # https://docs.wandb.ai/models/runs/forking
                fork_from = f"{parent_run_id}?_step={step_int}"
                init_kwargs["fork_from"] = fork_from
                self._logger.info(f"Creating forked run '{run_name}' from parent {parent_run_id} at step {step_int}")

            # Initialize the run
            run = wandb.init(**init_kwargs)
            wandb_run_id = run.id

            self._active_run = run
            self._current_run_name = run_name

            self._logger.info(f"Created run '{run_name}' with W&B ID {wandb_run_id}")
            return TargetRunId(wandb_run_id)

        except Exception:
            self._logger.error(
                f"Error creating project {project_id}, run '{run_name}'",
                exc_info=True,
            )
            raise

    def upload_run_data(
        self,
        run_data: Generator[pa.Table, None, None],
        run_id: TargetRunId,
        files_directory: Path,
        step_multiplier: int,
    ) -> None:
        """Upload all data for a single run to W&B.

        Args:
            step_multiplier: Step multiplier for converting decimal steps to integers
        """
        try:
            # Note: We assume the run is already active from create_run
            # If not, we would need to resume it
            if self._active_run is None or self._active_run.id != run_id:
                self._logger.error(f"Run {run_id} is not active. Call create_run first.")
                raise RuntimeError(f"Run {run_id} is not active")

            for run_data_part in run_data:
                run_df = run_data_part.to_pandas()

                self.upload_parameters(run_df, run_id)
                self.upload_metrics(run_df, run_id, step_multiplier)
                self.upload_artifacts(run_df, run_id, files_directory, step_multiplier)

            # Upload accumulated checkpoints/ONNX in correct sorted order
            self._flush_checkpoint_artifacts()

            # Set tags accumulated from all data chunks
            if self._pending_tags:
                self._active_run.tags = tuple(self._pending_tags)

            # Finish the run
            self._active_run.finish()
            self._active_run = None
            self._current_run_name = None
            self._pending_tags = set()
            self._pending_checkpoints = []
            self._pending_onnx = []

            self._logger.info(f"Successfully uploaded run {run_id} to W&B")

        except Exception:
            self._logger.error(f"Error uploading data for run {run_id}", exc_info=True)
            if self._active_run:
                try:
                    self._active_run.finish(exit_code=1)
                except Exception:
                    self._logger.warning("Failed to finish run on error", exc_info=True)
            self._active_run = None
            self._current_run_name = None
            self._pending_tags = set()
            self._pending_checkpoints = []
            self._pending_onnx = []
            raise

    def _flush_checkpoint_artifacts(self) -> None:
        """Upload accumulated checkpoint and ONNX artifacts in globally sorted order.

        Called after all parquet chunks have been processed, ensuring checkpoints
        are uploaded in correct (epoch, step) order regardless of chunk boundaries.
        """
        if self._active_run is None:
            return

        # Sort all checkpoints globally by (epoch, step)
        self._pending_checkpoints.sort(key=_ckpt_sort_key)

        for cleaned_path, file_path in self._pending_checkpoints:
            if not file_path.exists() or not file_path.is_file():
                self._logger.warning(f"Checkpoint file not found: {file_path}")
                continue

            raw_stem = Path(cleaned_path).name
            stem = _to_sunstone_checkpoint_name(raw_stem)
            epoch, step = _parse_checkpoint_stem(raw_stem)
            is_last = raw_stem == "last"

            artifact = wandb.Artifact(
                name=f"checkpoints-{self._active_run.id}",
                type="checkpoint",
                metadata={
                    "filename": stem,
                    "epoch": epoch,
                    "step": step,
                    "is_last": is_last,
                },
            )
            upload_name = f"{stem}.ckpt" if not Path(stem).suffix else stem
            artifact.add_file(str(file_path), name=upload_name)

            aliases = ["latest"]
            if not is_last:
                aliases.append(stem)
            if is_last:
                aliases.append("last")

            self._active_run.log_artifact(artifact, aliases=aliases)

        # Upload ONNX files without sorting (no epoch/step semantics)
        for cleaned_path, file_path in self._pending_onnx:
            if not file_path.exists() or not file_path.is_file():
                self._logger.warning(f"ONNX file not found: {file_path}")
                continue

            stem = Path(cleaned_path).stem
            epoch, step = _parse_checkpoint_stem(stem)
            artifact = wandb.Artifact(
                name=f"onnx-{self._active_run.id}",
                type="onnx",
                metadata={"filename": stem, "epoch": epoch, "step": step},
            )
            upload_name = f"{stem}.onnx" if not Path(stem).suffix else stem
            artifact.add_file(str(file_path), name=upload_name)

            aliases = ["latest", stem]
            self._active_run.log_artifact(artifact, aliases=aliases)

        self._pending_checkpoints = []
        self._pending_onnx = []

    def _upload_best_model_metadata(self, param_data: pd.DataFrame) -> set[str]:
        """Extract best model metadata from params and write to run.summary.

        Looks for:
        - model/best_model_path (string) — full path, we extract the stem
        - model/best_model_score (float) — metric value
        - config/model/best_model_monitor (string, optional) — metric name

        Writes to run.summary matching Sunstone's WandBTracker convention:
        - metadata/checkpoint/best_model_name_{monitor} = checkpoint stem
        - metadata/checkpoint/best_model_score_{monitor} = score

        Returns set of original (uncleaned) attribute paths that were handled,
        so upload_parameters can exclude them from config.
        """
        if self._active_run is None:
            return set()

        handled_paths: set[str] = set()
        best_path_value: str | None = None
        best_score_value: float | None = None
        monitor_value: str | None = None

        for _, row in param_data.iterrows():
            cleaned = self._clean_attribute_path(row["attribute_path"])
            if cleaned == "model/best_model_path":
                value = _extract_param_value(row)
                if isinstance(value, str) and value:
                    best_path_value = value
                    handled_paths.add(row["attribute_path"])
            elif cleaned == "model/best_model_score":
                value = _extract_param_value(row)
                if isinstance(value, (int, float)):
                    best_score_value = float(value)
                    handled_paths.add(row["attribute_path"])
            elif cleaned == "config/model/best_model_monitor":
                value = _extract_param_value(row)
                if isinstance(value, str) and value:
                    monitor_value = value
                    handled_paths.add(row["attribute_path"])

        if best_path_value is None:
            return handled_paths

        # Extract stem from path: "/checkpoints/model_step=002756.ckpt" -> "model_step=002756"
        raw_stem = Path(best_path_value).stem
        stem = _to_sunstone_checkpoint_name(raw_stem)

        # Sanitize monitor for use as key suffix (replace / with _)
        monitor_sanitized = monitor_value.replace("/", "_") if monitor_value else "unknown"

        self._active_run.summary[f"metadata/checkpoint/best_model_name_{monitor_sanitized}"] = stem
        if best_score_value is not None:
            self._active_run.summary[f"metadata/checkpoint/best_model_score_{monitor_sanitized}"] = best_score_value

        self._logger.info(
            f"Set best model metadata: monitor={monitor_sanitized}, name={stem}, score={best_score_value}"
        )

        return handled_paths

    def upload_parameters(self, run_data: pd.DataFrame, run_id: TargetRunId) -> None:
        """Upload parameters (configs) to W&B run.

        Routes parameters by cleaned path prefix:
        - config/*: stripped and nested into run.config as dicts
        - sys/tags, sys/group_tags: collected into pending tags for run.tags
        - sys/*, monitoring/*, metaflow/*, source_code/*: namespaced under _neptune/ in config
        - everything else: flat keys in config
        """
        if self._active_run is None:
            raise RuntimeError("No active run")

        param_types = {"float", "int", "string", "bool", "datetime", "string_set"}
        param_data = run_data[run_data["attribute_type"].isin(param_types)]

        if param_data.empty:
            return

        # Extract and write best model metadata to run.summary (before routing to config)
        handled_paths = self._upload_best_model_metadata(param_data)

        config_nested: dict = {}
        neptune_meta: dict = {}
        config_flat: dict = {}

        for _, row in param_data.iterrows():
            if row["attribute_path"] in handled_paths:
                continue

            cleaned = self._clean_attribute_path(row["attribute_path"])
            value = _extract_param_value(row)
            if value is None:
                continue

            if cleaned.startswith("config/"):
                # Strip "config/" prefix and build nested dict
                key_path = cleaned[len("config/") :]
                _deep_set(config_nested, key_path.split("/"), value)

            elif cleaned in ("sys/tags", "sys/group_tags"):
                # Collect tags for run.tags (set later in upload_run_data)
                if isinstance(value, list):
                    self._pending_tags.update(value)

            elif cleaned.startswith(_NEPTUNE_SYSTEM_PREFIXES):
                # Namespace Neptune system metadata under _neptune/
                _deep_set(neptune_meta, ["_neptune"] + cleaned.split("/"), value)

            else:
                # Everything else: flat key (e.g., "status", "hyperparams/lr")
                attr_name = self._sanitize_attribute_name(cleaned)
                config_flat[attr_name] = value

        if config_nested:
            self._active_run.config.update(config_nested)
        if neptune_meta:
            self._active_run.config.update(neptune_meta)
        if config_flat:
            self._active_run.config.update(config_flat)

        param_count = len(config_nested) + len(neptune_meta) + len(config_flat)
        if param_count:
            self._logger.info(f"Uploaded {param_count} parameters for run {run_id}")

    def upload_metrics(self, run_data: pd.DataFrame, run_id: TargetRunId, step_multiplier: int) -> None:
        """Upload metrics (float series) to W&B run.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._active_run is None:
            raise RuntimeError("No active run")

        metrics_data = run_data[run_data["attribute_type"] == "float_series"]

        if metrics_data.empty:
            return

        # Use global step multiplier (calculated from all series + fork_step)
        # Group by step to log all metrics at each step together
        for step_value, group in metrics_data.groupby("step"):
            if pd.notna(step_value):
                step = self._convert_step_to_int(step_value, step_multiplier)

                metrics = {}
                for _, row in group.iterrows():
                    if pd.notna(row["float_value"]):
                        cleaned = self._clean_attribute_path(row["attribute_path"])
                        attr_name = self._sanitize_attribute_name(cleaned)
                        metrics[attr_name] = row["float_value"]

                if metrics:
                    self._active_run.log(metrics, step=step)

        self._logger.info(f"Uploaded metrics for run {run_id}")

    def upload_artifacts(
        self,
        run_data: pd.DataFrame,
        run_id: TargetRunId,
        files_base_path: Path,
        step_multiplier: int,
    ) -> None:
        """Upload files and series as artifacts to W&B run.

        Args:
            step_multiplier: Global step multiplier for the run (calculated from all series + fork_step)
        """
        if self._active_run is None:
            raise RuntimeError("No active run")

        run_name = self._current_run_name or run_id

        # Handle regular files — separate checkpoints and ONNX from other files
        file_data = run_data[run_data["attribute_type"].isin(["file", "file_set", "artifact"])]

        other_rows: list[tuple[str, Path, pd.Series]] = []

        for _, row in file_data.iterrows():
            if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                file_path = files_base_path / row["file_value"]["path"]
                cleaned_path = self._clean_attribute_path(row["attribute_path"])
                if _is_checkpoint_path(cleaned_path):
                    self._pending_checkpoints.append((cleaned_path, file_path))
                elif _is_onnx_path(cleaned_path):
                    self._pending_onnx.append((cleaned_path, file_path))
                else:
                    other_rows.append((cleaned_path, file_path, row))

        # Handle remaining files — log images/HTML as native W&B media, others as artifacts
        for cleaned_path, file_path, row in other_rows:
            if file_path.exists():
                if file_path.is_file() and _is_image(file_path):
                    attr_name = self._sanitize_attribute_name(cleaned_path)
                    self._active_run.log({attr_name: wandb.Image(str(file_path))})
                elif file_path.is_file() and _is_html(file_path):
                    attr_name = self._sanitize_attribute_name(cleaned_path)
                    self._active_run.log({attr_name: wandb.Html(str(file_path))})
                else:
                    artifact_name = self._make_artifact_name(cleaned_path, run_name)
                    artifact = wandb.Artifact(name=artifact_name, type=row["attribute_type"])
                    if file_path.is_file():
                        artifact.add_file(str(file_path))
                    else:
                        artifact.add_dir(str(file_path))
                    self._active_run.log_artifact(artifact)
            else:
                self._logger.warning(f"File not found: {file_path}")

        # Handle file series — log images/HTML as native W&B media, others as artifacts.
        # Media uses define_metric with a custom step axis per series so it doesn't
        # conflict with the global step used by float_series metrics.
        file_series_data = run_data[run_data["attribute_type"] == "file_series"].copy()
        file_series_data["attribute_path"] = file_series_data["attribute_path"].apply(self._clean_attribute_path)
        for attr_path, group in file_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            # Detect if this series contains media files (check first valid file)
            is_media_series = False
            for _, probe_row in group.iterrows():
                if pd.notna(probe_row["file_value"]) and isinstance(probe_row["file_value"], dict):
                    probe_path = files_base_path / probe_row["file_value"]["path"]
                    if probe_path.exists() and probe_path.is_file():
                        is_media_series = _is_image(probe_path) or _is_html(probe_path)
                        break

            # Define a custom step axis for media series to avoid conflicts with metrics
            step_key = f"{attr_name}_step"
            if is_media_series:
                self._active_run.define_metric(attr_name, step_metric=step_key)

            for _, row in group.iterrows():
                if pd.notna(row["file_value"]) and isinstance(row["file_value"], dict):
                    file_path = files_base_path / row["file_value"]["path"]
                    if file_path.exists():
                        step = self._convert_step_to_int(row["step"], step_multiplier) if pd.notna(row["step"]) else 0

                        if file_path.is_file() and _is_image(file_path):
                            self._active_run.log({step_key: step, attr_name: wandb.Image(str(file_path))})
                        elif file_path.is_file() and _is_html(file_path):
                            self._active_run.log({step_key: step, attr_name: wandb.Html(str(file_path))})
                        else:
                            # Fall back to artifact for non-media files
                            artifact_name = self._make_artifact_name(attr_path, run_name, suffix=f"step_{step}")
                            artifact = wandb.Artifact(name=artifact_name, type="file_series")
                            if file_path.is_file():
                                artifact.add_file(str(file_path))
                            else:
                                artifact.add_dir(str(file_path))
                            self._active_run.log_artifact(artifact)
                    else:
                        self._logger.warning(f"File not found: {file_path}")

        # Handle string series as text artifacts
        string_series_data = run_data[run_data["attribute_type"] == "string_series"].copy()
        string_series_data["attribute_path"] = string_series_data["attribute_path"].apply(self._clean_attribute_path)
        for attr_path, group in string_series_data.groupby("attribute_path"):
            artifact_name = self._make_artifact_name(attr_path, run_name)

            # Create temporary file with text content
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as tmp_file:
                for _, row in group.iterrows():
                    if pd.notna(row["string_value"]):
                        series_step = (
                            self._convert_step_to_int(row["step"], step_multiplier) if pd.notna(row["step"]) else None
                        )
                        timestamp = row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None
                        text_line = f"{series_step}; {timestamp}; {row['string_value']}\n"
                        tmp_file.write(text_line)
                tmp_file_path = tmp_file.name

                # Create and log W&B artifact
                artifact = wandb.Artifact(name=artifact_name, type="string_series")
                artifact.add_file(tmp_file_path, name="series.txt")
                self._active_run.log_artifact(artifact)

        # Handle histogram series as W&B Histograms
        histogram_series_data = run_data[run_data["attribute_type"] == "histogram_series"].copy()
        histogram_series_data["attribute_path"] = histogram_series_data["attribute_path"].apply(
            self._clean_attribute_path
        )
        for attr_path, group in histogram_series_data.groupby("attribute_path"):
            attr_name = self._sanitize_attribute_name(attr_path)

            for _, row in group.iterrows():
                if pd.notna(row["histogram_value"]) and isinstance(row["histogram_value"], dict):
                    step = self._convert_step_to_int(row["step"], step_multiplier) if pd.notna(row["step"]) else 0
                    hist = row["histogram_value"]

                    # Convert Neptune histogram to W&B Histogram
                    # Neptune format: {"type": str, "edges": list, "values": list}
                    # W&B expects histogram data as np_histogram tuple or sequence
                    try:
                        wandb_hist = wandb.Histogram(np_histogram=(hist.get("values", []), hist.get("edges", [])))
                        self._active_run.log({attr_name: wandb_hist}, step=step)
                    except Exception:
                        self._logger.error(
                            f"Failed to log histogram for {attr_path} at step {step}",
                            exc_info=True,
                        )

        self._logger.info(f"Uploaded artifacts for run {run_id}")
