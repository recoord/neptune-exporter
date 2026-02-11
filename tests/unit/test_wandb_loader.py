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

import pandas as pd
import pyarrow as pa
from decimal import Decimal
from unittest.mock import Mock, patch
from pathlib import Path
import wandb

from neptune_exporter.loaders.wandb_loader import WandBLoader


@patch("wandb.login", spec=wandb.login)
def test_init(mock_login):
    """Test WandBLoader initialization."""
    loader = WandBLoader(
        entity="test-entity",
        api_key="test-key",
        name_prefix="test-prefix",
    )

    assert loader.entity == "test-entity"
    assert loader.name_prefix == "test-prefix"
    mock_login.assert_called_once_with(key="test-key")


def test_init_with_api_key():
    """Test WandBLoader initialization with API key authentication."""
    with patch("wandb.login", spec=wandb.login) as mock_login:
        loader = WandBLoader(entity="test-entity", api_key="test-api-key")

        mock_login.assert_called_once_with(key="test-api-key")
        assert loader.entity == "test-entity"


def test_sanitize_attribute_name():
    """Test attribute name sanitization for W&B."""
    loader = WandBLoader(entity="test-entity")

    # Test normal name
    assert loader._sanitize_attribute_name("normal_name") == "normal_name"

    # Test name with invalid characters (W&B preserves "/" for hierarchical grouping)
    assert loader._sanitize_attribute_name("invalid@name#with$chars/slashes") == "invalid_name_with_chars/slashes"

    # Test name starting with number (must start with letter or underscore)
    assert loader._sanitize_attribute_name("123_metric").startswith("_")

    # Test empty name
    assert loader._sanitize_attribute_name("") == "_attribute"


def test_get_project_name():
    """Test W&B project name generation."""
    loader = WandBLoader(entity="test-entity", name_prefix="test-prefix")
    loader_no_prefix = WandBLoader(entity="test-entity")

    # Test with prefix (org prefix is stripped)
    assert loader._get_project_name("my-org/my-project") == "test-prefix_my-project"

    # Test without prefix (org prefix is stripped)
    assert loader_no_prefix._get_project_name("my-org/my-project") == "my-project"


def test_convert_step_to_int():
    """Test step conversion from decimal to int."""
    loader = WandBLoader(entity="test-entity")

    # Test normal conversion
    assert loader._convert_step_to_int(Decimal("1.5"), 1000) == 1500

    # Test None step
    assert loader._convert_step_to_int(None, 1000) == 0

    # Test zero step
    assert loader._convert_step_to_int(Decimal("0"), 1000) == 0


def test_create_experiment():
    """Test creating a W&B project (experiment)."""
    loader = WandBLoader(entity="test-entity")

    project_name = loader.create_experiment("test-project", "experiment-name")

    assert project_name == "experiment-name"


@patch("wandb.init", spec=wandb.init)
def test_create_run(mock_init):
    """Test creating a W&B run."""
    mock_run = Mock()
    mock_run.id = "wandb-run-123"
    mock_init.return_value = mock_run

    loader = WandBLoader(entity="test-entity")
    run_id = loader.create_run("test-project", "run-name", "experiment-id")

    assert run_id == "wandb-run-123"
    mock_init.assert_called_once_with(
        entity="test-entity",
        project="test-project",
        group="experiment-id",
        name="run-name",
    )


@patch("wandb.init", spec=wandb.init)
def test_create_run_with_parent(mock_init):
    """Test creating a forked W&B run."""
    mock_run = Mock()
    mock_run.id = "wandb-run-child"
    mock_init.return_value = mock_run

    loader = WandBLoader(entity="test-entity")

    # Create child with parent
    run_id = loader.create_run("test-project", "child-run", "experiment-id", "wandb-run-parent")

    assert run_id == "wandb-run-child"

    # Check fork_from parameter
    call_kwargs = mock_init.call_args[1]
    assert "fork_from" in call_kwargs
    assert call_kwargs["fork_from"] == "wandb-run-parent?_step=0"


def test_upload_parameters():
    """Test parameter upload to W&B."""
    loader = WandBLoader(entity="test-entity")

    # Create mock active run
    mock_run = Mock()
    mock_config = Mock()
    mock_run.config = mock_config
    loader._active_run = mock_run

    # Create test data
    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/param1", "test/param2", "test/param3"],
            "attribute_type": ["string", "float", "int"],
            "string_value": ["test_value", None, None],
            "float_value": [None, 3.14, None],
            "int_value": [None, None, 42],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
        }
    )

    loader.upload_parameters(test_data, "RUN-123")

    # Verify config.update was called
    mock_config.update.assert_called_once()
    config_dict = mock_config.update.call_args[0][0]

    assert "test/param1" in config_dict
    assert "test/param2" in config_dict
    assert "test/param3" in config_dict
    assert config_dict["test/param1"] == "test_value"
    assert config_dict["test/param2"] == 3.14
    assert config_dict["test/param3"] == 42


def test_upload_parameters_config_namespaced():
    """Test that config/ params are namespaced under 'config' key in W&B config."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    mock_run.config = {}
    mock_run.summary = {}
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": [
                "config/model/learning_rate",
                "config/model/batch_size",
                "config/data/augmentation",
            ],
            "attribute_type": ["float", "int", "string"],
            "string_value": [None, None, "flip"],
            "float_value": [0.001, None, None],
            "int_value": [None, 32, None],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
        }
    )

    loader.upload_parameters(test_data, "RUN-123")

    # Config should be namespaced under "config" key
    assert "config" in mock_run.config
    config = mock_run.config["config"]
    assert config["model"]["learning_rate"] == 0.001
    assert config["model"]["batch_size"] == 32
    assert config["data"]["augmentation"] == "flip"


def test_upload_parameters_string_set():
    """Test parameter upload with string_set type."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    mock_config = Mock()
    mock_run.config = mock_config
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/string_set"],
            "attribute_type": ["string_set"],
            "string_value": [None],
            "float_value": [None],
            "int_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [["value1", "value2", "value3"]],
        }
    )

    loader.upload_parameters(test_data, "RUN-123")

    mock_config.update.assert_called_once()
    config_dict = mock_config.update.call_args[0][0]

    assert "test/string_set" in config_dict
    assert config_dict["test/string_set"] == ["value1", "value2", "value3"]


def test_upload_metrics():
    """Test metrics upload to W&B."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/metric1", "test/metric1", "test/metric2"],
            "attribute_type": ["float_series", "float_series", "float_series"],
            "step": [Decimal("1.0"), Decimal("2.0"), Decimal("1.0")],
            "timestamp": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-01"),
            ],
            "float_value": [0.5, 0.7, 0.3],
        }
    )

    loader.upload_metrics(test_data, "RUN-123", step_multiplier=1)

    # Verify log was called twice (once for each step)
    assert mock_run.log.call_count == 2

    # Check the calls
    calls = mock_run.log.call_args_list

    # Both calls should have step parameter
    for call in calls:
        assert "step" in call[1]


def test_upload_artifacts_files():
    """Test file artifact upload."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file1", "test/file2"],
            "attribute_type": ["file", "file"],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(test_data, "RUN-123", files_base_path, step_multiplier=1)

        # Verify artifacts were created and logged
        assert mock_artifact_class.call_count == 2
        assert mock_run.log_artifact.call_count == 2


def test_upload_artifacts_file_series():
    """Test file series artifact upload."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file_series", "test/file_series"],
            "attribute_type": ["file_series", "file_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "file_value": [{"path": "file1.txt"}, {"path": "file2.txt"}],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(test_data, "RUN-123", files_base_path, step_multiplier=1)

        # Verify artifacts include step in name
        assert mock_artifact_class.call_count == 2
        calls = mock_artifact_class.call_args_list

        # Check that step is included in artifact names
        for call in calls:
            artifact_name = call[1]["name"]
            assert "step_" in artifact_name


def test_upload_artifacts_string_series():
    """Test string series artifact upload as text artifact."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run
    loader._current_run_name = "RUN-123"

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/string_series", "test/string_series"],
            "attribute_type": ["string_series", "string_series"],
            "step": [Decimal("1.0"), Decimal("2.0")],
            "timestamp": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "string_value": ["value1", "value2"],
        }
    )

    with (
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
        patch("tempfile.NamedTemporaryFile") as mock_temp_file,
    ):
        # Mock temporary file
        mock_file = Mock()
        mock_file.name = "/tmp/test_series.txt"
        mock_file.write = Mock()
        mock_file.flush = Mock()
        mock_temp_file.return_value.__enter__.return_value = mock_file

        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(test_data, "RUN-123", files_base_path, step_multiplier=1)

        # Verify artifact was created and logged (name is scoped per-run)
        mock_artifact_class.assert_called_once_with(name="test__string_series-RUN-123", type="string_series")
        mock_artifact.add_file.assert_called_once()
        mock_run.log_artifact.assert_called_once_with(mock_artifact)

        # Verify text content was written
        assert mock_file.write.call_count >= 1
        # Get all written text (in case write is called multiple times)
        written_calls = mock_file.write.call_args_list
        all_written_text = "".join(call[0][0] for call in written_calls)
        assert "1; 2023-01-01T00:00:00; value1" in all_written_text
        assert "2; 2023-01-02T00:00:00; value2" in all_written_text


def test_upload_artifacts_histogram_series():
    """Test histogram series artifact upload as W&B Histogram."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/hist_series"],
            "attribute_type": ["histogram_series"],
            "step": [Decimal("1.0")],
            "timestamp": [pd.Timestamp("2023-01-01")],
            "histogram_value": [{"type": "histogram", "edges": [0.0, 1.0, 2.0], "values": [10, 20]}],
        }
    )

    with patch("wandb.Histogram", spec=wandb.Histogram) as mock_histogram_class:
        mock_histogram = Mock()
        mock_histogram_class.return_value = mock_histogram

        files_base_path = Path("/test/files")
        loader.upload_artifacts(test_data, "RUN-123", files_base_path, step_multiplier=1)

        # Verify Histogram was created and logged
        mock_histogram_class.assert_called_once()
        mock_run.log.assert_called_once()

        # Check histogram creation
        call_kwargs = mock_histogram_class.call_args[1]
        assert "np_histogram" in call_kwargs
        values, edges = call_kwargs["np_histogram"]
        assert values == [10, 20]
        assert edges == [0.0, 1.0, 2.0]


def test_upload_artifacts_file_set():
    """Test file_set artifact upload (directory)."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/file_set1", "test/file_set2"],
            "attribute_type": ["file_set", "file_set"],
            "file_value": [
                {"path": "file_set1_dir"},
                {"path": "file_set2_dir"},
            ],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=False),
        patch("pathlib.Path.is_dir", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(test_data, "RUN-123", files_base_path, step_multiplier=1)

        # Verify artifacts were created and logged
        assert mock_artifact_class.call_count == 2
        assert mock_run.log_artifact.call_count == 2

        # Verify artifact types are set correctly
        calls = mock_artifact_class.call_args_list
        assert calls[0][1]["type"] == "file_set"
        assert calls[1][1]["type"] == "file_set"

        # Verify add_dir was called (not add_file) for directories
        assert mock_artifact.add_dir.call_count == 2
        assert mock_artifact.add_file.call_count == 0


def test_upload_artifacts_artifact_type():
    """Test artifact type upload (JSON file containing artifact metadata)."""
    loader = WandBLoader(entity="test-entity")

    mock_run = Mock()
    loader._active_run = mock_run

    test_data = pd.DataFrame(
        {
            "attribute_path": ["test/artifact1", "test/artifact2"],
            "attribute_type": ["artifact", "artifact"],
            "file_value": [
                {"path": "project/run/test/artifact1/files_list.json"},
                {"path": "project/run/test/artifact2/files_list.json"},
            ],
        }
    )

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_artifact = Mock()
        mock_artifact_class.return_value = mock_artifact

        files_base_path = Path("/test/files")
        loader.upload_artifacts(test_data, "RUN-123", files_base_path, step_multiplier=1)

        # Verify artifacts were created and logged
        assert mock_artifact_class.call_count == 2
        assert mock_run.log_artifact.call_count == 2

        # Verify artifact types are set correctly
        calls = mock_artifact_class.call_args_list
        assert calls[0][1]["type"] == "artifact"
        assert calls[1][1]["type"] == "artifact"

        # Verify add_file was called (not add_dir) for files
        assert mock_artifact.add_file.call_count == 2
        assert mock_artifact.add_dir.call_count == 0

        # Verify file paths
        file_paths = [call[0][0] for call in mock_artifact.add_file.call_args_list]
        assert "/test/files/project/run/test/artifact1/files_list.json" in file_paths
        assert "/test/files/project/run/test/artifact2/files_list.json" in file_paths


def test_upload_run_data():
    """Test uploading complete run data."""
    loader = WandBLoader(entity="test-entity")

    # Create test data with all required schema columns
    test_data = pd.DataFrame(
        {
            "project_id": ["test-project"] * 3,
            "run_id": ["RUN-123"] * 3,
            "attribute_path": ["test/param", "test/metric", "test/file"],
            "attribute_type": ["string", "float_series", "file"],
            "step": [None, Decimal("1.0"), None],
            "timestamp": [None, pd.Timestamp("2023-01-01"), None],
            "int_value": [None, None, None],
            "float_value": [None, 0.5, None],
            "string_value": ["test_value", None, None],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
            "file_value": [None, None, {"path": "file.txt"}],
            "histogram_value": [None, None, None],
        }
    )

    with (
        patch("wandb.init", spec=wandb.init) as mock_init,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.config = Mock()
        mock_init.return_value = mock_run
        mock_artifact_class.return_value = Mock()

        # Create run first
        loader.create_run("test-project", "test-run", "test-experiment")

        # Convert to PyArrow table with proper schema
        from neptune_exporter import model

        table = pa.Table.from_pandas(test_data, schema=model.SCHEMA)

        # upload_run_data now expects a generator of tables
        def table_generator():
            yield table

        # Upload run data with step_multiplier
        loader.upload_run_data(table_generator(), "test-run-id", Path("/test/files"), step_multiplier=100)

        # Verify methods were called
        mock_run.config.update.assert_called_once()  # Parameters
        mock_run.log.assert_called()  # Metrics
        mock_run.log_artifact.assert_called_once()  # Files
        mock_run.finish.assert_called_once()  # Run finished


def test_parse_checkpoint_stem():
    """Test parsing epoch and step from checkpoint filename stems."""
    from neptune_exporter.loaders.wandb_loader import _parse_checkpoint_stem

    assert _parse_checkpoint_stem("epoch=0035") == (35, None)
    assert _parse_checkpoint_stem("epoch=16-step=34000") == (16, 34000)
    assert _parse_checkpoint_stem("model_step=002756") == (None, 2756)
    assert _parse_checkpoint_stem("epoch_00010-step_00001000") == (10, 1000)
    assert _parse_checkpoint_stem("last") == (None, None)


def test_to_sunstone_checkpoint_name():
    """Test conversion from Neptune checkpoint names to Sunstone format."""
    from neptune_exporter.loaders.wandb_loader import _to_sunstone_checkpoint_name

    assert _to_sunstone_checkpoint_name("epoch=0035") == "epoch_00035-step_00000000"
    assert _to_sunstone_checkpoint_name("epoch=0000") == "epoch_00000-step_00000000"
    assert _to_sunstone_checkpoint_name("epoch=16-step=34000") == "epoch_00016-step_00034000"
    assert _to_sunstone_checkpoint_name("model_step=002756") == "epoch_00000-step_00002756"
    assert _to_sunstone_checkpoint_name("epoch_00010-step_00001000") == "epoch_00010-step_00001000"
    assert _to_sunstone_checkpoint_name("last") == "last"
    assert _to_sunstone_checkpoint_name("some_random_name") == "some_random_name"


def test_ckpt_sort_key():
    """Test checkpoint sort key ordering."""
    from neptune_exporter.loaders.wandb_loader import _ckpt_sort_key

    items = [
        ("model/checkpoints/epoch=0035", Path("/fake")),
        ("model/checkpoints/epoch=0005", Path("/fake")),
        ("model/checkpoints/epoch=0010", Path("/fake")),
        ("model/checkpoints/last", Path("/fake")),
    ]
    sorted_items = sorted(items, key=_ckpt_sort_key)
    assert [Path(i[0]).name for i in sorted_items] == [
        "epoch=0005",
        "epoch=0010",
        "epoch=0035",
        "last",
    ]

    # Step-only sorting
    step_items = [
        ("model/checkpoints/model_step=002756", Path("/fake")),
        ("model/checkpoints/model_step=000100", Path("/fake")),
    ]
    sorted_step = sorted(step_items, key=_ckpt_sort_key)
    assert [Path(i[0]).name for i in sorted_step] == [
        "model_step=000100",
        "model_step=002756",
    ]


def test_upload_best_model_metadata():
    """Test best model metadata extraction and summary writing."""
    loader = WandBLoader(entity="test-entity")
    mock_run = Mock()
    mock_run.summary = {}
    loader._active_run = mock_run

    # Test with all 3 keys
    param_data = pd.DataFrame(
        {
            "attribute_path": [
                "model/best_model_path",
                "model/best_model_score",
                "config/model/best_model_monitor",
            ],
            "attribute_type": ["string", "float", "string"],
            "string_value": ["/checkpoints/epoch=0035.ckpt", None, "val/loss"],
            "float_value": [None, 0.123, None],
            "int_value": [None, None, None],
            "bool_value": [None, None, None],
            "datetime_value": [None, None, None],
            "string_set_value": [None, None, None],
        }
    )

    handled = loader._upload_best_model_metadata(param_data)
    assert len(handled) == 3
    assert mock_run.summary["metadata/checkpoint/best_model_name_val_loss"] == "epoch_00035-step_00000000"
    assert mock_run.summary["metadata/checkpoint/best_model_score_val_loss"] == 0.123

    # Test with no monitor (defaults to "unknown")
    mock_run.summary = {}
    param_data_no_monitor = pd.DataFrame(
        {
            "attribute_path": ["model/best_model_path", "model/best_model_score"],
            "attribute_type": ["string", "float"],
            "string_value": ["/checkpoints/model_step=002756.ckpt", None],
            "float_value": [None, 0.456],
            "int_value": [None, None],
            "bool_value": [None, None],
            "datetime_value": [None, None],
            "string_set_value": [None, None],
        }
    )

    handled = loader._upload_best_model_metadata(param_data_no_monitor)
    assert mock_run.summary["metadata/checkpoint/best_model_name_unknown"] == "epoch_00000-step_00002756"

    # Test with no best_model_path (no-op)
    mock_run.summary = {}
    param_data_empty = pd.DataFrame(
        {
            "attribute_path": ["some/other/param"],
            "attribute_type": ["string"],
            "string_value": ["hello"],
            "float_value": [None],
            "int_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
        }
    )

    handled = loader._upload_best_model_metadata(param_data_empty)
    assert len(mock_run.summary) == 0


def test_cross_chunk_checkpoint_accumulation():
    """Test that checkpoints from multiple parquet chunks are sorted globally before upload."""
    from neptune_exporter import model

    loader = WandBLoader(entity="test-entity")

    # Chunk 1: epoch=10 checkpoint
    chunk1_df = pd.DataFrame(
        {
            "project_id": ["test-project"],
            "run_id": ["RUN-123"],
            "attribute_path": ["model/checkpoints/epoch=0010"],
            "attribute_type": ["file"],
            "step": [None],
            "timestamp": [None],
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [{"path": "ckpt/epoch=0010.ckpt"}],
            "histogram_value": [None],
        }
    )

    # Chunk 2: epoch=5 checkpoint (earlier epoch, but in later chunk)
    chunk2_df = pd.DataFrame(
        {
            "project_id": ["test-project"],
            "run_id": ["RUN-123"],
            "attribute_path": ["model/checkpoints/epoch=0005"],
            "attribute_type": ["file"],
            "step": [None],
            "timestamp": [None],
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [{"path": "ckpt/epoch=0005.ckpt"}],
            "histogram_value": [None],
        }
    )

    chunk1 = pa.Table.from_pandas(chunk1_df, schema=model.SCHEMA)
    chunk2 = pa.Table.from_pandas(chunk2_df, schema=model.SCHEMA)

    def two_chunk_generator():
        yield chunk1
        yield chunk2

    with (
        patch("wandb.init", spec=wandb.init) as mock_init,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.config = Mock()
        mock_init.return_value = mock_run
        mock_artifact_class.return_value = Mock()

        # Create run first
        loader.create_run("test-project", "test-run", "test-experiment")

        # Upload with 2-chunk generator
        loader.upload_run_data(two_chunk_generator(), "test-run-id", Path("/test/files"), step_multiplier=1)

        # Get all Artifact() calls
        artifact_calls = mock_artifact_class.call_args_list

        # Should have 2 checkpoint artifacts
        assert len(artifact_calls) == 2

        # First artifact should be epoch=5, second should be epoch=10
        first_metadata = artifact_calls[0][1]["metadata"]
        second_metadata = artifact_calls[1][1]["metadata"]
        assert first_metadata["epoch"] == 5, f"Expected epoch=5 first, got {first_metadata}"
        assert second_metadata["epoch"] == 10, f"Expected epoch=10 second, got {second_metadata}"

        # Verify filenames are in Sunstone format
        assert first_metadata["filename"] == "epoch_00005-step_00000000"
        assert second_metadata["filename"] == "epoch_00010-step_00000000"

        # Verify aliases: epoch-based checkpoints get ["latest", stem] in Sunstone format
        alias_calls = mock_run.log_artifact.call_args_list
        assert alias_calls[0][1]["aliases"] == ["latest", "epoch_00005-step_00000000"]
        assert alias_calls[1][1]["aliases"] == ["latest", "epoch_00010-step_00000000"]


def test_cross_chunk_last_checkpoint_sorts_last():
    """Test that 'last' checkpoint sorts after all epoch-based checkpoints across chunks."""
    from neptune_exporter import model

    loader = WandBLoader(entity="test-entity")

    # Chunk 1: "last" checkpoint (arrives first but should sort last)
    chunk1_df = pd.DataFrame(
        {
            "project_id": ["test-project"],
            "run_id": ["RUN-123"],
            "attribute_path": ["model/checkpoints/last"],
            "attribute_type": ["file"],
            "step": [None],
            "timestamp": [None],
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [{"path": "ckpt/last.ckpt"}],
            "histogram_value": [None],
        }
    )

    # Chunk 2: epoch=5 checkpoint (lower epoch, arrives second)
    chunk2_df = pd.DataFrame(
        {
            "project_id": ["test-project"],
            "run_id": ["RUN-123"],
            "attribute_path": ["model/checkpoints/epoch=0005"],
            "attribute_type": ["file"],
            "step": [None],
            "timestamp": [None],
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [{"path": "ckpt/epoch=0005.ckpt"}],
            "histogram_value": [None],
        }
    )

    chunk1 = pa.Table.from_pandas(chunk1_df, schema=model.SCHEMA)
    chunk2 = pa.Table.from_pandas(chunk2_df, schema=model.SCHEMA)

    def two_chunk_generator():
        yield chunk1
        yield chunk2

    with (
        patch("wandb.init", spec=wandb.init) as mock_init,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
        patch("wandb.Artifact", spec=wandb.Artifact) as mock_artifact_class,
    ):
        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.config = Mock()
        mock_init.return_value = mock_run
        mock_artifact_class.return_value = Mock()

        loader.create_run("test-project", "test-run", "test-experiment")

        loader.upload_run_data(two_chunk_generator(), "test-run-id", Path("/test/files"), step_multiplier=1)

        artifact_calls = mock_artifact_class.call_args_list
        assert len(artifact_calls) == 2

        # epoch=5 should be first (version 1), "last" should be second (version 2)
        first_metadata = artifact_calls[0][1]["metadata"]
        second_metadata = artifact_calls[1][1]["metadata"]
        assert first_metadata["epoch"] == 5, f"Expected epoch=5 first, got {first_metadata}"
        assert second_metadata["is_last"] is True, f"Expected 'last' second, got {second_metadata}"

        # Verify filename is in Sunstone format
        assert first_metadata["filename"] == "epoch_00005-step_00000000"

        # Verify aliases: epoch gets ["latest", stem] in Sunstone format, last gets ["latest", "last"]
        alias_calls = mock_run.log_artifact.call_args_list
        assert alias_calls[0][1]["aliases"] == ["latest", "epoch_00005-step_00000000"]
        assert alias_calls[1][1]["aliases"] == ["latest", "last"]


def test_state_cleanup_after_upload_failure():
    """Test that pending state is cleared when upload_run_data fails mid-upload."""
    from neptune_exporter import model

    loader = WandBLoader(entity="test-entity")

    chunk_df = pd.DataFrame(
        {
            "project_id": ["test-project"],
            "run_id": ["RUN-123"],
            "attribute_path": ["model/checkpoints/epoch=0010"],
            "attribute_type": ["file"],
            "step": [None],
            "timestamp": [None],
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [{"path": "ckpt/epoch=0010.ckpt"}],
            "histogram_value": [None],
        }
    )
    chunk = pa.Table.from_pandas(chunk_df, schema=model.SCHEMA)

    def failing_generator():
        yield chunk
        raise RuntimeError("Simulated chunk read failure")

    with (
        patch("wandb.init", spec=wandb.init) as mock_init,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.config = Mock()
        mock_init.return_value = mock_run

        loader.create_run("test-project", "test-run", "test-experiment")

        import pytest

        with pytest.raises(RuntimeError, match="Simulated chunk read failure"):
            loader.upload_run_data(failing_generator(), "test-run-id", Path("/test/files"), step_multiplier=1)

        # Verify all state is cleaned up
        assert loader._pending_checkpoints == []
        assert loader._pending_onnx == []
        assert loader._pending_tags == set()
        assert loader._active_run is None
        assert loader._current_run_name is None


def test_state_cleanup_when_finish_throws():
    """Test that state is cleaned up even when finish(exit_code=1) raises."""
    from neptune_exporter import model

    loader = WandBLoader(entity="test-entity")

    chunk_df = pd.DataFrame(
        {
            "project_id": ["test-project"],
            "run_id": ["RUN-123"],
            "attribute_path": ["model/checkpoints/epoch=0010"],
            "attribute_type": ["file"],
            "step": [None],
            "timestamp": [None],
            "int_value": [None],
            "float_value": [None],
            "string_value": [None],
            "bool_value": [None],
            "datetime_value": [None],
            "string_set_value": [None],
            "file_value": [{"path": "ckpt/epoch=0010.ckpt"}],
            "histogram_value": [None],
        }
    )
    chunk = pa.Table.from_pandas(chunk_df, schema=model.SCHEMA)

    def failing_generator():
        yield chunk
        raise RuntimeError("Simulated chunk read failure")

    with (
        patch("wandb.init", spec=wandb.init) as mock_init,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        mock_run = Mock()
        mock_run.id = "test-run-id"
        mock_run.config = Mock()
        # Make finish() raise to simulate network error during cleanup
        mock_run.finish.side_effect = ConnectionError("W&B server unreachable")
        mock_init.return_value = mock_run

        loader.create_run("test-project", "test-run", "test-experiment")

        import pytest

        with pytest.raises(RuntimeError, match="Simulated chunk read failure"):
            loader.upload_run_data(failing_generator(), "test-run-id", Path("/test/files"), step_multiplier=1)

        # State must still be cleaned up despite finish() throwing
        assert loader._pending_checkpoints == []
        assert loader._pending_onnx == []
        assert loader._pending_tags == set()
        assert loader._active_run is None
        assert loader._current_run_name is None
