"""
Comprehensive tests for StorageManager module.

This test file covers:
1. StorageManager initialization and configuration
2. _LoadDataManager context manager scenarios
3. File operations (open, read, write)
4. Cloud + local synchronization
5. finalize_artifact workflow
6. Manifest registration and management
7. Error handling and edge cases
8. Critical bug fixes validation
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call, ANY
import pytest

from cclang.core.storage import (
    StorageManager,
    CloudConfig,
    StorageManagerError,
    ErasingUploadCallback,
)
from cclang.io.fs import LocalConfig, FileManager
from cclang.config.s3 import S3Config
from cclang.io.cloud import S3Store, S3Mapping
from cclang.io.schemas import UploadManifestRecord
from cclang.io.exceptions import LocalStorageError


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_base(tmp_path):
    """Create temporary base directory for tests"""
    base = tmp_path / "test_data"
    base.mkdir()
    return base


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for temp files"""
    temp = tmp_path / "temp"
    temp.mkdir()
    return temp


@pytest.fixture
def local_config(temp_base, temp_dir):
    """Create LocalConfig for testing"""
    return LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )


@pytest.fixture
def local_config_no_save(temp_base, temp_dir):
    """LocalConfig with save_local=False"""
    return LocalConfig(
        base_path=temp_base,
        save_local=False,
        cache_files=False,
        temp_base=temp_dir,
    )


@pytest.fixture
def s3_config():
    """Create S3Config for testing"""
    return S3Config(
        enable=True,
        bucket="test-bucket",
        access_key="test-key",
        secret_key="test-secret",
        region="ru-central1",
        root_prefix=Path("prefix"),
    )


@pytest.fixture
def cloud_config(s3_config):
    """Create CloudConfig with S3 enabled"""
    return CloudConfig(
        enable=True,
        s3_config=s3_config,
        max_upload_threads=2,
        max_download_threads=0,
        cloud_max_attempts=3,
        cloud_base_backoff=0.1,
    )


@pytest.fixture
def cloud_config_disabled(s3_config):
    """Create CloudConfig with S3 disabled"""
    s3_config.enable = False
    return CloudConfig(
        enable=False,
        s3_config=s3_config,
    )


@pytest.fixture
def storage_manager(local_config, cloud_config_disabled):
    """Create StorageManager with cloud disabled for basic tests"""
    sm = StorageManager(local_config, cloud_config_disabled)
    yield sm
    sm.close()


@pytest.fixture
def storage_manager_with_cloud(local_config, cloud_config):
    """Create StorageManager with mocked cloud"""
    with patch("cclang.core.storage.S3Store") as mock_s3:
        mock_s3_instance = MagicMock(spec=S3Store)
        mock_s3.return_value = mock_s3_instance

        sm = StorageManager(local_config, cloud_config)
        sm._cloud = mock_s3_instance

        yield sm, mock_s3_instance
        sm.close()


# ============================================================
# Tests: Initialization
# ============================================================

def test_initialization_local_only(local_config, cloud_config_disabled):
    """Test StorageManager initialization with local storage only"""
    sm = StorageManager(local_config, cloud_config_disabled)

    assert sm.local_cfg == local_config
    assert sm.cloud_cfg == cloud_config_disabled
    assert sm._file_manager is not None
    assert sm._cloud is None  # Cloud should be None when disabled
    assert len(sm._manifests) == 1  # Only transfer_manifest
    assert '_transfer_manifest' in sm._manifests

    sm.close()


def test_initialization_with_cloud(local_config, cloud_config):
    """Test StorageManager initialization with cloud enabled"""
    with patch("cclang.core.storage.S3Store") as mock_s3:
        mock_s3_instance = MagicMock(spec=S3Store)
        mock_s3.return_value = mock_s3_instance

        sm = StorageManager(local_config, cloud_config)

        assert sm._cloud is not None
        assert sm._cloud == mock_s3_instance
        mock_s3.assert_called_once()

        sm.close()


def test_save_local_enabled(storage_manager):
    """Test save_local_enabled() method"""
    assert storage_manager.save_local_enabled() is True

    storage_manager.local_cfg.save_local = False
    assert storage_manager.save_local_enabled() is False


def test_save_cloud_enabled(local_config, cloud_config, cloud_config_disabled):
    """Test save_cloud_enabled() method"""
    with patch("cclang.core.storage.S3Store") as mock_s3:
        mock_s3_instance = MagicMock(spec=S3Store)
        mock_s3.return_value = mock_s3_instance

        sm_enabled = StorageManager(local_config, cloud_config)
        assert sm_enabled.save_cloud_enabled() is True
        sm_enabled.close()

    sm_disabled = StorageManager(local_config, cloud_config_disabled)
    assert sm_disabled.save_cloud_enabled() is False
    sm_disabled.close()


# ============================================================
# Tests: BUG FIX #1 - _cloud initialization
# ============================================================

def test_bugfix_cloud_initialization_type(local_config, cloud_config):
    """
    CRITICAL BUG FIX TEST
    Bug was: self._cloud = Optional[S3Store] (type hint instead of assignment)
    Should be: self._cloud: Optional[S3Store] = None
    """
    with patch("cclang.core.storage.S3Store") as mock_s3:
        mock_s3_instance = MagicMock(spec=S3Store)
        mock_s3.return_value = mock_s3_instance

        sm = StorageManager(local_config, cloud_config)

        # _cloud should be S3Store instance, not a type
        assert sm._cloud is not None
        assert isinstance(sm._cloud, MagicMock)  # Mock of S3Store
        assert sm._cloud == mock_s3_instance

        sm.close()


# ============================================================
# Tests: BUG FIX #2 - _exists_local return value
# ============================================================

def test_bugfix_exists_local_returns_value(storage_manager, temp_base):
    """
    CRITICAL BUG FIX TEST
    Bug was: _exists_local() didn't return the result
    Should return: bool value from self._file_manager.exists(path)
    """
    # Create a test file
    test_file = temp_base / "test.txt"
    test_file.write_text("test content")

    # Test existing file
    result = storage_manager._exists_local(Path("test.txt"))
    assert result is True
    assert isinstance(result, bool)

    # Test non-existing file
    result = storage_manager._exists_local(Path("nonexistent.txt"))
    assert result is False
    assert isinstance(result, bool)


def test_exists_method_uses_exists_local(storage_manager, temp_base):
    """Test that exists() properly uses _exists_local()"""
    test_file = temp_base / "exists_test.txt"
    test_file.write_text("content")

    # Should return True for existing file
    assert storage_manager.exists(Path("exists_test.txt")) is True

    # Should return False for non-existing file
    assert storage_manager.exists(Path("does_not_exist.txt")) is False


# ============================================================
# Tests: BUG FIX #3 - Write mode save condition
# ============================================================

def test_bugfix_write_mode_saves_on_success(temp_base, temp_dir):
    """
    CRITICAL BUG FIX TEST
    Bug was: if exc_type is not None and self._is_write_mode:
    Should be: if exc_type is None and self._is_write_mode:

    This test validates the condition is correct by directly testing the __exit__ logic.
    """
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # Create test file first (so it's not in temp)
        test_file = temp_base / "test_write.txt"
        test_file.write_text("existing")

        # Write to file successfully (no exception)
        test_path = Path("test_write.txt")
        with sm.open(test_path, mode="w") as f:
            f.write("test data")
        # No exception - context manager exits normally (exc_type is None)

        # Cloud upload should be called because:
        # 1. exc_type is None (success)
        # 2. _is_write_mode is True
        # 3. save_cloud_enabled() is True
        sm._cloud.upload.assert_called()

        sm.close()


def test_bugfix_write_mode_not_saves_on_error(temp_base, temp_dir):
    """
    Test that file is NOT saved to cloud when write fails (exc_type is not None)
    """
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,  # Keep local to avoid cleanup issues
        cache_files=False,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # Write with exception
        test_path = Path("test_error.txt")
        try:
            with sm.open(test_path, mode="w") as f:
                f.write("test data")
                raise ValueError("Simulated error during write")
        except ValueError:
            pass  # Expected

        # Cloud upload should NOT be called because exc_type is not None
        sm._cloud.upload.assert_not_called()

        sm.close()


# ============================================================
# Tests: _LoadDataManager - Read Mode
# ============================================================

def test_load_data_manager_read_existing_file(storage_manager, temp_base):
    """Test reading existing local file"""
    test_file = temp_base / "read_test.txt"
    test_content = "Hello, World!"
    test_file.write_text(test_content)

    with storage_manager.open(Path("read_test.txt"), mode="r") as f:
        content = f.read()

    assert content == test_content


def test_load_data_manager_read_nonexistent_raises(storage_manager):
    """Test that reading non-existent file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError, match="File not found neither localy nor in cloud"):
        with storage_manager.open(Path("nonexistent.txt"), mode="r") as f:
            pass


def test_load_data_manager_read_absolute_path(storage_manager, temp_base):
    """Test reading file with absolute path"""
    test_file = temp_base / "absolute_test.txt"
    test_file.write_text("absolute path content")

    with storage_manager.open(test_file, mode="r") as f:
        content = f.read()

    assert content == "absolute path content"


# ============================================================
# Tests: _LoadDataManager - Write Mode
# ============================================================

def test_load_data_manager_write_new_file(storage_manager, temp_base):
    """Test creating and writing new file"""
    test_path = Path("new_file.txt")
    test_content = "New file content"

    with storage_manager.open(test_path, mode="w") as f:
        f.write(test_content)

    # Verify file was created
    assert (temp_base / "new_file.txt").exists()
    assert (temp_base / "new_file.txt").read_text() == test_content


def test_load_data_manager_write_requires_save_location(local_config_no_save, cloud_config_disabled):
    """Test that write mode requires either save_local or cloud enabled"""
    sm = StorageManager(local_config_no_save, cloud_config_disabled)

    with pytest.raises(StorageManagerError, match="Writting is not save"):
        with sm.open(Path("test.txt"), mode="w") as f:
            f.write("content")

    sm.close()


def test_load_data_manager_append_mode(storage_manager, temp_base):
    """Test append mode creates parent directories"""
    test_path = Path("subdir/append_test.txt")

    with storage_manager.open(test_path, mode="w") as f:
        f.write("first line\n")

    with storage_manager.open(test_path, mode="a") as f:
        f.write("appended line\n")

    content = (temp_base / "subdir" / "append_test.txt").read_text()
    assert "first line" in content
    assert "appended line" in content


# ============================================================
# Tests: _LoadDataManager - Cloud Download
# ============================================================

def test_load_data_manager_downloads_from_cloud(temp_base, temp_dir):
    """Test downloading file from cloud when not in local storage"""
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        # Setup: file exists in cloud but not locally
        mock_s3.exists.return_value = True

        def mock_download(cloud_key, local_path, blocking=True):
            # Simulate download by creating the file
            local_path.write_text("cloud content")

        mock_s3.download.side_effect = mock_download

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # Open file - should trigger download
        test_path = Path("cloud_file.txt")
        with sm.open(test_path, mode="r") as f:
            content = f.read()

        assert content == "cloud content"
        mock_s3.download.assert_called_once()

        sm.close()


def test_load_data_manager_temp_download_when_no_cache(temp_base, temp_dir):
    """Test that cloud file is downloaded to temp when cache_files=False"""
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=False,  # No caching
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        mock_s3.exists.return_value = True

        def mock_download(cloud_key, local_path, blocking=True):
            local_path.write_text("temp cloud content")

        mock_s3.download.side_effect = mock_download

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        test_path = Path("temp_download.txt")
        with sm.open(test_path, mode="r") as f:
            content = f.read()

        assert content == "temp cloud content"

        # File should NOT be in base_path (because cache_files=False)
        assert not (temp_base / "temp_download.txt").exists()

        sm.close()


# ============================================================
# Tests: finalize_artifact
# ============================================================

def test_finalize_artifact_local_only(storage_manager, temp_base, temp_dir):
    """Test finalize_artifact with local storage only"""
    # Create temp file
    temp_file = temp_dir / "temp_artifact.txt"
    temp_file.write_text("artifact content")

    dest_path = Path("artifacts/final.txt")

    storage_manager.finalize_artifact(temp_file, dest_path, blocking=True)

    # Verify file moved to final location
    final_file = temp_base / "artifacts" / "final.txt"
    assert final_file.exists()
    assert final_file.read_text() == "artifact content"

    # Temp file should still exist (moved via copy in FileManager)
    # Actually, finalize_artifact uses atomic_move which deletes source
    # But let's verify the final file exists


def test_finalize_artifact_with_cloud(temp_base, temp_dir):
    """Test finalize_artifact uploads to cloud"""
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # Create temp file
        temp_file = temp_dir / "temp_cloud_artifact.txt"
        temp_file.write_text("cloud artifact")

        dest_path = Path("cloud_artifacts/final.txt")

        sm.finalize_artifact(temp_file, dest_path, blocking=True)

        # Verify cloud upload was called
        mock_s3.upload.assert_called_once()

        # Verify local file exists
        final_file = temp_base / "cloud_artifacts" / "final.txt"
        assert final_file.exists()

        sm.close()


def test_finalize_artifact_cloud_only_erases_local(temp_base, temp_dir):
    """Test that finalize_artifact erases local file when save_local=False"""
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=False,
        cache_files=False,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        temp_file = temp_dir / "erase_me.txt"
        temp_file.write_text("to be erased")

        dest_path = Path("cloud_only/final.txt")

        sm.finalize_artifact(temp_file, dest_path, blocking=True)

        # Upload should be called
        mock_s3.upload.assert_called_once()

        # Local file should NOT exist (erased)
        final_file = temp_base / "cloud_only" / "final.txt"
        assert not final_file.exists()

        # Temp file should be deleted
        assert not temp_file.exists()

        sm.close()


def test_finalize_artifact_temp_not_exists_raises(storage_manager):
    """Test that finalize_artifact raises if temp file doesn't exist"""
    with pytest.raises(FileNotFoundError, match="Temp file .* does not exist"):
        storage_manager.finalize_artifact(
            Path("/nonexistent/temp.txt"),
            Path("dest.txt"),
        )


def test_finalize_artifact_requires_save_location(temp_base, temp_dir):
    """Test finalize_artifact requires either local or cloud save"""
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=False,
        cache_files=False,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=False,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=False, s3_config=s3_cfg)

    sm = StorageManager(local_cfg, cloud_cfg)

    temp_file = temp_dir / "temp.txt"
    temp_file.write_text("content")

    with pytest.raises(StorageManagerError, match="Writting is not save"):
        sm.finalize_artifact(temp_file, Path("dest.txt"))

    sm.close()


# ============================================================
# Tests: Manifest Management
# ============================================================

def test_register_manifest(storage_manager):
    """Test registering a new manifest"""
    from pydantic import BaseModel, Field

    class TestManifest(BaseModel):
        test_id: str
        value: int

        @property
        def id(self):
            return self.test_id

    manifest = storage_manager.register_manifest(
        "test_manifest.jsonl",
        TestManifest,
        flush_every=5
    )

    assert manifest is not None
    assert "test_manifest.jsonl" in storage_manager._manifests
    assert storage_manager._manifests["test_manifest.jsonl"] == manifest


def test_push_manifest_single(temp_base, temp_dir):
    """Test pushing a single manifest to cloud"""
    from pydantic import BaseModel

    class TestRecord(BaseModel):
        record_id: str
        data: str

        @property
        def id(self):
            return self.record_id

    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # Register and add data to manifest
        manifest = sm.register_manifest("push_test.jsonl", TestRecord)
        record = TestRecord(record_id="1", data="test")
        manifest.mark(record)
        manifest.flush()

        # Push manifest
        sm.push_manifest("push_test.jsonl")

        # Verify upload was called
        mock_s3.upload.assert_called()

        sm.close()


def test_push_all_manifests(temp_base, temp_dir):
    """Test pushing all manifests when manifest_name=None"""
    from pydantic import BaseModel

    class TestRecord(BaseModel):
        record_id: str

        @property
        def id(self):
            return self.record_id

    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # Register multiple manifests
        sm.register_manifest("manifest1.jsonl", TestRecord)
        sm.register_manifest("manifest2.jsonl", TestRecord)

        # Push all (including _transfer_manifest)
        sm.push_manifest(manifest_name=None)

        # Should upload 3 manifests (2 registered + transfer_manifest)
        assert mock_s3.upload.call_count == 3

        sm.close()


def test_push_nonexistent_manifest_raises(storage_manager):
    """Test that pushing non-existent manifest raises error"""
    with pytest.raises(StorageManagerError, match="Manifest .* does not exist"):
        storage_manager.push_manifest("nonexistent.jsonl")


# ============================================================
# Tests: Helper Methods
# ============================================================

def test_create_temp_file(storage_manager, temp_dir):
    """Test create_temp_file creates file in temp directory"""
    temp_file = storage_manager.create_temp_file(suffix=".test")

    assert temp_file.exists()
    assert temp_file.parent == temp_dir
    assert str(temp_file).endswith(".test")


def test_push_data(temp_base, temp_dir):
    """Test push_data uploads file to cloud"""
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # Create test file
        test_file = temp_base / "push_test.txt"
        test_file.write_text("push data")

        sm.push_data(Path("push_test.txt"), blocking=True)

        mock_s3.upload.assert_called_once()

        sm.close()


def test_close_cleans_up_cloud(temp_base, temp_dir):
    """Test that close() properly cleans up cloud connection"""
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        sm.close()

        mock_s3.close.assert_called_once()
        assert sm._cloud is None


# ============================================================
# Tests: ErasingUploadCallback
# ============================================================

def test_erasing_upload_callback_success(temp_dir):
    """Test ErasingUploadCallback deletes file on success"""
    test_file = temp_dir / "to_erase.txt"
    test_file.write_text("erase me")

    callback = ErasingUploadCallback(test_file)
    mapping = S3Mapping(loc_path=test_file, cloud_key="test/key")

    callback.on_success(mapping)

    assert not test_file.exists()


def test_erasing_upload_callback_already_erased(temp_dir):
    """Test ErasingUploadCallback handles already deleted file"""
    test_file = temp_dir / "already_gone.txt"

    callback = ErasingUploadCallback(test_file)
    mapping = S3Mapping(loc_path=test_file, cloud_key="test/key")

    # Should not raise even though file doesn't exist
    callback.on_success(mapping)


# ============================================================
# Tests: Edge Cases and Error Handling
# ============================================================

def test_concurrent_open_same_file(storage_manager, temp_base):
    """Test concurrent access to same file (potential race condition)"""
    import threading

    test_file = temp_base / "concurrent.txt"
    test_file.write_text("initial")

    errors = []

    def read_file():
        try:
            with storage_manager.open(Path("concurrent.txt"), mode="r") as f:
                content = f.read()
                time.sleep(0.01)  # Small delay
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=read_file) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should not have errors (reads are safe)
    assert len(errors) == 0


def test_exists_cloud_when_cloud_is_none(storage_manager):
    """Test _exists_cloud returns False when cloud is None"""
    result = storage_manager._exists_cloud(Path("test.txt"))
    assert result is False


def test_file_handle_closed_on_exception(storage_manager, temp_base):
    """Test that file handle is properly closed even on exception"""
    test_file = temp_base / "close_test.txt"
    test_file.write_text("test")

    try:
        with storage_manager.open(Path("close_test.txt"), mode="r") as f:
            content = f.read()
            raise ValueError("Test exception")
    except ValueError:
        pass

    # File should be closed and we can open it again
    with storage_manager.open(Path("close_test.txt"), mode="r") as f:
        content = f.read()

    assert content == "test"


# ============================================================
# Tests: Integration Scenarios
# ============================================================

def test_full_workflow_local_to_cloud(temp_base, temp_dir):
    """
    Integration test: Create file -> Finalize -> Upload -> Download -> Verify
    """
    local_cfg = LocalConfig(
        base_path=temp_base,
        save_local=True,
        cache_files=True,
        temp_base=temp_dir,
    )

    s3_cfg = S3Config(
        enable=True,
        bucket="test",
        access_key="key",
        secret_key="secret",
        region="region",
        root_prefix=Path("prefix"),
    )

    cloud_cfg = CloudConfig(enable=True, s3_config=s3_cfg)

    with patch("cclang.core.storage.S3Store") as mock_s3_class:
        mock_s3 = MagicMock(spec=S3Store)
        mock_s3_class.return_value = mock_s3

        uploaded_files = {}

        def mock_upload(local_path, relative_key, blocking=True, callback=None):
            uploaded_files[str(relative_key)] = local_path.read_text()
            if callback:
                mapping = S3Mapping(loc_path=local_path, cloud_key=str(relative_key))
                callback.on_success(mapping)

        def mock_download(cloud_key, local_path, blocking=True):
            content = uploaded_files.get(str(cloud_key))
            if content:
                local_path.write_text(content)

        mock_s3.upload.side_effect = mock_upload
        mock_s3.download.side_effect = mock_download
        mock_s3.exists.side_effect = lambda key: str(key) in uploaded_files

        sm = StorageManager(local_cfg, cloud_cfg)
        sm._cloud = mock_s3

        # 1. Create temp file
        temp_file = sm.create_temp_file(".artifact")
        temp_file.write_text("integration test data")

        # 2. Finalize artifact (moves and uploads)
        dest_path = Path("integration/artifact.txt")
        sm.finalize_artifact(temp_file, dest_path, blocking=True)

        # 3. Verify upload happened
        assert str(dest_path) in uploaded_files
        assert uploaded_files[str(dest_path)] == "integration test data"

        # 4. Delete local file to simulate download scenario
        (temp_base / "integration" / "artifact.txt").unlink()

        # 5. Download from cloud
        with sm.open(dest_path, mode="r") as f:
            content = f.read()

        # 6. Verify content matches
        assert content == "integration test data"

        sm.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
