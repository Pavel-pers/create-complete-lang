# ==== Exceptions =============================================================


class FileManagerError(Exception):
    """Base exception for FileManager-related errors."""


class LocalStorageError(FileManagerError):
    """Errors related to local filesystem operations."""


class LocalPathError(LocalStorageError, ValueError):
    """Invalid or unsafe path for the configured local base directory."""


class CloudStorageError(FileManagerError):
    """Errors related to cloud (S3-compatible) operations."""


class CloudNotConfiguredError(CloudStorageError):
    """Cloud operation was requested but cloud backend is disabled."""


class FileNotFoundInStorageError(LocalStorageError, FileNotFoundError):
    """File was not found neither locally nor in cloud storage."""