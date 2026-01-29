from pathlib import Path


def ensure_directory(path: Path | str) -> Path:
    """
    Ensure that the given path exists and refers to a directory.

    If the directory does not exist, it is created along with any missing
    parent directories.

    Parameters
    ----------
    path : Path or str
        Path to the target directory.

    Returns
    -------
    Path
        Path object pointing to the existing directory.

    Raises
    ------
    NotADirectoryError
        If the path exists but is not a directory.
    OSError
        If the directory cannot be created.
    """

    p = Path(path)

    if not p.exists():
        try: p.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create directory {p}: {e}") from e
    elif not p.is_dir():
        raise(NotADirectoryError(f"The path exists but is not a directory: {p}"))
    
    return p


def ensure_file_path(file_path: Path | str) -> Path:
    """
    Ensure that the given file path is valid and its parent directory exists.

    If the parent directories do not exist, they will be created.

    Parameters
    ----------
    file_path : Path or str
        Path to the target file.

    Returns
    -------
    Path
        Path object pointing to the file (may not exist yet).

    Raises
    ------
    OSError
        If the parent directory cannot be created.
    ValueError
        If the file_path refers to an existing directory instead of a file.
    """

    f = Path(file_path)

    if f.exists() and f.is_dir():
        raise ValueError(f"The path exists but is a directory, not a file: {f}")

    parent_dir = f.parent
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create parent directory {parent_dir}: {e}") from e

    return f
