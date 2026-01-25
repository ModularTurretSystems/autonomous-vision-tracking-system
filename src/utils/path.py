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
