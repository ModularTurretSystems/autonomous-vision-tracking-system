def normalize_extension(
    extensions: str | tuple[str, ...],
    allowed: tuple[str, ...]
) -> tuple[str, ...]:
    if isinstance(extensions, str): extensions = (extensions, )
    
    normalized: list[str] = []

    for ext in extensions:
        if ext.lower() not in allowed:
            raise ValueError(f"Extension '{ext}' is not allowed. Allowed: {allowed}")
        normalized.append(ext.lower())

    return tuple(normalized)
