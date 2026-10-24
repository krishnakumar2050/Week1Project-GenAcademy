"""Deterministic bounded repository snapshots; never executes repository code."""

import fnmatch
import os
import stat
from pathlib import Path

EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".cs",
    ".sh",
    ".sql",
    ".toml",
}
EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
}


def load_files(root: Path, max_bytes=120_000, exclude: tuple[str, ...] = ()):
    if max_bytes <= 0:
        raise ValueError("Snapshot byte budget must be positive")
    if not root.is_dir():
        raise ValueError("Repository must be a directory")
    files = {}
    total = 0

    def traversal_error(error):
        raise error

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=traversal_error):

        def excluded(path):
            relative = path.relative_to(root).as_posix()
            return any(
                fnmatch.fnmatchcase(relative, pattern)
                or fnmatch.fnmatchcase(relative + "/", pattern)
                for pattern in exclude
            )

        dirs[:] = sorted(
            d
            for d in dirs
            if d not in EXCLUDED
            and not d.startswith(".")
            and not (Path(directory) / d).is_symlink()
            and not excluded(Path(directory) / d)
        )
        for name in sorted(names):
            path = Path(directory) / name
            if (
                path.is_symlink()
                or name.startswith(".")
                or path.suffix not in EXTENSIONS
                or excluded(path)
            ):
                continue
            if path.stat().st_size > max_bytes - total:
                raise ValueError(
                    "Snapshot exceeds byte budget; select a smaller directory or increase --max-bytes"
                )
            # Bound reads even if the file grows after stat; reject symlinks at open.
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            )
            with os.fdopen(descriptor, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError(f"Source is not a regular file: {path.relative_to(root)}")
                data = handle.read(max_bytes - total + 1)
            total += len(data)
            if total > max_bytes:
                raise ValueError("Snapshot exceeds byte budget")
            files[path.relative_to(root).as_posix()] = data.decode("utf-8")
    if not files:
        raise ValueError("No supported source files found")
    return files
