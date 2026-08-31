from __future__ import annotations

import os
import shutil
import tempfile


def replace_with_link_or_copy(source: str, destination: str) -> None:
    """Atomically update an alias without duplicating storage when links are supported."""
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(dir=directory)
    os.close(file_descriptor)
    os.unlink(temporary_path)
    try:
        try:
            os.link(source, temporary_path)
        except OSError:
            shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
