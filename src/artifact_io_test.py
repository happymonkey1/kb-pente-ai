from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.artifact_io import replace_with_link_or_copy


class ReplaceWithLinkOrCopyTest(unittest.TestCase):
    def test_atomically_replaces_existing_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "latest"
            source.write_bytes(b"new-generation")
            destination.write_bytes(b"old-generation")

            replace_with_link_or_copy(str(source), str(destination))

            self.assertEqual(b"new-generation", destination.read_bytes())

    def test_falls_back_to_copy_when_hard_links_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "latest"
            source.write_bytes(b"portable-generation")

            with patch("src.artifact_io.os.link", side_effect=OSError("unsupported")):
                replace_with_link_or_copy(str(source), str(destination))

            self.assertEqual(b"portable-generation", destination.read_bytes())


if __name__ == "__main__":
    unittest.main()
