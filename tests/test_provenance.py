import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from embodied_silent_failures.provenance import (
    file_sha256,
    json_sha256,
    load_json,
    source_file_record,
)


class ProvenanceTests(unittest.TestCase):
    def test_file_and_json_hashes_are_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"recorded artifact")

            digest = file_sha256(path)

        self.assertEqual(digest, hashlib.sha256(b"recorded artifact").hexdigest())
        self.assertEqual(json_sha256({"b": 2, "a": 1}), json_sha256({"a": 1, "b": 2}))

    def test_json_loader_requires_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_path = root / "object.json"
            list_path = root / "list.json"
            object_path.write_text(json.dumps({"value": 1}), encoding="utf-8")
            list_path.write_text(json.dumps([1]), encoding="utf-8")

            self.assertEqual(load_json(object_path), {"value": 1})
            with self.assertRaises(ValueError):
                load_json(list_path)

    def test_json_loader_reads_gzip_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as file:
                json.dump({"value": 1}, file)

            self.assertEqual(load_json(path), {"value": 1})

    def test_source_record_names_and_hashes_the_implementation_file(self) -> None:
        record = source_file_record(self)

        self.assertEqual(record["class"], f"{type(self).__module__}.{type(self).__qualname__}")
        self.assertEqual(record["sha256"], file_sha256(Path(record["path"])))


if __name__ == "__main__":
    unittest.main()
