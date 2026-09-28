import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.irene_event_catalog import CatalogError, EventCatalog


class EventCatalogTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.catalog = EventCatalog("http://next-evt-db:2112", self.root.name)

    def test_catalog_metadata_uses_indexed_endpoints(self):
        file_record = {"id": 42, "run": 16081, "ldc": 1, "counter": 42, "trigger": 0, "event_count": 12}
        with patch.object(self.catalog, "_get", side_effect=[
            {"runs": [16080, 16081]},
            {"ldcs": [{"ldc": 1, "file_count": 203}]},
            {"files": [file_record], "total": 203},
        ]) as get:
            self.assertEqual(self.catalog.runs(), [16080, 16081])
            self.assertEqual(self.catalog.ldcs(16081), [(1, 203)])
            self.assertEqual(self.catalog.files(16081, 1, 100, 200), ([file_record], 203))
        self.assertEqual(get.call_args_list[0].args, ("/api/v1/runs",))
        self.assertEqual(get.call_args_list[1].args, ("/api/v1/runs/16081/ldcs",))
        self.assertEqual(get.call_args_list[2].args[0], "/api/v1/runs/16081/files")
        self.assertEqual(get.call_args_list[2].args[1]["offset"], 200)

    def test_selected_file_path_must_match_and_stay_under_analysis(self):
        path = Path(self.root.name) / "16081/hdf5/data/ldc1/run_16081_42_ldc1_trg0.waveforms.h5"
        record = {"file": {"id": 42, "run": 16081, "ldc": 1, "path": str(path)}}
        with patch.object(self.catalog, "_get", return_value=record):
            self.assertEqual(self.catalog.file_path(42, 16081, 1), path)
            with self.assertRaises(CatalogError):
                self.catalog.file_path(43, 16081, 1)
        record["file"]["path"] = "/tmp/outside.waveforms.h5"
        with patch.object(self.catalog, "_get", return_value=record):
            with self.assertRaises(CatalogError):
                self.catalog.file_path(42, 16081, 1)


if __name__ == "__main__":
    unittest.main()
