import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tables as tb

from scripts.irene_event_catalog import EventCatalog
from scripts.dev.irene_catalog import catalog_response, scan


class LocalIreneCatalogTest(unittest.TestCase):
    def test_copied_waveforms_match_catalog_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for counter, ldc in ((3, 1), (4, 1), (4, 2)):
                path = root / "16083" / "hdf5" / "data" / f"ldc{ldc}" / f"run_16083_{counter:04d}_ldc{ldc}_trg0.waveforms.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                with tb.open_file(path, "w") as h5:
                    h5.create_group("/", "Run")
                    events = h5.create_table("/Run", "events", {"evt_number": tb.Int32Col()})
                    row = events.row
                    row["evt_number"] = 1
                    row.append()
                    events.flush()
                    h5.create_group("/", "RD")
                    h5.create_carray("/RD", "pmtrwf", obj=np.zeros((1, 1, 2), dtype=np.int16))
                    h5.create_carray("/RD", "sipmrwf", obj=np.zeros((1, 1, 2), dtype=np.int16))
            self.assertEqual(len(scan(root)), 3)
            catalog = EventCatalog("http://127.0.0.1:32121", root)
            def route(path, query=None):
                return catalog_response(scan(root), path, {
                    key: [str(value)] for key, value in (query or {}).items()
                })
            with patch.object(catalog, "_get", side_effect=route):
                self.assertEqual(catalog.runs(), [16083])
                self.assertEqual(catalog.ldcs(16083), [(1, 2), (2, 1)])
                page, total = catalog.files(16083, 1, limit=1, offset=1)
                self.assertEqual(total, 2)
                self.assertEqual(len(page), 1)
                self.assertEqual(page[0]["counter"], 4)
                self.assertTrue(catalog.file_path(page[0]["id"], 16083, 1).is_file())


if __name__ == "__main__":
    unittest.main()
