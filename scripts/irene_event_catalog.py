"""Read NEXT100 waveform metadata from next-evt-db without scanning /analysis."""

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class CatalogError(RuntimeError):
    pass


class EventCatalog:
    def __init__(self, base_url, analysis_root, timeout=5):
        self.base_url = base_url.rstrip("/")
        self.analysis_root = Path(analysis_root).resolve()
        self.timeout = timeout

    def _get(self, path, query=None):
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)
        try:
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=self.timeout) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise CatalogError("NEXT event catalog request failed: {}".format(exc)) from exc
        if not isinstance(payload, dict):
            raise CatalogError("NEXT event catalog returned an invalid response")
        return payload

    def runs(self):
        payload = self._get("/api/v1/runs")
        try:
            return [int(run) for run in payload["runs"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError("NEXT event catalog returned an invalid run list") from exc

    def ldcs(self, run):
        payload = self._get("/api/v1/runs/{}/ldcs".format(int(run)))
        try:
            return [(int(item["ldc"]), int(item["file_count"])) for item in payload["ldcs"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError("NEXT event catalog returned an invalid LDC list") from exc

    def files(self, run, ldc, limit=100, offset=0):
        payload = self._get(
            "/api/v1/runs/{}/files".format(int(run)),
            {"ldc": int(ldc), "limit": int(limit), "offset": int(offset)},
        )
        try:
            files = payload["files"]
            total = int(payload["total"])
            if not isinstance(files, list) or total < 0 or len(files) > limit:
                raise ValueError("invalid files or total")
            for item in files:
                if not isinstance(item, dict):
                    raise ValueError("invalid file record")
                if int(item["run"]) != int(run) or int(item["ldc"]) != int(ldc):
                    raise ValueError("file outside selected run or LDC")
                for key in ("id", "counter", "trigger", "event_count"):
                    int(item[key])
            return files, total
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError("NEXT event catalog returned an invalid file page") from exc

    def file_path(self, file_id, run, ldc):
        payload = self._get("/api/v1/files/{}".format(int(file_id)))
        try:
            item = payload["file"]
            if (int(item["id"]), int(item["run"]), int(item["ldc"])) != (
                int(file_id), int(run), int(ldc)
            ):
                raise ValueError("file identity changed")
            raw_path = item["path"]
            if not isinstance(raw_path, str) or not raw_path.startswith("/"):
                raise ValueError("invalid waveform path")
            path = Path(raw_path).resolve()
            if path == self.analysis_root or self.analysis_root not in path.parents:
                raise ValueError("waveform path is outside the analysis root")
            if not path.name.endswith(".waveforms.h5"):
                raise ValueError("not a waveform file")
            return path
        except (KeyError, TypeError, ValueError) as exc:
            raise CatalogError("NEXT event catalog returned an invalid file record: {}".format(exc)) from exc
