"""Local next-evt-db subset for Irene development with copied waveforms."""

import argparse
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import tables as tb


FILE_NAME = re.compile(r"run_(\d+)_(\d+)_ldc(\d+)_(?:trg|trigger)(\d+).*\.waveforms\.h5$", re.I)


def scan(root: Path) -> list[dict]:
    """Index copied files; only the small /Run/events table is read."""
    files = []
    for path in sorted(root.rglob("*.waveforms.h5")):
        match = FILE_NAME.fullmatch(path.name)
        if not match:
            continue
        try:
            with tb.open_file(path, "r") as h5:
                event_count = int(h5.root.Run.events.nrows)
                h5.root.RD.pmtrwf.shape
                h5.root.RD.sipmrwf.shape
        except (OSError, tb.NoSuchNodeError):
            continue
        run, counter, ldc, trigger = map(int, match.groups())
        files.append({"run": run, "counter": counter, "ldc": ldc,
                      "trigger": trigger, "path": str(path.resolve()),
                      "event_count": event_count})
    files.sort(key=lambda f: (f["run"], f["counter"], f["ldc"], f["trigger"], f["path"]))
    for file_id, item in enumerate(files, 1):
        item["id"] = file_id
    return files


def catalog_response(files: list[dict], path: str, query: dict[str, list[str]]) -> dict:
    if path == "/healthz":
        result = {"ok": True, "catalog": {"status": "ready", "api_version": 1,
                  "runs": len({f["run"] for f in files}), "files": len(files)}}
    elif path == "/api/v1/runs":
        runs = sorted({f["run"] for f in files})
        result = {"runs": runs, "latest": runs[-1] if runs else None}
    elif match := re.fullmatch(r"/api/v1/runs/(\d+)/ldcs", path):
        run = int(match.group(1))
        selected = [f for f in files if f["run"] == run]
        if not selected:
            raise KeyError("run was not found")
        ldcs = sorted({f["ldc"] for f in selected})
        result = {"run": run, "ldcs": [{"ldc": ldc, "file_count": sum(f["ldc"] == ldc for f in selected)}
                                        for ldc in ldcs]}
    elif match := re.fullmatch(r"/api/v1/runs/(\d+)/files", path):
        run = int(match.group(1))
        if run not in {f["run"] for f in files}:
            raise KeyError("run was not found")
        limit = int(query.get("limit", ["100"])[0])
        offset = int(query.get("offset", ["0"])[0])
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("invalid pagination")
        selected = [f for f in files if f["run"] == run]
        if "ldc" in query:
            selected = [f for f in selected if f["ldc"] == int(query["ldc"][0])]
        page = selected[offset:offset + limit]
        result = {"run": run, "files": [{k: f[k] for k in
                  ("id", "run", "counter", "ldc", "trigger", "event_count")} for f in page],
                  "limit": limit, "offset": offset, "total": len(selected),
                  "next_offset": offset + len(page) if offset + len(page) < len(selected) else None}
    elif match := re.fullmatch(r"/api/v1/files/(\d+)", path):
        file_id = int(match.group(1))
        item = next((f for f in files if f["id"] == file_id), None)
        if item is None:
            raise KeyError("file was not found")
        result = {"file": item}
    else:
        raise KeyError("route was not found")
    return result


class Handler(BaseHTTPRequestHandler):
    root: Path

    def reply(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            self.reply(HTTPStatus.OK, catalog_response(scan(self.root), parsed.path, parse_qs(parsed.query)))
        except KeyError as exc:
            self.reply(HTTPStatus.NOT_FOUND, {"detail": str(exc)})
        except (ValueError, TypeError) as exc:
            self.reply(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=32121)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    files = scan(root)
    if not files:
        parser.error(f"No readable NEXT100 waveform files under {root}")
    Handler.root = root
    print(f"Local Irene catalog: {len(files)} files under {root}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
