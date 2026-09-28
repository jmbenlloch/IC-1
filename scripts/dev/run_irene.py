"""Start Irene and its local catalog using copied NEXT100 waveform files."""

import argparse
import os
import subprocess
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from irene_catalog import Handler, scan


def main() -> None:
    ic_dir = Path(__file__).resolve().parents[2]
    default_data = Path(__file__).resolve().parent / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--catalog-port", type=int, default=32121)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    root = args.data_dir.resolve()
    files = scan(root)
    if not files:
        parser.error(f"No readable NEXT100 waveform files under {root}; copy Canfranc files first")
    Handler.root = root
    catalog = ThreadingHTTPServer(("127.0.0.1", args.catalog_port), Handler)
    Thread(target=catalog.serve_forever, daemon=True).start()
    env = dict(os.environ, ICTDIR=str(ic_dir), ICDIR=str(ic_dir / "invisible_cities"),
               IRENE_DATA_DIR=str(root), IRENE_CATALOG_URL=f"http://127.0.0.1:{args.catalog_port}",
               STREAMLIT_BROWSER_GATHER_USAGE_STATS="false",
               PYTHONPATH=str(ic_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    print(f"Local Irene catalog: {len(files)} files under {root}", flush=True)
    print(f"Open http://127.0.0.1:{args.port}/", flush=True)
    try:
        subprocess.run([str(args.python), "-m", "streamlit", "run", "scripts/irene_interactive_app.py",
                        f"--server.address=127.0.0.1", f"--server.port={args.port}",
                        "--server.headless=true"], cwd=ic_dir, env=env, check=True)
    except KeyboardInterrupt:
        pass
    finally:
        catalog.shutdown()
        catalog.server_close()


if __name__ == "__main__":
    main()
