# Irene local development

The local catalog emulator lets Irene use copied Canfranc waveform files
without connecting to the production `next-evt-db` service.

## Prepare an environment and files

From the IC checkout, activate its Python environment as described in the
top-level `README.rst`:

```bash
source manage.sh activate
python -m pip install streamlit plotly
```

The environment must also have IC and PyTables available. Copy one or more
NEXT100 waveform HDF5 files from Canfranc into `scripts/dev/data/`, preserving
their filenames. This directory is ignored by Git. You can keep their
run/LDC folder structure or place them directly in the directory; the local
catalog scans recursively. Filenames must include
`run_<run>_<counter>_ldc<ldc>_trg<trigger>.waveforms.h5` (or
`_trigger<trigger>`). The files must contain `/Run/events`, `/RD/pmtrwf`, and
`/RD/sipmrwf`.

## Run

From the IC checkout, with that environment active:

```bash
python scripts/dev/run_irene.py
```

Open `http://127.0.0.1:8501/`. The launcher starts both the catalog emulator
and Streamlit on localhost. It sets `ICTDIR`, `ICDIR`, `IRENE_DATA_DIR`, and
`IRENE_CATALOG_URL` for the app. If the copied files are elsewhere, pass
`--data-dir PATH`. `--port` and `--catalog-port` select different localhost
ports; `--python PATH` selects a different Python for Streamlit.

The emulator implements the four catalog routes Irene uses: runs, LDC counts,
paged files, and exact file details. It indexes filenames and event counts in
`/Run/events`; Irene opens the selected local HDF5 file for waveform data.
It rescans the directory on each catalog request. Irene caches catalog
responses for 10 seconds; its **Refresh** button clears that cache.
