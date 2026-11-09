"""
Phase 1.1 — Get the BIRD and Spider datasets.

Two jobs, two functions:
  download_all()  -> run ONCE. Downloads the official zips and stores them in
                     Google Drive: MyDrive/text2sql/data/raw/zips/
  unpack_all()    -> run at the start of EVERY Colab session that needs data.
                     Unzips from Drive onto Colab's local disk (/content/data/raw),
                     which is much faster for SQLite than reading through Drive.
  check_data()    -> prints counts so you can confirm everything arrived.

Typical Colab use:
    from src.get_data import download_all, unpack_all, check_data
    download_all()                               # once, ever
    paths = unpack_all(include_bird_train=True)  # each session
    check_data(paths)
"""

import hashlib
import json
import shutil
import sqlite3
import subprocess
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------- locations
DRIVE_ZIPS = Path("/content/drive/MyDrive/text2sql/data/raw/zips")  # permanent
LOCAL_RAW = Path("/content/data/raw")      # fast, but wiped when the session ends
TMP = Path("/content/tmp_downloads")       # scratch space while downloading

GB = 1024 ** 3

# ---------------------------------------------------------------- sources
SOURCES = {
    "bird_dev": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        "zip": "bird_dev.zip",
        "license": "CC BY-SA 4.0",
    },
    "bird_train": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
        "zip": "bird_train.zip",
        "license": "CC BY-SA 4.0",
    },
    "spider": {
        # Official Yale release (Google Drive file id)
        "gdrive_id": "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J",
        # Fallback: community re-host of the same zip on Hugging Face
        "hf_repo": "HAL-9001/spider-databases",
        "hf_file": "spider_data.zip",
        "sha256": "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
        "zip": "spider_data.zip",
        "license": "CC BY-SA 4.0",
    },
}

# Where each dataset's question files and database folder live after unzipping.
# Patterns are searched recursively, so changing folder names (e.g. dev_20240627)
# don't break anything.
LAYOUT = {
    "bird_dev": {"questions": {"dev": "dev.json"}, "db_dir": "dev_databases"},
    "bird_train": {"questions": {"train": "train.json"}, "db_dir": "train_databases"},
    "spider": {
        "questions": {"train": "train_spider.json", "dev": "dev.json"},
        "db_dir": "database",
    },
}


# ================================================================ helpers
def sha256_of(path, chunk=8 * 1024 * 1024):
    """Fingerprint a file; reading in chunks keeps memory use small."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def remote_size(url):
    """Ask the server for the file size in bytes without downloading it (0 = unknown)."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def _download_url(url, dest):
    """Download with wget. -c resumes a partial file if the connection drops."""
    original = dest.parent / url.rsplit("/", 1)[-1]  # e.g. dev.zip
    subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-P", str(dest.parent), url],
        check=True,
    )
    original.rename(dest)


def _download_spider(src, dest):
    """Try the official Google Drive link; fall back to the Hugging Face re-host."""
    try:
        import gdown  # preinstalled on Colab

        gdown.download(id=src["gdrive_id"], output=str(dest), quiet=False)
        if not zipfile.is_zipfile(dest):
            raise RuntimeError("Google Drive did not return a zip (often a download-quota limit)")
        print("  Spider: downloaded from the official Google Drive link.")
    except Exception as e:
        print(f"  Official link failed ({e}). Trying the Hugging Face re-host...")
        dest.unlink(missing_ok=True)
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(
            repo_id=src["hf_repo"], filename=src["hf_file"],
            repo_type="dataset", local_dir=str(dest.parent),
        ))
        if p != dest:
            shutil.move(str(p), str(dest))
        print("  Spider: downloaded from the Hugging Face re-host.")


def _unzip(zip_path, out_dir):
    """Extract a zip, skipping macOS junk files (__MACOSX/, ._name)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [
            m for m in z.namelist()
            if not m.startswith("__MACOSX") and not Path(m).name.startswith("._")
        ]
        z.extractall(out_dir, members=members)


def _unzip_all(zip_path, out_dir):
    """Extract a zip, then keep extracting any zips found inside (BIRD nests them)."""
    _unzip(zip_path, out_dir)
    while True:
        inner = list(Path(out_dir).rglob("*.zip"))
        if not inner:
            break
        for p in inner:
            _unzip(p, p.parent)
            p.unlink()  # delete the inner zip to save disk space


def _find_one(root, name):
    """Find a file or folder by exact name anywhere under root."""
    hits = sorted(p for p in Path(root).rglob(name) if "__MACOSX" not in p.parts)
    if not hits:
        raise FileNotFoundError(f"'{name}' not found under {root}")
    return hits[0]


# ================================================================ main steps
def download_all(names=("bird_dev", "spider", "bird_train")):
    """Download each dataset once and store the zip in Drive. Safe to re-run:
    anything already in Drive is skipped. Small datasets go first so the test
    sets land even if the big BIRD train download fails."""
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Mount Google Drive first: drive.mount('/content/drive')")
    DRIVE_ZIPS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    manifest_path = DRIVE_ZIPS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for name in names:
        src = SOURCES[name]
        final = DRIVE_ZIPS / src["zip"]
        if final.exists() and name in manifest:
            print(f"✓ {name}: already in Drive ({final.stat().st_size / GB:.2f} GB), skipping")
            continue

        # Check Drive space BEFORE spending time downloading.
        # (Colab reports Drive free space approximately.)
        size = remote_size(src["url"]) if "url" in src else 0
        free = shutil.disk_usage(DRIVE_ZIPS).free
        print(f"→ {name}: zip size {size / GB:.2f} GB (0.00 = unknown), "
              f"Drive free ≈ {free / GB:.2f} GB")
        if size and size > free - 0.5 * GB:
            print(f"  ✗ Not enough Drive space for {name}; skipped. Free up space and re-run.")
            continue

        local = TMP / src["zip"]
        if name == "spider":
            _download_spider(src, local)
        else:
            _download_url(src["url"], local)

        if not zipfile.is_zipfile(local):
            raise RuntimeError(f"{name}: downloaded file is not a valid zip")

        print(f"  Computing checksum...")
        digest = sha256_of(local)
        if "sha256" in src:
            status = "matches" if digest == src["sha256"] else "DIFFERS from"
            print(f"  SHA-256 {status} the published checksum")

        print(f"  Copying to Drive (big files take a few minutes)...")
        shutil.copyfile(local, final)
        local.unlink()  # free Colab disk

        manifest[name] = {
            "file": src["zip"],
            "source": src.get("url") or f"gdrive:{src['gdrive_id']}",
            "bytes": final.stat().st_size,
            "sha256": digest,
            "license": src["license"],
            "downloaded": str(date.today()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))  # save after each one
        print(f"✓ {name}: saved to {final}")

    print("\nManifest:\n" + json.dumps(manifest, indent=2))
    return manifest


def unpack_all(include_bird_train=False, zip_dir=DRIVE_ZIPS, out_dir=LOCAL_RAW):
    """Unzip datasets to fast local disk and return their paths.
    BIRD train is off by default: it's big and only needed from Phase 2 on.
    On your laptop: unpack_all(zip_dir="data/raw/zips", out_dir="data/raw")."""
    zip_dir, out_dir = Path(zip_dir), Path(out_dir)
    names = ["bird_dev", "spider"] + (["bird_train"] if include_bird_train else [])
    paths = {}

    for name in names:
        dest = out_dir / name
        marker = dest / ".unpacked"  # tells us a previous unpack finished
        if not marker.exists():
            zip_path = zip_dir / SOURCES[name]["zip"]
            if not zip_path.exists():
                print(f"✗ {name}: {zip_path} not found; run download_all() first. Skipping.")
                continue
            print(f"Unpacking {name} ...")
            shutil.rmtree(dest, ignore_errors=True)  # clear any half-finished attempt
            dest.mkdir(parents=True)
            _unzip_all(zip_path, dest)
            marker.touch()

        layout = LAYOUT[name]
        paths[name] = {
            "questions": {s: str(_find_one(dest, f)) for s, f in layout["questions"].items()},
            "db_dir": str(_find_one(dest, layout["db_dir"])),
        }

    (out_dir / "paths.json").write_text(json.dumps(paths, indent=2))
    print(f"✓ Paths saved to {out_dir / 'paths.json'}")
    return paths


def check_data(paths):
    """Print question and database counts; open one database read-only."""
    for name, info in paths.items():
        db_dir = Path(info["db_dir"])
        db_ids = sorted(d.name for d in db_dir.iterdir() if d.is_dir())
        print(f"\n{name}: {len(db_ids)} database folders in {db_dir}")

        for split, qpath in info["questions"].items():
            qs = json.loads(Path(qpath).read_text())
            used = {q["db_id"] for q in qs}
            missing = used - set(db_ids)
            print(f"  {split}: {len(qs)} questions over {len(used)} databases"
                  f"  | missing DB folders: {len(missing)}")

        # Open one database read-only (same way the eval harness will) and count tables
        sample = db_ids[0]
        db_file = db_dir / sample / f"{sample}.sqlite"
        con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        n_tables = con.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        con.close()
        print(f"  sample DB '{sample}': {n_tables} tables (opened read-only ✓)")