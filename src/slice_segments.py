#!/usr/bin/env python3
"""
slice_segments.py
-----------------
Cut test clips (Ayesha1.wav, Sabeeh3.wav, U2.wav, ...) into fixed-length
windows so speaker-ID accuracy can be measured as a function of how much
audio the system sees (0.5 / 1 / 1.5 / 2 / 3 / 5 s).

Design choices (all overridable):
  * Windows are NON-overlapping by default, so every window is an independent
    trial. Trailing audio shorter than a full window is dropped (a short
    window would silently corrupt the duration axis).
  * Enrollment files (Name_15s.wav) are skipped automatically.
  * Ground truth is written to a manifest CSV, because the notebook's
    ground_truth_from_filename() regex will NOT parse slice filenames.
  * RMS level (dBFS) of every window is logged; use --min-dbfs to drop
    near-silent windows (pauses between words) if they pollute the results.

Usage
-----
  python slice_segments.py --input-dir tests --output-dir segments
  python slice_segments.py --input-dir . --output-dir segments --windows 1 2 3
  python slice_segments.py --input-dir tests --output-dir segments --min-dbfs -45

Output
------
  segments/
    0p5s/Ayesha1_0p5s_000.wav, Ayesha1_0p5s_001.wav, ...
    1s/  1p5s/  2s/  3s/  5s/
    segments_manifest.csv   <- path, source_file, ground_truth, window_s, ...

Requires: numpy, soundfile   (pip install soundfile; already present in Colab)
"""
import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_WINDOWS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
ENROLL_RE = re.compile(r"^[A-Za-z]+_\d+s\.wav$", re.IGNORECASE)  # e.g. Ayesha_15s.wav
MANIFEST_NAME = "segments_manifest.csv"


def ground_truth_from_filename(fn: str) -> str:
    """Same logic as the Sprint 1 notebook: 'Ayesha3.wav' -> 'Ayesha',
    'U2.wav' -> 'UNKNOWN', Colab ' (1)' duplicate suffixes stripped."""
    fn = re.sub(r"\s*\(\d+\)(?=\.\w+$)", "", fn)
    m = re.match(r"([A-Za-z]+)\d*\.wav", fn, re.IGNORECASE)
    name = m.group(1) if m else fn
    return "UNKNOWN" if name.upper().startswith("U") else name


def window_label(w: float) -> str:
    """0.5 -> '0p5s', 1.0 -> '1s', 1.5 -> '1p5s'."""
    return f"{w:g}".replace(".", "p") + "s"


def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * np.log10(max(rms, 1e-10))


def n_windows(n_samples: int, win: int, hop: int) -> int:
    return 0 if n_samples < win else 1 + (n_samples - win) // hop


def find_inputs(input_dir: Path, output_dir: Path, pattern: str):
    out_resolved = output_dir.resolve()
    files = []
    for p in sorted(input_dir.rglob(pattern)):
        if not p.is_file():
            continue
        if out_resolved in p.resolve().parents:  # don't re-slice our own output
            continue
        if ENROLL_RE.match(p.name):  # enrollment audio is not a test clip
            continue
        files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=".", help="folder containing test clips (searched recursively)")
    ap.add_argument("--output-dir", default="segments", help="where to write windows + manifest")
    ap.add_argument("--windows", type=float, nargs="+", default=DEFAULT_WINDOWS,
                    help="window lengths in seconds (default: 0.5 1 1.5 2 3 5)")
    ap.add_argument("--pattern", default="*.wav", help="glob for input files (default: *.wav)")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="fractional overlap between consecutive windows, 0 <= x < 1 (default 0)")
    ap.add_argument("--min-dbfs", type=float, default=None,
                    help="skip windows quieter than this RMS level, e.g. -45 (default: keep all)")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete existing window folders for the requested lengths before writing")
    args = ap.parse_args()

    if not 0.0 <= args.overlap < 1.0:
        sys.exit("--overlap must be in [0, 1)")
    if any(w <= 0 for w in args.windows):
        sys.exit("window lengths must be positive")

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    windows = sorted(set(args.windows))
    files = find_inputs(in_dir, out_dir, args.pattern)
    if not files:
        sys.exit(f"No test clips found in {in_dir.resolve()} (pattern {args.pattern}; "
                 f"enrollment-style names like Name_15s.wav are skipped).")

    # Refuse to mix runs: stale windows from old settings would contaminate the experiment.
    for w in windows:
        d = out_dir / window_label(w)
        if d.exists() and any(d.iterdir()):
            if args.overwrite:
                shutil.rmtree(d)
            else:
                sys.exit(f"{d} already has files. Re-run with --overwrite to replace them.")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    coverage = defaultdict(dict)  # clip -> {window: n_kept}
    leftover = {}                 # clip -> {window: dropped seconds}
    durations = {}

    for f in files:
        data, sr = sf.read(str(f), always_2d=True)  # (frames, channels), float64
        durations[f.name] = len(data) / sr
        truth = ground_truth_from_filename(f.name)
        leftover[f.name] = {}

        for w in windows:
            win = int(round(w * sr))
            hop = max(1, int(round(win * (1.0 - args.overlap))))
            n = n_windows(len(data), win, hop)
            label = window_label(w)
            kept = 0
            for i in range(n):
                s = i * hop
                seg = data[s:s + win]
                level = rms_dbfs(seg)
                keep = args.min_dbfs is None or level >= args.min_dbfs
                rel = ""
                if keep:
                    d = out_dir / label
                    d.mkdir(parents=True, exist_ok=True)
                    rel = f"{label}/{f.stem}_{label}_{i:03d}.wav"
                    sf.write(str(out_dir / rel), seg, sr, subtype="PCM_16")
                    kept += 1
                rows.append({
                    "path": rel,
                    "source_file": f.name,
                    "ground_truth": truth,
                    "window_s": w,
                    "index": i,
                    "start_s": round(s / sr, 4),
                    "end_s": round((s + win) / sr, 4),
                    "rms_dbfs": round(level, 2),
                    "kept": keep,
                })
            coverage[f.name][w] = kept
            used_end = ((n - 1) * hop + win) if n else 0
            leftover[f.name][w] = round((len(data) - used_end) / sr, 2) if n else round(len(data) / sr, 2)

    with open(out_dir / MANIFEST_NAME, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["path"])
        wr.writeheader()
        wr.writerows(rows)

    # ---- summary -------------------------------------------------------
    print(f"\nSliced {len(files)} clips -> {out_dir.resolve()}")
    print(f"Windows: {', '.join(window_label(w) for w in windows)} | overlap: {args.overlap:g} | "
          f"min-dbfs: {args.min_dbfs}\n")
    head = f"{'clip':<16}{'len(s)':>8}" + "".join(f"{window_label(w):>7}" for w in windows)
    print(head)
    print("-" * len(head))
    for name in sorted(coverage):
        print(f"{name:<16}{durations[name]:>8.1f}" + "".join(f"{coverage[name][w]:>7}" for w in windows))

    print("\nWindows per (ground truth x window length):")
    tot = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["kept"]:
            tot[r["ground_truth"]][r["window_s"]] += 1
    head = f"{'speaker':<12}" + "".join(f"{window_label(w):>7}" for w in windows)
    print(head)
    print("-" * len(head))
    for spk in sorted(tot):
        print(f"{spk:<12}" + "".join(f"{tot[spk][w]:>7}" for w in windows))

    empty = [(n, w) for n, d in coverage.items() for w, k in d.items() if k == 0]
    if empty:
        print("\n⚠ Clips with ZERO windows at some lengths (clip shorter than window, or all silent):")
        for n, w in empty:
            print(f"   {n} @ {window_label(w)}")
    print(f"\nManifest: {out_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
