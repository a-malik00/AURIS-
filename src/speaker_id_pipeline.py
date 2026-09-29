"""
speaker_id_pipeline.py  --  Sprint 1 speaker-ID experiment, made repeatable
===========================================================================

One call re-runs the whole experiment and refreshes every output:

    import speaker_id_pipeline as sp
    embed = sp.make_pyannote_embedder(HF_TOKEN)
    out   = sp.run_experiment(embed, root="/content/drive/MyDrive/speaker_id_fyp",
                              sheet_name="FYP Speaker ID Results")

Folder layout under `root` (all on Google Drive so nothing dies with the runtime):

    audio/enrollment/   Ayesha_5s.wav  Ayesha_10s.wav ... Sabeeh_30s.wav  <new voices go here>
    audio/tests/        Ayesha1.wav ... Sabeeh5.wav, U1.wav ...  (subfolders OK, e.g. tests/unknown/)
    enrolled_speakers.json      <- embeddings + model id + threshold (written by this module)
    test_embeddings_cache.json  <- test-clip embeddings (so re-runs are fast)
    results/*.csv               <- latest results;  results/archive/<run_id>/ keeps every run

Design rules (why the outputs are trustworthy):
  1. Raw cosine scores against EVERY enrolled speaker are logged for EVERY clip,
     unknown speakers included. Decisions are derived from scores, never the reverse.
  2. ONE threshold per run, stored in enrolled_speakers.json and written into every row.
  3. The threshold is calibrated on the "dev" clips (odd clip numbers) and threshold-
     dependent metrics are reported on the "eval" clips (even numbers) -- never on
     the clips the threshold was tuned on.
  4. Every rate comes with a Wilson 95% interval, because n is small.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
UNKNOWN = "UNKNOWN"

ENROLL_STEM_RE = re.compile(r"^(?P<name>[A-Za-z]+)_(?P<dur>\d+)s(?:_\w+)?$")   # Ayesha_15s, Ayesha_15s_b (extra take)
TEST_STEM_RE = re.compile(r"^(?P<name>[A-Za-z]+)(?P<num>\d+)$")       # Ayesha3, U2
UNKNOWN_STEM_RE = re.compile(r"^U(nknown)?\d+$", re.IGNORECASE)        # U1, Unknown2
DUP_SUFFIX_RE = re.compile(r"\s*\(\d+\)$")                             # Colab's " (1)"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _canon(name: str) -> str:
    return name[:1].upper() + name[1:].lower()


# --------------------------------------------------------------------------- embedder
def make_pyannote_embedder(hf_token: str, model_id: str = MODEL_ID):
    """Returns embed(path) -> L2-normalised 1-D numpy vector (same model/settings as the notebook)."""
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(model_id, token=hf_token)
    inference = Inference(model, window="whole")

    def embed(path) -> np.ndarray:
        e = np.asarray(inference(str(path))).flatten().astype(np.float64)
        return e / np.linalg.norm(e)

    embed.model_id = model_id
    return embed


def make_wavlm_embedder(model_id: str = "microsoft/wavlm-base-plus-sv"):
    """WavLM X-vector speaker embedding (HuggingFace transformers). Returns embed(path) -> L2-normalised vector.
    Needs: pip install -q transformers librosa soundfile
    Model card: https://huggingface.co/microsoft/wavlm-base-plus-sv (X-vector head, trained on VoxCeleb1)."""
    import librosa
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = WavLMForXVector.from_pretrained(model_id).to(device).eval()

    def embed(path) -> np.ndarray:
        wav, _ = librosa.load(str(path), sr=16000, mono=True)  # model expects 16kHz mono
        inputs = extractor(wav, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            e = model(**inputs).embeddings[0].detach().cpu().numpy().astype(np.float64)
        return e / np.linalg.norm(e)

    embed.model_id = model_id
    return embed


def make_titanet_embedder(model_id: str = "titanet_large"):
    """TitaNet speaker embedding (NVIDIA NeMo). Returns embed(path) -> L2-normalised vector.
    Needs: pip install -q "nemo_toolkit[asr]"  -- a large, slow install; do it in its own fresh
    runtime, separate from the pyannote/WavLM one, to avoid the dependency clashes you already hit once.
    model_id can also be the HF alias "nvidia/speakerverification_en_titanet_large"."""
    import os
    import tempfile

    import librosa
    import nemo.collections.asr as nemo_asr
    import soundfile as sf

    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_id)
    speaker_model.eval()

    def embed(path) -> np.ndarray:
        # TitaNet expects 16 kHz mono: convert every file first so stereo / 44.1 kHz phone recordings can't break it
        wav, _ = librosa.load(str(path), sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, wav, 16000)
            e = speaker_model.get_embedding(tmp_path)
        finally:
            os.remove(tmp_path)
        e = np.asarray(e.detach().cpu() if hasattr(e, "detach") else e).flatten().astype(np.float64)
        return e / np.linalg.norm(e)

    embed.model_id = model_id
    return embed


# --------------------------------------------------------------------------- model -> folder tag
# Each model's enrolment store, embedding cache and results live in their own subfolder, so
# testing WavLM/TitaNet can never overwrite your existing wespeaker JSON/CSVs. The original
# model keeps its original flat layout (tag "") so nothing about your current setup changes.
MODEL_TAGS = {
    MODEL_ID: "",
    "microsoft/wavlm-base-plus-sv": "wavlm",
    "microsoft/wavlm-base-sv": "wavlm",
    "titanet_large": "titanet",
    "nvidia/speakerverification_en_titanet_large": "titanet",
}


def _tag_for(model_id: str) -> str:
    if model_id in MODEL_TAGS:
        return MODEL_TAGS[model_id]
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")[:40]  # unknown model -> safe folder name


def compare_models(root, model_ids=None, filename: str = "enrollment_length_comparison.csv",
                    enrollment: str = "mean_all") -> pd.DataFrame:
    """Line up one results CSV per model (as written by run_experiment) for side-by-side comparison.
    Run each model at least once first. Pass model_ids to restrict/order which models to include."""
    root = Path(root)
    rows, seen_tags = [], set()
    for mid in (model_ids or list(MODEL_TAGS)):
        tag = _tag_for(mid)
        if tag in seen_tags:          # two aliases (e.g. wavlm-base-sv / -plus-sv) share one folder -> one row
            continue
        seen_tags.add(tag)
        p = (root / tag if tag else root) / "results" / filename
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(0, "model", mid)
        rows.append(df[df.enrollment == enrollment] if "enrollment" in df.columns else df)
    if not rows:
        raise FileNotFoundError(f"No {filename} found yet for any of {model_ids or list(MODEL_TAGS)} under {root}. "
                                "Run each model with run_experiment() first.")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- enrolment store
def load_store(path, model: str = MODEL_ID) -> dict:
    p = Path(path)
    if p.exists():
        store = json.loads(p.read_text())
        if store.get("model") != model:
            raise ValueError(
                f"{p.name} was built with {store.get('model')}, but you are using {model}. "
                "Embeddings from different models are not comparable -- use a new file.")
        return store
    return {"schema": 1, "model": model, "created": _now(), "updated": _now(),
            "threshold": None, "speakers": {}}


def save_store(store: dict, path) -> None:
    store["updated"] = _now()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(store))


def enroll_folder(store: dict, embed, enroll_dir) -> list:
    """Embed every NEW file named Name_<N>s.wav and add it to the store. Idempotent:
    files already in the store are skipped, so adding voices = drop files in + re-run."""
    have = {c["file"] for s in store["speakers"].values() for c in s["clips"]}
    added = []
    for p in sorted(Path(enroll_dir).glob("*.wav")):
        m = ENROLL_STEM_RE.match(DUP_SUFFIX_RE.sub("", p.stem))
        if not m:
            warnings.warn(f"Skipped {p.name}: enrolment files must look like Name_15s.wav (letters only in Name)")
            continue
        if p.name in have:
            continue
        name, label = _canon(m["name"]), f"{int(m['dur'])}s"
        store["speakers"].setdefault(name, {"clips": []})["clips"].append(
            {"label": label, "file": p.name, "embedding": np.asarray(embed(p)).tolist(), "added": _now()})
        added.append((name, label))
    return added


def remove_speaker(store: dict, name: str) -> None:
    store["speakers"].pop(_canon(name))
    store["threshold"] = None  # impostor set changed -> must recalibrate


def profile(store: dict, name: str, condition: str):
    """Unit-norm mean embedding of a speaker's clips. condition = '15s' etc. or 'mean_all'."""
    embs = [np.asarray(c["embedding"]) for c in store["speakers"][name]["clips"]
            if condition == "mean_all" or c["label"] == condition]
    if not embs:
        return None
    m = np.mean(embs, axis=0)
    return m / np.linalg.norm(m)


def enrollment_conditions(store: dict) -> list:
    """Duration labels that EVERY speaker has (so conditions stay comparable) + 'mean_all'."""
    label_sets = [{c["label"] for c in s["clips"]} for s in store["speakers"].values()]
    common = set.intersection(*label_sets) if label_sets else set()
    dropped = set.union(*label_sets) - common if label_sets else set()
    if dropped:
        warnings.warn(f"Enrolment lengths {sorted(dropped)} are not available for every speaker and are "
                      "excluded from the length comparison. Record the same lengths for each voice.")
    return sorted(common, key=lambda s: int(s[:-1])) + ["mean_all"]


def profile_similarity(store: dict) -> pd.DataFrame:
    """Cosine similarity between speakers' mean profiles -- a quick 'who sounds like whom' check."""
    names = sorted(store["speakers"])
    P = np.stack([profile(store, n, "mean_all") for n in names])
    df = pd.DataFrame(np.round(P @ P.T, 3), index=names, columns=names)
    return df.reset_index().rename(columns={"index": "speaker"})


# --------------------------------------------------------------------------- embedding cache
class EmbeddingCache:
    """Wraps embed() so each test clip is embedded once, ever (persisted as JSON)."""

    def __init__(self, embed, path=None, model: str = MODEL_ID):
        self.embed, self.path, self.model, self.data = embed, path, model, {}
        if path and Path(path).exists():
            blob = json.loads(Path(path).read_text())
            if blob.get("model") == model:
                self.data = blob["embeddings"]

    def __call__(self, p: Path) -> np.ndarray:
        key = f"{p.parent.name}/{p.name}|{p.stat().st_size}"
        if key not in self.data:
            self.data[key] = np.asarray(self.embed(p)).tolist()
        return np.asarray(self.data[key])

    def save(self):
        if self.path:
            Path(self.path).write_text(json.dumps({"model": self.model, "embeddings": self.data}))


# --------------------------------------------------------------------------- test clips
def discover_tests(test_dir, speakers) -> list:
    """Find test clips and derive ground truth from the filename.
    'U<N>' / 'Unknown<N>' -> UNKNOWN; otherwise the name MUST be an enrolled speaker
    (so a typo or a not-yet-enrolled voice fails loudly instead of silently becoming UNKNOWN)."""
    lookup = {s.lower(): s for s in speakers}
    clips, seen = [], set()
    for p in sorted(Path(test_dir).rglob("*.wav")):
        stem = DUP_SUFFIX_RE.sub("", p.stem)
        if ENROLL_STEM_RE.match(stem):
            continue
        m = TEST_STEM_RE.match(stem)
        if not m:
            raise ValueError(f"Unrecognised test filename '{p.name}'. Expected Name<N>.wav or U<N>.wav "
                             "(sliced segments belong in their own folder, not in tests/).")
        if UNKNOWN_STEM_RE.match(stem):
            gt = UNKNOWN
        else:
            gt = lookup.get(m["name"].lower())
            if gt is None:
                raise ValueError(f"'{p.name}': '{m['name']}' is not an enrolled speaker. Enrol them first, "
                                 "or rename the clip U<N>.wav if they are meant to be an unknown voice.")
        num = int(m["num"])
        if (gt, num) in seen:
            warnings.warn(f"Duplicate clip id {gt}{num} ('{p.name}') -- Colab '(1)' copy? It is counted twice.")
        seen.add((gt, num))
        clips.append({"clip": p.name, "path": str(p), "ground_truth": gt, "clip_num": num,
                      "split": "dev" if num % 2 == 1 else "eval"})
    if not clips:
        raise FileNotFoundError(f"No test clips found under {test_dir}")
    return clips


# --------------------------------------------------------------------------- scoring
def score_clips(store: dict, embed, clips: list, conditions: list) -> pd.DataFrame:
    """Threshold-free step: cosine similarity of every clip to every speaker, per enrolment condition."""
    speakers = sorted(store["speakers"])
    profs = {c: {s: profile(store, s, c) for s in speakers} for c in conditions}
    rows = []
    for clip in clips:
        e = embed(Path(clip["path"]))
        for cond in conditions:
            row = {k: clip[k] for k in ("clip", "ground_truth", "clip_num", "split")}
            row["enrollment"] = cond
            row.update({f"sim_{s}": round(float(e @ profs[cond][s]), 4) for s in speakers})
            rows.append(row)
    return pd.DataFrame(rows)


def add_decisions(scores: pd.DataFrame, threshold: float, run_id: str) -> pd.DataFrame:
    df = scores.copy()
    sim_cols = [c for c in df.columns if c.startswith("sim_")]
    names = [c[4:] for c in sim_cols]
    S = df[sim_cols].to_numpy()
    order = np.argsort(-S, axis=1)
    rows = np.arange(len(S))
    top = S[rows, order[:, 0]]
    second = S[rows, order[:, 1]] if S.shape[1] > 1 else np.full(len(S), np.nan)
    df["top_speaker"] = [names[i] for i in order[:, 0]]
    df["top_score"] = top
    df["second_score"] = second
    df["margin"] = np.round(top - second, 4)
    df["threshold"] = round(float(threshold), 4)
    df["predicted"] = np.where(top >= threshold, df["top_speaker"], UNKNOWN)
    df["correct"] = df["predicted"] == df["ground_truth"]
    df["closed_set_correct"] = [
        (t == g) if g != UNKNOWN else None for t, g in zip(df["top_speaker"], df["ground_truth"])]

    def ftype(g, p):
        if g == p:
            return ""
        if g == UNKNOWN:
            return "false_accept"          # stranger accepted as an enrolled speaker
        if p == UNKNOWN:
            return "false_reject"          # enrolled speaker rejected as unknown
        return "wrong_speaker"             # accepted, but as the wrong person

    df["failure_type"] = [ftype(g, p) for g, p in zip(df["ground_truth"], df["predicted"])]
    df.insert(0, "run_id", run_id)
    return df


# --------------------------------------------------------------------------- threshold
def calibrate_threshold(scores: pd.DataFrame, condition: str = "mean_all", split: str = "dev") -> dict:
    """Pick ONE threshold from dev-split verification trials (every clip vs every profile).
    target = clip vs its own speaker; non-target = clip vs any other speaker, incl. unknowns.
    If the classes separate, use the midpoint of the gap; otherwise the equal-error-rate point."""
    d = scores[(scores.enrollment == condition) & (scores.split == split)]
    sim_cols = [c for c in scores.columns if c.startswith("sim_")]
    tgt, non = [], []
    for _, r in d.iterrows():
        for c in sim_cols:
            (tgt if c == f"sim_{r.ground_truth}" else non).append(r[c])
    tgt, non = np.array(tgt, float), np.array(non, float)
    if len(tgt) == 0 or len(non) == 0:
        raise ValueError("Need dev-split clips (odd-numbered) for both known and other/unknown speakers.")
    info = {"condition": condition, "split": split, "n_target": len(tgt), "n_nontarget": len(non),
            "max_nontarget": round(float(non.max()), 4), "min_target": round(float(tgt.min()), 4),
            "calibrated_at": _now()}
    if non.max() < tgt.min():
        info.update(value=round(float((non.max() + tgt.min()) / 2), 4), method="gap-midpoint", eer=0.0)
    else:
        cand = np.unique(np.concatenate([tgt, non]))
        frr = np.array([(tgt < t).mean() for t in cand])
        far = np.array([(non >= t).mean() for t in cand])
        gap = np.abs(far - frr)
        best = np.where(gap == gap.min())[0]
        info.update(value=round(float(cand[best].mean()), 4), method="EER",
                    eer=round(float((far[best] + frr[best]).mean() / 2), 4))
    return info


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (Brown, Cai & DasGupta 2001)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def enrollment_comparison(res: pd.DataFrame) -> pd.DataFrame:
    sim_cols = [c for c in res.columns if c.startswith("sim_")]
    out = []
    for cond, g in res.groupby("enrollment", sort=False):
        known, unk = g[g.ground_truth != UNKNOWN], g[g.ground_truth == UNKNOWN]
        k_ok = int(known.closed_set_correct.astype(bool).sum())
        ke, ue = known[known.split == "eval"], unk[unk.split == "eval"]
        tgt = [r[f"sim_{r.ground_truth}"] for _, r in known.iterrows()]
        non = [r[c] for _, r in g.iterrows() for c in sim_cols if c != f"sim_{r.ground_truth}"]
        lo, hi = wilson(k_ok, len(known))
        out.append({
            "run_id": g.run_id.iloc[0], "enrollment": cond, "threshold": g.threshold.iloc[0],
            "n_known": len(known), "closed_set_acc": round(k_ok / len(known), 3) if len(known) else np.nan,
            "closed_set_ci_lo": lo, "closed_set_ci_hi": hi,
            "n_known_eval": len(ke),
            "known_id_acc_eval": round(float(ke.correct.mean()), 3) if len(ke) else np.nan,
            "n_unknown_eval": len(ue),
            "unknown_reject_eval": round(float(ue.correct.mean()), 3) if len(ue) else np.nan,
            "mean_target_sim": round(float(np.mean(tgt)), 3) if tgt else np.nan,
            "mean_nontarget_sim": round(float(np.mean(non)), 3) if non else np.nan,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- outputs
def write_outputs(frames: dict, results_dir, run_id: str, append_frames: dict) -> None:
    results = Path(results_dir)
    arch = results / "archive" / run_id
    arch.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(results / f"{name}.csv", index=False)
        df.to_csv(arch / f"{name}.csv", index=False)
    for name, df in append_frames.items():  # e.g. runs.csv keeps growing
        p = results / f"{name}.csv"
        df.to_csv(p, mode="a", header=not p.exists(), index=False)


def df_to_values(df: pd.DataFrame) -> list:
    """DataFrame -> 2-D list of native Python types (what the Sheets API wants). NaN/None -> ''."""
    body = json.loads(df.to_json(orient="values", double_precision=6))
    return [list(map(str, df.columns))] + [["" if v is None else v for v in row] for row in body]


def sync_to_sheets(replace: dict, append: dict, sheet_name: str, gc=None):
    """Push results to a Google Sheet. `replace` tabs are cleared and rewritten (charts pointing at
    them keep working); `append` tabs only gain new rows. Creates the sheet/tabs if missing."""
    import gspread

    if gc is None:  # Colab: one-time Google sign-in popup per runtime
        from google.auth import default
        from google.colab import auth
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
    try:
        sh = gc.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(sheet_name)
        print(f"Created Google Sheet '{sheet_name}'")

    def get_ws(tab, nrows, ncols):
        try:
            return sh.worksheet(tab)
        except gspread.WorksheetNotFound:
            return sh.add_worksheet(title=tab, rows=max(nrows + 20, 100), cols=max(ncols + 2, 10))

    for tab, df in replace.items():
        vals = df_to_values(df)
        ws = get_ws(tab, len(vals), len(vals[0]))
        ws.clear()
        ws.resize(rows=max(len(vals) + 20, 100), cols=max(len(vals[0]) + 2, 10))
        ws.update(range_name="A1", values=vals)          # named args work in gspread v5 and v6
    for tab, df in append.items():
        vals = df_to_values(df)
        ws = get_ws(tab, len(vals), len(vals[0]))
        if ws.get_all_values():
            ws.append_rows(vals[1:])
        else:
            ws.update(range_name="A1", values=vals)
    print(f"Google Sheet updated: {sh.url}")
    return sh


# --------------------------------------------------------------------------- orchestration
def run_experiment(embed, root, sheet_name=None, threshold=None, recalibrate=False, gc=None):
    """Enrol any new voices -> score all test clips -> calibrate/apply ONE threshold ->
    write CSVs to Drive -> (optionally) refresh the Google Sheet.

    The audio you record (audio/enrollment, audio/tests) is shared across every model you try.
    Each model's embeddings, threshold and result CSVs live in their own subfolder (see MODEL_TAGS),
    so running WavLM or TitaNet here never touches your existing wespeaker files -- call this the
    same way for any of make_pyannote_embedder / make_wavlm_embedder / make_titanet_embedder, and
    use compare_models(root) afterwards to line their results up side by side."""
    root = Path(root)
    model = getattr(embed, "model_id", MODEL_ID)
    tag = _tag_for(model)
    base = (root / tag) if tag else root          # "" (wespeaker) keeps the original flat layout
    base.mkdir(parents=True, exist_ok=True)
    store_path = base / "enrolled_speakers.json"
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")

    store = load_store(store_path, model)
    added = enroll_folder(store, embed, root / "audio" / "enrollment")
    if not store["speakers"]:
        raise FileNotFoundError(f"No enrolment files found in {root / 'audio' / 'enrollment'}")
    print(f"Model: {model}" + (f"  (results under {tag}/)" if tag else ""))
    print("Newly enrolled:", added if added else "nothing (all enrolment files already in store)")
    print("Speakers:", {n: sorted(c["label"] for c in s["clips"]) for n, s in store["speakers"].items()})

    cache = EmbeddingCache(embed, base / "test_embeddings_cache.json", model)
    clips = discover_tests(root / "audio" / "tests", store["speakers"].keys())
    conds = enrollment_conditions(store)
    scores = score_clips(store, cache, clips, conds)
    cache.save()

    if threshold is not None:
        thr = {"value": float(threshold), "method": "manual", "calibrated_at": _now()}
    elif store.get("threshold") and not added and not recalibrate:
        thr = store["threshold"]
        print(f"Using stored threshold {thr['value']} ({thr['method']}, {thr['calibrated_at']})")
    else:
        thr = calibrate_threshold(scores)
        print(f"Calibrated threshold on dev clips: {thr['value']} ({thr['method']}; "
              f"max non-target {thr['max_nontarget']}, min target {thr['min_target']})")
    store["threshold"] = thr
    save_store(store, store_path)

    res = add_decisions(scores, thr["value"], run_id)
    unk = res[res.ground_truth == UNKNOWN]
    frames = {
        "identification_results": res,
        "enrollment_length_comparison": enrollment_comparison(res),
        "failure_log": res[~res.correct],
        "unknown_scores": unk,
        "profile_similarity": profile_similarity(store),
    }
    runs = pd.DataFrame([{
        "run_id": run_id, "timestamp": _now(), "model": model, "threshold": thr["value"],
        "threshold_method": thr["method"], "n_speakers": len(store["speakers"]),
        "speakers": ", ".join(sorted(store["speakers"])), "n_test_clips": len(clips),
        "n_unknown_clips": int((pd.Series([c["ground_truth"] for c in clips]) == UNKNOWN).sum()),
        "newly_enrolled": ", ".join(f"{n}@{l}" for n, l in added)}])
    write_outputs(frames, base / "results", run_id, {})                 # per-model CSVs + archive
    write_outputs({}, root / "results", run_id, {"runs": runs})         # one shared cross-model run log
    print(f"CSVs written to {base / 'results'}  (run {run_id})")

    if sheet_name:
        prefix = f"{tag}_" if tag else ""
        sync_to_sheets({f"{prefix}{k}": v for k, v in frames.items()}, {"runs": runs}, sheet_name, gc=gc)

    print("\n" + frames["enrollment_length_comparison"].to_string(index=False))
    return frames
