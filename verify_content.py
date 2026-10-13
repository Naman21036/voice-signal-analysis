#!/usr/bin/env python3
"""Optional offline speech-recognition cross-check of the declared utterances.

Usage:  python verify_content.py [--audio-dir audio] [--output-dir outputs] [--whisper-model PATH]

Writes outputs/content_verification.json. Audio never leaves the machine:
  * faster-whisper (if installed) with a model that is already present locally (no download is attempted);
  * the Windows built-in speech recognizer (System.Speech) when running on Windows.
Recognition output is supporting evidence, not ground truth. If no engine is available the file records
"unverified" together with a manual verification procedure.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import assignment1 as a

MANUAL_PROCEDURE = [
    "Play the original file (not the processed one) with headphones.",
    "Write down exactly what you hear, once, without looking at the expected phrase.",
    "Compare with the declared utterance in audio/manifest.json and record the result in manual_listening.json.",
    "A second listener makes the check more reliable.",
]


def normalise(text: str) -> str:
    """Lower-case letters and digits only, so 'B-I-T.' and 'bit' compare equal."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def compare(expected: str, heard: str) -> dict:
    e, h = normalise(expected), normalise(heard)
    ratio = difflib.SequenceMatcher(None, e, h).ratio() if e and h else 0.0
    return {"expected_normalised": e, "heard_normalised": h, "similarity": round(ratio, 3),
            "consistent_with_expected": bool(h) and ratio >= 0.8}


def prepare_16k_mono(path: Path, tmpdir: Path) -> Path:
    data, fs = a.load_audio(path)
    mono = data.mean(axis=1)
    g = np.gcd(fs, 16000)
    y = resample_poly(mono, 16000 // g, fs // g)
    out = tmpdir / (path.stem + "_16k_mono.wav")
    sf.write(str(out), y, 16000, subtype="PCM_16")
    return out


def run_whisper(wav: Path, model_path: str | None) -> dict:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # not installed or broken
        return {"engine": "faster-whisper", "available": False, "reason": f"import failed: {exc}"}
    if model_path is None:
        cands = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-*whisper*/snapshots/*")))
        model_path = cands[0] if cands else None
    if not model_path:
        return {"engine": "faster-whisper", "available": False, "reason": "no locally cached model found (no download attempted)"}
    try:
        model = WhisperModel(model_path, device="cpu", compute_type="int8", local_files_only=True)
        audio, _ = sf.read(str(wav), dtype="float32")
        segs, _ = model.transcribe(audio, language="en", beam_size=5, temperature=0.0, condition_on_previous_text=False)
        segs = list(segs)
    except Exception as exc:
        return {"engine": "faster-whisper", "available": False, "reason": f"run failed: {exc}"}
    text = " ".join(s.text.strip() for s in segs).strip()
    model_name = Path(model_path).parts[-3] if "snapshots" in model_path else Path(model_path).name
    return {"engine": "faster-whisper (offline, CPU, int8)", "available": True, "model": model_name, "transcript": text,
            "avg_logprob": [round(float(s.avg_logprob), 3) for s in segs],
            "no_speech_prob": [round(float(s.no_speech_prob), 3) for s in segs],
            "note": "Whisper may write spoken letters as 'B-I-T'; comparison ignores punctuation."}


_PS = r"""
Add-Type -AssemblyName System.Speech
$e = New-Object System.Speech.Recognition.SpeechRecognitionEngine([Globalization.CultureInfo]'en-US')
$e.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
$e.SetInputToWaveFile($args[0])
$out = @()
try { while ($true) { $r = $e.Recognize(); if ($r -eq $null) { break }; $out += @{text=$r.Text; confidence=[double]$r.Confidence} } } catch {}
if ($out.Count -gt 0) { ConvertTo-Json -InputObject $out -Compress }
"""


def run_windows_sapi(wav: Path) -> dict:
    if not sys.platform.startswith("win"):
        return {"engine": "Windows System.Speech", "available": False, "reason": "not running on Windows"}
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", _PS.replace("$args[0]", f"'{wav}'")],
                             capture_output=True, text=True, timeout=120)
        items = json.loads(res.stdout.strip() or "[]")
    except Exception as exc:
        return {"engine": "Windows System.Speech", "available": False, "reason": f"run failed: {exc}"}
    if isinstance(items, dict):
        items = [items]
    return {"engine": "Windows System.Speech dictation (offline, en-US)", "available": True,
            "transcript": " ".join(i["text"] for i in items).strip(),
            "confidence": [round(i["confidence"], 3) for i in items],
            "note": "Desktop dictation engine; weak on isolated short words."}


def verify(audio_dir: Path, whisper_model: str | None = None) -> dict:
    found = a.discover_recordings(audio_dir)
    out = {"basis": "Recognition output is supporting evidence only, not ground truth. Audio stays on this machine.",
           "manual_procedure": MANUAL_PROCEDURE, "recordings": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for key, e in found.items():
            wav = prepare_16k_mono(e["path"], Path(tmp))
            engines = []
            for res in (run_whisper(wav, whisper_model), run_windows_sapi(wav)):
                if res.get("available"):
                    res["comparison"] = compare(e["declared_utterance"], res["transcript"])
                engines.append(res)
            avail = [x for x in engines if x.get("available")]
            agree = [x for x in avail if x["comparison"]["consistent_with_expected"]]
            if not avail:
                status, note = "unverified", "No recognition engine was available; follow the manual procedure."
            elif agree and len(agree) == len(avail):
                status, note = "supported_by_all_engines_run", "All engines run agree with the declared utterance (still not ground truth)."
            elif agree:
                status, note = "partially_supported", (f"{len(agree)} of {len(avail)} engines agree; the others disagreed or "
                                                       "returned low-confidence text. Treat as unverified until listened to.")
            else:
                status, note = "not_supported", "No engine's transcript matched the declared utterance; check the file manually."
            out["recordings"][key] = {"file": e["path"].name, "declared_utterance": e["declared_utterance"],
                                      "status": status, "note": note, "engines": engines,
                                      "confidence": "limited: very short recordings, few engines, no ground-truth transcript"}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio-dir", type=Path, default=a.ROOT / "audio")
    p.add_argument("--output-dir", type=Path, default=a.ROOT / "outputs")
    p.add_argument("--whisper-model", default=None, help="path to a local faster-whisper model directory")
    args = p.parse_args(argv)
    try:
        res = verify(args.audio_dir, args.whisper_model)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "content_verification.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, r in res["recordings"].items():
        heard = "; ".join(f"{e['engine']}: '{e['transcript']}'" for e in r["engines"] if e.get("available")) or "no engine"
        print(f"[{k}] declared '{r['declared_utterance']}' -> {r['status']} ({heard})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
