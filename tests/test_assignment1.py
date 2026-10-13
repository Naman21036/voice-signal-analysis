import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import assignment1 as a  # noqa: E402

FS = 16000
CFG = a.Config()


def tone(dur, amp=0.5, f=300):
    t = np.arange(int(dur * FS)) / FS
    return amp * np.sin(2 * np.pi * f * t)


def synth(stereo=False):
    rng = np.random.default_rng(0)
    noise = lambda d: 1e-4 * rng.standard_normal(int(d * FS))  # noqa: E731
    x = np.concatenate([noise(0.5), tone(0.5), noise(1.0), tone(0.5), noise(0.5)])
    return np.stack([x, x * 0.8], axis=1) if stereo else x[:, None]


def write(path, data):
    sf.write(str(path), data, FS, subtype="PCM_16")


def test_load_metadata_duration(tmp_path):
    write(tmp_path / "bit.wav", synth())
    data, fs = a.load_audio(tmp_path / "bit.wav")
    meta = a.get_metadata(tmp_path / "bit.wav", data, fs)
    assert fs == FS and meta["num_samples"] == data.shape[0]
    assert meta["duration_s"] == pytest.approx(data.shape[0] / FS)


def test_mono_and_stereo(tmp_path):
    write(tmp_path / "m.wav", synth()); write(tmp_path / "s.wav", synth(True))
    assert a.load_audio(tmp_path / "m.wav")[0].shape[1] == 1
    s, _ = a.load_audio(tmp_path / "s.wav")
    assert s.shape[1] == 2 and a.to_mono(s).ndim == 1


def test_silence_detection_known_intervals():
    det = a.detect_silence(a.to_mono(synth()), FS, CFG)
    kinds = [r["type"] for r in det["silence"]]
    assert kinds == ["leading", "internal", "trailing"]
    lead, internal, trail = det["silence"]
    assert lead["end_s"] == pytest.approx(0.5, abs=0.05)
    assert internal["start_s"] == pytest.approx(1.0, abs=0.05) and internal["end_s"] == pytest.approx(2.0, abs=0.05)
    assert trail["start_s"] == pytest.approx(2.5, abs=0.05)


def test_trim_does_not_modify_original_and_shortens():
    data = synth(True)
    before = data.copy()
    det = a.detect_silence(a.to_mono(data), FS, CFG)
    out, info = a.trim_silence(data, FS, det, CFG)
    assert np.array_equal(data, before)
    assert out.shape[1] == 2 and out.shape[0] < data.shape[0]
    assert len(info["shortened_gaps"]) == 1  # the 1 s internal gap exceeds 400 ms
    assert np.max(np.abs(out)) > 0.4  # speech retained


@pytest.mark.parametrize("x", [np.zeros(FS), np.array([]), np.array([0.1, 0.2, 0.1]), np.full(FS, 0.3)])
def test_degenerate_inputs_do_not_crash(x):
    det = a.detect_silence(x, FS, CFG)
    out, _ = a.trim_silence(x[:, None], FS, det, CFG)
    assert out.shape[0] <= max(len(x), 0)
    if x.size > 3:
        a.compute_spectrogram(x, FS, CFG)
    else:
        f, t, s = a.compute_spectrogram(np.array([0.1, 0.2, 0.1]), FS, CFG)
        assert s.size > 0


def test_invalid_input(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(Exception):
        a.load_audio(bad)
    with pytest.raises(FileNotFoundError):
        a.discover_recordings(tmp_path / "missing")


def test_duration_reduction():
    d = a.duration_reduction(1000, 750, 100)
    assert d["removed_duration_s"] == pytest.approx(2.5) and d["percent_removed"] == pytest.approx(25.0)


def test_pipeline_outputs_and_rerun(tmp_path):
    audio = tmp_path / "audio"; audio.mkdir()
    write(audio / "Naman_BIT.wav", synth(True)); write(audio / "Naman_Name.wav", synth())
    orig = (audio / "Naman_BIT.wav").read_bytes()
    out = tmp_path / "out"
    for _ in range(2):  # rerun is safe
        res = a.run(audio, out, CFG)
    assert (audio / "Naman_BIT.wav").read_bytes() == orig
    for k, r in res["recordings"].items():
        for p in r["plots"].values():
            assert (tmp_path / p).stat().st_size > 1000
        proc, fs = sf.read(str(tmp_path / r["processed_file"]), always_2d=True)
        assert fs == FS and proc.shape[0] == r["validation"]["num_samples"] and r["validation"]["passed"]
        assert r["reduction"]["processed_duration_s"] == pytest.approx(proc.shape[0] / fs)
    assert (out / "analysis_results.json").exists() and (out / "analysis_report.md").exists()
