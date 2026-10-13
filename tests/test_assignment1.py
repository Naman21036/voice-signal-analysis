import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import assignment1 as a  # noqa: E402
import evaluation as ev  # noqa: E402

FS = 16000
CFG = a.Config()
KEEP = a.Config(trim_ambiguous=False)  # conservative mode: ambiguous regions preserved


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


# ---- loading / metadata
def test_load_metadata_duration(tmp_path):
    write(tmp_path / "bit.wav", synth())
    data, fs = a.load_audio(tmp_path / "bit.wav")
    meta = a.get_metadata(tmp_path / "bit.wav", data, fs)
    assert fs == FS and meta["num_samples"] == data.shape[0]
    assert meta["duration_s"] == pytest.approx(data.shape[0] / FS)


def test_mono_and_stereo(tmp_path):
    write(tmp_path / "m.wav", synth())
    write(tmp_path / "s.wav", synth(True))
    assert a.load_audio(tmp_path / "m.wav")[0].shape[1] == 1
    s, _ = a.load_audio(tmp_path / "s.wav")
    assert s.shape[1] == 2 and a.to_mono(s).ndim == 1


def test_invalid_input(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(Exception):
        a.load_audio(bad)
    with pytest.raises(FileNotFoundError):
        a.discover_recordings(tmp_path / "missing")


def test_manifest_identity_and_missing_file(tmp_path):
    write(tmp_path / "x.wav", synth())
    write(tmp_path / "y.wav", synth())
    man = {"recordings": {"bit": {"file": "x.wav", "declared_utterance": "BIT"},
                          "name": {"file": "y.wav", "declared_utterance": "Naman Gupta"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    found = a.discover_recordings(tmp_path)
    assert found["bit"]["declared_utterance"] == "BIT" and found["bit"]["identity_source"] == "manifest.json"
    man["recordings"]["bit"]["file"] = "nope.wav"
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(FileNotFoundError):
        a.discover_recordings(tmp_path)


# ---- detection
def test_silence_detection_known_intervals():
    det = a.detect_silence(synth(), FS, CFG)
    assert [r["type"] for r in det["silence"]] == ["leading", "internal", "trailing"]
    lead, internal, trail = det["silence"]
    assert lead["end_s"] == pytest.approx(0.5, abs=0.05)
    assert internal["start_s"] == pytest.approx(1.0, abs=0.05) and internal["end_s"] == pytest.approx(2.0, abs=0.05)
    assert trail["start_s"] == pytest.approx(2.5, abs=0.05)
    assert len(det["speech"]) == 2 and not det["ambiguous"]


def test_hysteresis_dip_between_thresholds_does_not_split_speech():
    """A dip that stays above the LOW threshold must not split speech even though it is below the HIGH one."""
    rng = np.random.default_rng(1)
    noise = lambda d: 1e-3 * rng.standard_normal(int(d * FS))  # noqa: E731
    x = np.concatenate([noise(0.5), tone(0.4, 0.3), tone(0.3, 0.02), tone(0.4, 0.3), noise(0.5)])
    det = a.detect_silence(x[:, None], FS, CFG)
    assert len(det["speech"]) == 1 and [r["type"] for r in det["silence"]] == ["leading", "trailing"]


def test_weak_activity_is_ambiguous_not_deleted():
    rng = np.random.default_rng(2)
    n = lambda: 1e-3 * rng.standard_normal(FS)  # noqa: E731
    x = np.concatenate([n(), tone(0.5, 0.01), n(), tone(0.5, 0.3), n()])[:, None]
    det = a.detect_silence(x, FS, CFG)
    assert det["ambiguous"], "weak tone should be flagged as ambiguous"
    out, _ = a.trim_silence(x, FS, det, KEEP)
    # leading 1 s trimmed to 0.1 s padding; weak tone and the 1 s gap next to it are kept (2.2 s total)
    assert out.shape[0] == pytest.approx(2.2 * FS, abs=0.1 * FS)
    assert np.max(np.abs(out[: int(0.7 * FS)])) > 0.005  # weak tone still present


def test_low_amplitude_speech_preserved():
    x, truth = ev.build([(0.5, 1.1, 0.1), (1.6, 2.4, 0.003)], 3.0, 0.0003, 3)
    m = ev.evaluate_one(x, truth, a.detect_silence(x, ev.FS, CFG), CFG)
    assert m["speech_preserved_fraction"] >= 0.99 and m["recall"] >= 0.95


def test_stereo_opposite_polarity_not_cancelled():
    x, _ = ev.build([(0.5, 1.1, 0.1), (1.6, 2.4, 0.1)], 3.0, 1e-3, 9, stereo="opposite")
    assert a.detect_silence(x, ev.FS, CFG)["speech"]
    assert not a.detect_silence_baseline(x, ev.FS, CFG)["speech"]  # documents the baseline failure


def test_frame_energy_pools_channels():
    x = tone(0.2)[:, None]
    both = np.concatenate([x, 0.5 * x], axis=1)
    db1, db2 = a.frame_rms_db(x, 400, 160), a.frame_rms_db(both, 400, 160)
    assert db2.shape == db1.shape and db2[0] < db1[0]


def test_low_snr_declares_no_silence():
    x, _ = ev.build([(0.5, 1.1, 0.1), (1.6, 2.4, 0.1)], 3.0, 0.0316, 2)  # ~10 dB SNR
    det = a.detect_silence(x, ev.FS, CFG)
    assert det["flags"] and not det["silence"]
    out, _ = a.trim_silence(x, ev.FS, det, CFG)
    assert out.shape[0] == x.shape[0]


@pytest.mark.parametrize("x", [np.zeros((FS, 1)), np.zeros((0, 1)), np.array([[0.1], [0.2], [0.1]]), np.full((FS, 1), 0.3)])
@pytest.mark.parametrize("det_fn", [a.detect_silence, a.detect_silence_baseline])
def test_degenerate_inputs_do_not_crash(x, det_fn):
    det = det_fn(x, FS, CFG)
    out, _ = a.trim_silence(x, FS, det, CFG)
    assert out.shape[0] <= x.shape[0] and np.all(np.isfinite(out))
    assert a.compute_spectrogram(x[:, 0], FS, CFG)[2].size > 0


def test_all_silent_is_one_silent_region():
    det = a.detect_silence(np.zeros((FS, 1)), FS, CFG)
    assert det["silence"][0]["type"] == "entire recording" and not det["active"]


# ---- trimming
def test_trim_does_not_modify_original_and_shortens():
    data = synth(True)
    before = data.copy()
    det = a.detect_silence(data, FS, CFG)
    out, info = a.trim_silence(data, FS, det, CFG)
    assert np.array_equal(data, before)
    assert out.shape[1] == 2 and out.shape[0] < data.shape[0]
    assert len(info["shortened_gaps"]) == 1  # 1 s confident-silence gap between speech regions
    assert np.max(np.abs(out)) > 0.4
    g = info["shortened_gaps"][0]
    assert g["removed_s"] == pytest.approx(g["duration_s"] - CFG.kept_gap_ms / 1000, abs=1e-3)


def test_joins_have_no_discontinuity():
    data = synth()
    det = a.detect_silence(data, FS, CFG)
    out, info = a.trim_silence(data, FS, det, CFG)
    j = a.check_joins(out, info["join_samples"])
    assert j["checked"] == 1 and j["ok"]


def test_gap_next_to_ambiguous_is_preserved():
    det = {"active": [{"start_s": 0.5, "end_s": 0.8, "duration_s": 0.3, "type": "ambiguous"},
                      {"start_s": 1.8, "end_s": 2.3, "duration_s": 0.5, "type": "speech"}], "speech": []}
    data = np.random.default_rng(0).standard_normal((3 * FS, 1)) * 0.01
    _, info = a.trim_silence(data, FS, det, KEEP)
    assert not info["shortened_gaps"] and info["preserved_gaps"][0]["reason"].startswith("neighbour")


def test_trim_boundaries_clamped():
    det = {"active": [{"start_s": 0.0, "end_s": 1.0, "duration_s": 1.0, "type": "speech"}], "speech": []}
    out, _ = a.trim_silence(np.ones((FS, 1)) * 0.1, FS, det, CFG)
    assert out.shape[0] == FS  # padding cannot extend past the file


def test_duration_reduction():
    d = a.duration_reduction(1000, 750, 100)
    assert d["removed_duration_s"] == pytest.approx(2.5) and d["percent_removed"] == pytest.approx(25.0)


# ---- spectrogram
def test_spectrogram_limits_and_short_signal():
    f, t, s = a.compute_spectrogram(tone(0.5, f=1000), FS, CFG)
    assert f[-1] <= FS / 2 and f[np.argmax(s.mean(axis=1))] == pytest.approx(1000, abs=FS / CFG.spec_nfft)
    assert a.compute_spectrogram(np.array([0.1]), 8000, CFG)[2].size > 0


# ---- synthetic evaluation
def test_synthetic_evaluation_speech_preserved_and_baseline_failures_fixed():
    E = ev.run_synthetic_evaluation(CFG)
    sc = E["scenarios"]
    for name, s in sc.items():
        assert s["improved"]["speech_preserved_fraction"] >= 0.98, name
    little = sc["little_silence_(speech_fills_file)"]
    assert little["improved"]["f1"] > little["baseline"]["f1"]
    st = sc["stereo_opposite_polarity"]
    assert st["improved"]["f1"] > 0.9 > st["baseline"]["f1"]
    assert sc["clean"]["improved"]["f1"] >= 0.95 and abs(sc["clean"]["improved"]["start_error_ms"]) < 30


# ---- pipeline / integrity / reruns
def test_baseline_reproduces_frozen_results_on_real_audio():
    frozen = ROOT / "outputs" / "baseline_results.json"
    if not frozen.exists() or not (ROOT / "audio" / "Naman_BIT.wav").exists():
        pytest.skip("frozen baseline or real audio not present")
    b = json.loads(frozen.read_text())["recordings"]
    for key, fn in (("bit", "Naman_BIT.wav"), ("name", "Naman_Name.wav")):
        d, fs = a.load_audio(ROOT / "audio" / fn)
        p, _ = a.trim_silence(d, fs, a.detect_silence_baseline(d, fs, CFG), CFG)
        assert p.shape[0] == b[key]["validation"]["num_samples"]


def test_pipeline_outputs_integrity_serialization_and_rerun(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    write(audio / "Naman_BIT.wav", synth(True))
    write(audio / "Naman_Name.wav", synth())
    orig = (audio / "Naman_BIT.wav").read_bytes()
    out = tmp_path / "out"
    for _ in range(2):  # rerun is safe
        res = a.run(audio, out, CFG)
    assert (audio / "Naman_BIT.wav").read_bytes() == orig
    for r in res["recordings"].values():
        for p in r["plots"].values():
            assert (tmp_path / p).stat().st_size > 1000
        proc, fs = sf.read(str(tmp_path / r["processed_file"]), always_2d=True)
        assert fs == FS and proc.shape[0] == r["validation"]["num_samples"] and r["validation"]["passed"]
        assert r["reduction"]["processed_duration_s"] == pytest.approx(proc.shape[0] / fs)
        assert r["validation"]["checks"]["original_file_unchanged"]
        assert r["declared_utterance"] and "NOT independently verified" in r["content_verification"]
    saved = json.loads((out / "analysis_results.json").read_text())
    assert saved["recordings"]["bit"]["metadata"]["sampling_frequency_hz"] == FS
    assert "synthetic_evaluation" in saved
    assert "## 8. Controlled Evaluation" in (out / "analysis_report.md").read_text(encoding="utf-8")


def test_validate_wav_detects_problems(tmp_path):
    x = np.full((100, 1), 0.2)
    p = tmp_path / "v.wav"
    sf.write(str(p), x, FS, subtype="PCM_16")
    assert a.validate_wav(p, FS, 1, x, 0.2)["passed"]
    assert not a.validate_wav(p, 8000, 1, x, 0.2)["passed"]  # wrong sampling frequency
    sf.write(str(p), np.zeros((100, 1)), FS, subtype="PCM_16")
    assert not a.validate_wav(p, FS, 1, np.zeros((100, 1)), 0.2)["passed"]  # all zero


def test_real_recordings_outputs_consistent():
    res_path = ROOT / "outputs" / "analysis_results.json"
    if not res_path.exists():
        pytest.skip("run python assignment1.py first")
    res = json.loads(res_path.read_text())
    for r in res["recordings"].values():
        info = sf.info(str(ROOT / "audio" / r["metadata"]["source_file"]))
        assert info.samplerate == r["metadata"]["sampling_frequency_hz"] and info.frames == r["metadata"]["num_samples"]
        proc = sf.info(str(ROOT / r["processed_file"]))
        assert proc.frames == r["validation"]["num_samples"] and proc.samplerate == info.samplerate
        assert r["validation"]["passed"]


# ---- periodicity gate, region evidence, joins, verification (second round)
import evaluation_events as ee  # noqa: E402
import regions as rg  # noqa: E402
import verify_content as vc  # noqa: E402


def test_periodicity_high_for_voiced_low_for_noise():
    rng = np.random.default_rng(5)
    voiced = ev.speech_like(0.5, 0.1, rng)[:, None]
    noisy = (0.1 * rng.standard_normal(int(0.5 * ev.FS)))[:, None]
    pv = a.frame_periodicity(voiced, ev.FS, CFG)
    pn = a.frame_periodicity(noisy, ev.FS, CFG)
    assert np.median(pv) > 0.8 and np.median(pn) < 0.4


def test_periodicity_handles_short_and_empty():
    assert a.frame_periodicity(np.zeros((0, 1)), FS, CFG).size == 0
    assert np.all(np.isfinite(a.frame_periodicity(np.ones((50, 1)), FS, CFG)))


def test_gate_turns_loud_noise_burst_into_ambiguous_but_preserves_it():
    sc = ee.event_scenarios()["loud_white_noise_burst"]
    from dataclasses import replace
    off = a.detect_silence(sc["x"], ev.FS, replace(CFG, use_periodicity_gate=False))
    on = a.detect_silence(sc["x"], ev.FS, replace(CFG, use_periodicity_gate=True))
    dur = lambda d: sum(r["duration_s"] for r in d["speech"])  # noqa: E731
    assert dur(on) < dur(off)
    out, info = a.trim_silence(sc["x"], ev.FS, on, KEEP)
    assert info["kept_segments"][0][0] <= 0.1 + 1e-6  # event at 0.1-0.5 s is kept, not trimmed


def test_feature_evaluation_reports_expected_structure_and_benefit():
    fe = ee.run_feature_evaluation(CFG)
    assert set(fe["feature_auc"]) >= {"periodicity", "zcr", "spectral_flux"}
    s = fe["scenarios"]["loud_click_train"]
    assert s["with_periodicity_gate"]["speech_false_positive_rate"] < s["energy_only"]["speech_false_positive_rate"]
    for v in fe["scenarios"].values():
        assert v["with_periodicity_gate"]["active_false_negative_rate"] <= 0.02  # gating never causes data loss


def test_synthetic_join_validation_detects_bad_join_and_fade_fixes_it():
    res, audio = ee.join_validation(CFG)
    assert res["natural_case"]["joins_created"] == 1 and res["natural_case"]["check"]["ok"]
    assert not res["stress_case"]["plain_cut_fade_0ms"]["ok"]
    assert res["stress_case"]["configured_fade"]["ok"]
    sig, joins = audio["configured_fade"]
    assert sig.shape[0] > joins[0] > 0


def test_real_style_pipeline_reports_join_not_applicable(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    x = np.concatenate([tone(0.3, 0.001), tone(0.5, 0.4), tone(0.3, 0.001)])[:, None]  # no internal gap
    write(audio / "Naman_BIT.wav", x)
    write(audio / "Naman_Name.wav", x)
    res = a.run(audio, tmp_path / "out", CFG)
    j = res["recordings"]["bit"]["validation"]["joins"]
    assert j["joins_created"] == 0 and j["applicability"].startswith("not applicable")
    assert "joins_without_discontinuity" not in res["recordings"]["bit"]["validation"]["checks"]
    assert "synthetic_join_validation" in res and res["listening_materials"]["listening_performed"] is False
    assert (tmp_path / "out" / "listening_checklist.md").exists()


def test_region_features_returns_evidence_without_claiming_speech():
    rng = np.random.default_rng(3)
    x = np.concatenate([1e-3 * rng.standard_normal(FS), 0.01 * rng.standard_normal(FS // 2), 1e-3 * rng.standard_normal(FS),
                        tone(0.5, 0.3), 1e-3 * rng.standard_normal(FS)])[:, None]
    det = a.detect_silence(x, FS, CFG)
    assert det["ambiguous"]
    e = rg.region_features(x, FS, det, det["ambiguous"][0], CFG)
    assert e["decision"].startswith(("preserved", "removable")) and "speech" in e["automated_label"] + "speech"
    assert e["region_features"]["rms_db_median"] < e["reference_confident_speech"]["rms_db_median"]


def test_verify_content_comparison_and_unverified_without_engines(monkeypatch, tmp_path):
    assert vc.compare("BIT", "B-I-T.")["consistent_with_expected"]
    assert not vc.compare("Naman Gupta", "The city ID")["consistent_with_expected"]
    write(tmp_path / "Naman_BIT.wav", synth())
    write(tmp_path / "Naman_Name.wav", synth())
    monkeypatch.setattr(vc, "run_whisper", lambda *a_, **k: {"engine": "x", "available": False, "reason": "test"})
    monkeypatch.setattr(vc, "run_windows_sapi", lambda *a_, **k: {"engine": "y", "available": False, "reason": "test"})
    res = vc.verify(tmp_path)
    assert all(r["status"] == "unverified" for r in res["recordings"].values())
    assert res["manual_procedure"]


def test_real_outputs_paths_exist_and_durations_match_saved_audio():
    res_path = ROOT / "outputs" / "analysis_results.json"
    if not res_path.exists():
        pytest.skip("run python assignment1.py first")
    res = json.loads(res_path.read_text())
    for r in res["recordings"].values():
        for p in list(r["plots"].values()) + [r["processed_file"]] + [e["focus_plot"] for e in r["region_analysis"]]:
            assert (ROOT / p).is_file(), p
        saved = sf.info(str(ROOT / r["processed_file"]))
        assert saved.frames / saved.samplerate == pytest.approx(r["reduction"]["processed_duration_s"])
    assert (ROOT / res["synthetic_join_validation"]["plot"]).is_file()
    for k, r in res["recordings"].items():
        h = r["validation"]["original_sha256"]
        assert a.sha256_of(ROOT / "audio" / r["metadata"]["source_file"]) == h


def test_trim_ambiguous_default_removes_ambiguous_but_keep_mode_preserves():
    rng = np.random.default_rng(2)
    n = lambda: 1e-3 * rng.standard_normal(FS)  # noqa: E731
    x = np.concatenate([n(), tone(0.5, 0.01), n(), tone(0.5, 0.3), n()])[:, None]
    det = a.detect_silence(x, FS, CFG)
    kept, _ = a.trim_silence(x, FS, det, KEEP)
    removed, info = a.trim_silence(x, FS, det, CFG)
    assert CFG.trim_ambiguous and removed.shape[0] < kept.shape[0]
    assert info["kept_segments"][0][0] > 2.0  # starts at the confident speech minus padding
