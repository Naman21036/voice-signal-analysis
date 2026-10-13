"""Synthetic evaluation of (a) supplementary speech/non-speech features and (b) edited-join continuity.

Synthetic signals only: they test the algorithms and never replace the real recordings.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

import assignment1 as a
from evaluation import FS, GRID_MS, _mask, build, noise, speech_like


def _unvoiced(duration: float, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Fricative-like sound: high-passed (3 kHz) noise, no periodicity."""
    n = int(duration * FS)
    b, a_ = sps.butter(2, 3000 / (FS / 2), btype="high")
    x = sps.lfilter(b, a_, rng.standard_normal(n))
    r = int(0.01 * FS)
    x[:r] *= np.linspace(0, 1, r)
    x[-r:] *= np.linspace(1, 0, r)
    return x / np.sqrt(np.mean(x ** 2)) * rms


def event_scenarios() -> dict[str, dict]:
    """Each: x (N, 1), speech truth segments, non-speech event segments (truth = NOT speech)."""
    base = [(0.8, 1.4, 0.1), (2.0, 2.7, 0.1)]

    def make(seed, events, extra_speech=(), noise_rms=1e-3, segs=base):
        x, truth = build(list(segs), 3.5, noise_rms, seed)
        rng = np.random.default_rng(seed + 100)
        ev_t = []
        for kind, t0, t1, amp in events:
            i0, i1 = int(t0 * FS), int(t1 * FS)
            m = i1 - i0
            if kind == "click_train":
                for tc in np.linspace(t0, t1, 4, endpoint=False):
                    j = int(tc * FS)
                    x[j:j + int(0.002 * FS), 0] += amp * rng.choice([-1, 1], int(0.002 * FS))
            elif kind == "white_burst":
                x[i0:i1, 0] += rng.standard_normal(m) * amp
            elif kind == "breath":
                x[i0:i1, 0] += noise(m, amp, rng)
            elif kind == "thump":
                t = np.arange(m) / FS
                x[i0:i1, 0] += amp * np.exp(-t * 15) * np.sin(2 * np.pi * 80 * t)
            ev_t.append((t0, t1))
        speech = list(truth)
        for t0, t1, rms in extra_speech:
            seg = _unvoiced(t1 - t0, rms, rng)
            x[int(t0 * FS):int(t0 * FS) + seg.size, 0] += seg
            speech.append((t0, t1))
        return {"x": x, "speech": sorted(speech), "events": ev_t}

    return {
        "loud_click_train": make(21, [("click_train", 0.2, 0.6, 0.6)]),
        "loud_white_noise_burst": make(22, [("white_burst", 0.1, 0.5, 0.1)]),
        "breath_like_noise_(-26dBFS)": make(23, [("breath", 0.1, 0.6, 0.05)]),
        "low_frequency_thump_(80Hz)": make(24, [("thump", 0.1, 0.4, 0.5)]),
        "quiet_voiced_speech_(rms_0.004)": make(25, [], segs=[(0.8, 1.4, 0.004), (2.0, 2.7, 0.004)], noise_rms=3e-4),
        "standalone_unvoiced_sound_(true_speech)": make(26, [], extra_speech=[(0.3, 0.5, 0.05)]),
        "unvoiced_sound_next_to_voiced_(true_speech)": make(27, [], extra_speech=[(1.4, 1.55, 0.05)]),
    }


def _speech_class_metrics(speech_truth, det, total) -> dict:
    n_grid = int(round(total * 1000 / GRID_MS))
    truth = _mask(speech_truth, n_grid)
    spk = _mask([(r["start_s"], r["end_s"]) for r in det["speech"]], n_grid)
    act = _mask([(r["start_s"], r["end_s"]) for r in det["active"]], n_grid)
    return {"speech_false_positive_rate": float((~truth & spk).sum() / max(1, (~truth).sum())),
            "speech_false_negative_rate": float((truth & ~spk).sum() / max(1, truth.sum())),
            "active_false_negative_rate": float((truth & ~act).sum() / max(1, truth.sum()))}


def _auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """P(feature of a positive frame > feature of a negative frame); ties count half."""
    if pos.size == 0 or neg.size == 0:
        return None
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def run_feature_evaluation(cfg: a.Config) -> dict:
    """Does a periodicity gate (or another frame feature) separate loud non-speech from speech-like activity?"""
    off = a.Config(**{**cfg.__dict__, "use_periodicity_gate": False})
    on = a.Config(**{**cfg.__dict__, "use_periodicity_gate": True})
    frame, hop = a._frame_params(FS, cfg)
    out = {"note": "Synthetic signals only. 'speech' = periodic harmonic bursts or high-passed noise labelled as speech; "
                   "events = clicks, noise bursts, breath-like noise and a low-frequency thump (labelled NOT speech). "
                   "Rates are on a 10 ms grid; speech false negative counts frames not accepted as confident speech "
                   "(they may still be preserved as ambiguous: see active_false_negative_rate).",
           "scenarios": {}, "feature_auc": {}}
    pos_f: dict[str, list] = {}
    neg_f: dict[str, list] = {}
    for name, sc in event_scenarios().items():
        x = sc["x"]
        feats = a.frame_features(x, FS, cfg)
        nf = len(feats["rms_db"])
        t = (np.arange(nf) * hop + frame / 2) / FS
        quiet = np.sort(feats["rms_db"])[: max(3, nf // 10)]
        loud = feats["rms_db"] > np.median(quiet) + cfg.high_margin_db
        in_ev = np.zeros(nf, bool)
        for t0, t1 in sc["events"]:
            in_ev |= (t >= t0) & (t <= t1)
        voiced = np.zeros(nf, bool)
        for t0, t1 in sc["speech"]:
            voiced |= (t >= t0) & (t <= t1)
        if "unvoiced" not in name:  # AUC: voiced speech-like frames vs loud non-speech frames
            for k in ("periodicity", "zcr", "spectral_flux", "spectral_flatness", "low_freq_fraction"):
                pos_f.setdefault(k, []).append(feats[k][loud & voiced])
                neg_f.setdefault(k, []).append(feats[k][loud & in_ev])
        total = x.shape[0] / FS
        out["scenarios"][name] = {"energy_only": _speech_class_metrics(sc["speech"], a.detect_silence(x, FS, off), total),
                                  "with_periodicity_gate": _speech_class_metrics(sc["speech"], a.detect_silence(x, FS, on), total)}
    for k in pos_f:
        out["feature_auc"][k] = _auc(np.concatenate(pos_f[k]), np.concatenate(neg_f[k]))
    return out


def join_validation(cfg: a.Config) -> dict:
    """Exercise the join code on synthetic signals with a known internal gap.

    Natural case: two speech bursts around a 1.2 s noise gap, so trim_silence really shortens the gap.
    Stress case: a continuous loud sine with fabricated active regions so the cut lands at high amplitude;
    a plain cut (fade 0 ms) is compared with the configured fade.
    """
    rng = np.random.default_rng(31)
    quiet = lambda d: 1e-3 * rng.standard_normal(int(d * FS))  # noqa: E731
    x = np.concatenate([quiet(0.5), speech_like(0.6, 0.1, rng, f0=120), quiet(1.2),
                        speech_like(0.6, 0.1, rng, f0=137), quiet(0.5)])[:, None]
    det = a.detect_silence(x, FS, cfg)
    proc, info = a.trim_silence(x, FS, det, cfg)
    natural = {"joins_created": len(info["join_samples"]), "check": a.check_joins(proc, info["join_samples"]),
               "shortened_gaps": info["shortened_gaps"], "join_samples": info["join_samples"]}
    t = np.arange(3 * FS) / FS
    loud = (0.5 * np.sin(2 * np.pi * 233.3 * t))[:, None]
    fake = {"active": [{"start_s": 0.5, "end_s": 1.0, "duration_s": 0.5, "type": "speech"},
                       {"start_s": 2.0, "end_s": 2.5, "duration_s": 0.5, "type": "speech"}], "speech": []}
    stress, stress_audio = {}, {}
    for label, c in (("plain_cut_fade_0ms", a.Config(**{**cfg.__dict__, "fade_ms": 0.0})), ("configured_fade", cfg)):
        p, i = a.trim_silence(loud, FS, fake, c)
        stress[label] = {**a.check_joins(p, i["join_samples"]), "join_samples": i["join_samples"]}
        stress_audio[label] = (p, i["join_samples"])
    result = {"natural_case": natural, "stress_case": stress,
              "interpretation": "Numerical continuity only: the largest sample-to-sample step at a join compared with "
                                "the 99.9th percentile step of the file. It says nothing about perceptual smoothness."}
    return result, {"natural": (proc, info["join_samples"]), **stress_audio}
