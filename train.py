"""
Water plant anomaly detection on SWaT.

Learns what normal plant behaviour looks like from the normal file only,
then scores the attack file. Labels are used only to grade the result.
Saves the model, scores, alerts, metrics and plots to results/.

The idea in one line: teach a model to "redraw" normal sensor readings.
When the plant behaves abnormally, the model redraws it badly, and that
redrawing error is our anomaly score.

Usage:
    python train.py --normal SWaT_Dataset_Normal_v1.xlsx --attack SWaT_Dataset_Attack_v0.xlsx
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")             # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

WINDOW = 10               # seconds averaged into one sample (smooths sensor noise, 10x less data)
STARTUP_SKIP = 6 * 3600   # first 6 hours of normal data are plant start-up, not steady operation
SMOOTH = 3                # samples averaged when smoothing the score (one blip shouldn't alarm)


# ---------------------------------------------------------------- loading
# Get the raw spreadsheets into clean numbers: sensor table, labels, timestamps.

def read_table(path, **kw):
    if Path(path).suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    return pd.read_csv(path, low_memory=False, **kw)


def load_swat(path):
    """Return (features DataFrame, labels 0/1, timestamps). Cached, since xlsx is slow to read."""
    # Reading the Excel file takes ~5 minutes, so the first time we save a fast copy in cache/.
    cache = Path("cache") / (Path(path).stem + ".pkl")
    if cache.exists():
        return pd.read_pickle(cache)
    result = _load_swat(path)
    cache.parent.mkdir(exist_ok=True)
    pd.to_pickle(result, cache)
    return result


def _load_swat(path):
    # Some SWaT files have an extra title row above the real header.
    # Find the row that contains a known sensor name and use it as the header.
    head = read_table(path, header=None, nrows=5)
    header_row = next(
        (i for i in range(len(head)) if head.iloc[i].astype(str).str.contains("LIT101").any()),
        0,
    )
    df = read_table(path, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]   # " MV101" -> "MV101"

    label_col = next(c for c in df.columns if "attack" in c.lower())   # "Normal/Attack"
    time_col = next((c for c in df.columns if "time" in c.lower()), None)

    # Labels are "Normal" / "Attack", sometimes misspelt "A ttack".
    # Turn them into 0 (normal) / 1 (attack). Used ONLY for grading, never for training.
    labels = df[label_col].astype(str).str.replace(" ", "").str.lower()
    y = (labels == "attack").astype(int).to_numpy()

    # Timestamps are written day-first, e.g. "28/12/2015 10:00:00 AM".
    t = None
    if time_col:
        t = pd.to_datetime(df[time_col].astype(str).str.strip(),
                           dayfirst=True, format="mixed", errors="coerce")
    if t is None or t.isna().all():
        t = pd.Series(pd.to_datetime(np.arange(len(df)), unit="s"))
    t = t.ffill().to_numpy()

    # Everything else is a sensor or actuator reading. Force to numbers, fill gaps.
    feats = df.drop(columns=[c for c in (label_col, time_col) if c])
    feats = feats.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)
    return feats, y, t


def to_windows(X, y, t):
    """Average every WINDOW rows into one sample. A window is an attack if any row is."""
    n = len(X) // WINDOW * WINDOW                              # drop leftover rows at the end
    Xw = X[:n].reshape(-1, WINDOW, X.shape[1]).mean(axis=1)    # 10 rows -> 1 averaged row
    yw = y[:n].reshape(-1, WINDOW).max(axis=1)                 # any attack second -> attack window
    tw = t[:n:WINDOW]                                          # timestamp of each window's start
    return Xw, yw, tw


# ---------------------------------------------------------------- scoring
# Helpers to turn a model into anomaly scores and grade them against the answer key.

def runs(mask):
    """(start, end) index pairs for each run of True values, end exclusive."""
    # e.g. [0,1,1,0,1] -> [(1,3), (4,5)]. Used to turn rows into "events" (attacks or alerts).
    m = np.concatenate([[0], np.asarray(mask, dtype=int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def errors(model, X):
    # Ask the model to rebuild the input, then measure how wrong it was, per sensor.
    # Autoencoder: predict(X). PCA: squeeze to fewer numbers (transform) and expand back.
    recon = model.predict(X) if hasattr(model, "predict") else model.inverse_transform(model.transform(X))
    return (X - recon) ** 2


def smooth(score):
    # Rolling average over the last SMOOTH samples, using only the past (works live).
    return pd.Series(score).rolling(SMOOTH, min_periods=1).mean().to_numpy()


def load_attack_list(path):
    """Read SWaT's List_of_attacks_Final.xlsx into one row per attack with full start/end times."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=["Start Time", "End Time"])   # skip "no physical impact" rows
    start = pd.to_datetime(df["Start Time"])
    # End Time holds only a time of day, on the same date as the start.
    end = pd.to_datetime(start.dt.strftime("%Y-%m-%d ") + df["End Time"].astype(str), errors="coerce")
    return pd.DataFrame({
        "attack": df["Attack #"], "start": start, "end": end,
        "target": df["Attack Point"], "action": df["Attack"],
        "intent": df["Expected Impact or attacker intent"],
    }).dropna(subset=["end"])


def evaluate(y, alert):
    # Row-level counts: compare each 10 s window's alert with the answer key.
    tp = int(((alert == 1) & (y == 1)).sum())   # alarm during an attack (good)
    fp = int(((alert == 1) & (y == 0)).sum())   # alarm while normal (false alarm)
    fn = int(((alert == 0) & (y == 1)).sum())   # silent during an attack (missed)
    tn = int(((alert == 0) & (y == 0)).sum())   # silent while normal (good)
    precision = tp / max(tp + fp, 1)            # of all alarms, how many were real
    recall = tp / max(tp + fn, 1)               # of all attack time, how much was flagged

    # Event-level: an attack counts as "detected" if we alarm at least once during it.
    # Delay = seconds from attack start to our first alarm.
    events = runs(y == 1)
    delays = []
    for s, e in events:
        hits = np.where(alert[s:e] == 1)[0]
        if len(hits):
            delays.append(hits[0] * WINDOW)
    # A false alarm event is an alarm period with no attack in it at all.
    false_alarms = sum(1 for s, e in runs(alert == 1) if y[s:e].sum() == 0)

    return {
        "attacks_detected": f"{len(delays)}/{len(events)}",
        "event_recall": round(len(delays) / max(len(events), 1), 3),
        "median_delay_s": float(np.median(delays)) if delays else None,
        "false_alarm_events": false_alarms,
        "false_alarm_rate": round(fp / max(fp + tn, 1), 4),   # share of normal time we alarmed
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True, help="SWaT normal file (csv or xlsx)")
    ap.add_argument("--attack", required=True, help="SWaT attack file (csv or xlsx)")
    ap.add_argument("--attack-list", help="optional List_of_attacks_Final.xlsx, to describe each alert")
    ap.add_argument("--drop", default="",
                    help="comma-separated sensors to exclude, e.g. analyzers that drifted between recordings")
    ap.add_argument("--out", default="results")
    ap.add_argument("--percentile", type=float, default=99.5,
                    help="threshold = this percentile of scores on held-out normal data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # ---- Step 1: load both files
    print("Loading normal data (xlsx files take a few minutes) ...")
    Xn, yn, tn = load_swat(args.normal)
    print(f"  {len(Xn):,} rows, {Xn.shape[1]} sensors/actuators")
    print("Loading attack data ...")
    Xa, ya, ta = load_swat(args.attack)
    print(f"  {len(Xa):,} rows, {ya.sum():,} labelled attack")

    # ---- Step 2: choose which sensors to use
    dropped = {s.strip() for s in args.drop.split(",") if s.strip()}
    features = [c for c in Xn.columns if c in Xa.columns and c not in dropped]
    if dropped:
        print(f"Excluding: {', '.join(sorted(dropped))}")

    # ---- Step 3: prepare training data (normal only) and test data (attack file)
    # Training data: normal file, minus start-up, minus anything labelled attack.
    skip = STARTUP_SKIP if len(Xn) > 2 * STARTUP_SKIP else 0
    if not skip:
        print("  normal file is short, not skipping start-up period")
    keep = np.arange(len(Xn)) >= skip
    keep &= yn == 0
    Xn_w, _, _ = to_windows(Xn[features].to_numpy()[keep], yn[keep], tn[keep])
    Xa_w, ya_w, ta_w = to_windows(Xa[features].to_numpy(), ya, ta)

    # Sensors that never change in normal operation carry no information.
    varying = Xn_w.std(axis=0) > 1e-6
    features = [f for f, v in zip(features, varying) if v]
    Xn_w, Xa_w = Xn_w[:, varying], Xa_w[:, varying]
    print(f"Using {len(features)} sensors, {len(Xn_w):,} normal and {len(Xa_w):,} test samples "
          f"({WINDOW}s windows)")

    # Hold out the last 20% of normal data (in time order) to set the threshold.
    # Not shuffled: the future must never leak into training (Arp et al.).
    split = int(len(Xn_w) * 0.8)
    # Rescale every sensor to 0..1 using training data only, so tank levels (~800)
    # don't drown out flows (~2).
    scaler = MinMaxScaler().fit(Xn_w[:split])
    train, val, test = (scaler.transform(a) for a in (Xn_w[:split], Xn_w[split:], Xa_w))

    # ---- Step 4: the two models
    models = {
        # Autoencoder: a small neural network that squeezes ~37 readings into 4 numbers
        # (the "bottleneck") and expands them back. It can only do this well for
        # patterns it saw in training, i.e. normal operation.
        # Small and regularised: bigger networks overfit and false-alarm on unseen normal data.
        "autoencoder": MLPRegressor(hidden_layer_sizes=(16, 4, 16), alpha=1e-2, max_iter=300,
                                    early_stopping=True, random_state=0),
        # PCA: the classic linear version of the same idea, as a baseline to beat.
        "pca_baseline": PCA(n_components=0.95, random_state=0),
    }

    # ---- Step 5: train, set threshold, score the attack file, grade
    rows, results = {}, {}
    for name, model in models.items():
        print(f"Training {name} ...")
        if name == "autoencoder":
            model.fit(train, train)       # input = target: learn to reproduce normal data
        else:
            model.fit(train)
        # Divide each sensor's error by its typical error on training data, so naturally
        # noisy sensors need a bigger deviation before they count.
        typical = errors(model, train).mean(axis=0) + 1e-9
        # Threshold: a score higher than 99.5% of held-out NORMAL scores. No attack data used.
        threshold = float(np.percentile(smooth((errors(model, val) / typical).mean(axis=1)), args.percentile))
        err = errors(model, test) / typical       # per-sensor error on the attack file
        score = smooth(err.mean(axis=1))          # one anomaly score per 10 s window
        alert = (score > threshold).astype(int)   # alarm when above threshold
        rows[name] = evaluate(ya_w, alert)        # only now look at the answer key
        results[name] = (threshold, err, score, alert)
        print(f"  {rows[name]}")

    metrics = pd.DataFrame(rows).T
    metrics.to_csv(out / "metrics.csv")
    print("\n", metrics.to_string())

    # ---- Step 6: save everything the dashboard and the application need
    # Everything below uses the autoencoder, the main model.
    threshold, err, score, alert = results["autoencoder"]
    joblib.dump({"scaler": scaler, "model": models["autoencoder"], "features": features,
                 "threshold": threshold, "window_s": WINDOW}, out / "model.joblib")

    # One table per test window: time, answer key, score, alarm, raw readings, per-sensor error.
    frame = pd.DataFrame(Xa_w, columns=features)
    frame.insert(0, "time", ta_w)
    frame.insert(1, "label", ya_w)
    frame.insert(2, "score", score)
    frame.insert(3, "alert", alert)
    frame["threshold"] = threshold
    err_df = pd.DataFrame(err, columns=[f"err__{f}" for f in features])
    pd.concat([frame, err_df], axis=1).to_pickle(out / "test_windows.pkl")

    # One row per alert, with the sensors that looked most abnormal.
    # This is the "where should the operator look?" explanation.
    alert_rows = []
    for s, e in runs(alert == 1):
        top = pd.Series(err[s:e].mean(axis=0), index=features).nlargest(3).index
        alert_rows.append({
            "start_idx": s, "end_idx": e,
            "start": ta_w[s], "end": ta_w[e - 1],
            "duration_s": (e - s) * WINDOW,
            "peak_score": round(float(score[s:e].max()), 4),
            "top_sensors": ", ".join(top),
            "real_attack": bool(ya_w[s:e].any()),
        })
    alerts = pd.DataFrame(alert_rows)

    # Match alerts to the documented attacks, so the demo can say what really happened.
    if args.attack_list:
        attacks = load_attack_list(args.attack_list)
        attacks.to_csv(out / "attacks.csv", index=False)
        if len(alerts):
            def describe(a):
                # Any attack whose time span overlaps this alert's time span.
                hit = attacks[(attacks["start"] <= a["end"]) & (attacks["end"] >= a["start"])]
                if not len(hit):
                    return ""
                h = hit.iloc[0]
                return f"#{h['attack']}: {h['action']} on {h['target']} (intent: {h['intent']})"
            alerts["matched_attack"] = alerts.apply(describe, axis=1)
    alerts.to_csv(out / "alerts.csv", index=False)

    # Overview plot: score over time, threshold, true attacks shaded red.
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ta_w, score, lw=0.6, label="anomaly score")
    ax.axhline(threshold, color="k", ls="--", lw=1, label="threshold")
    for s, e in runs(ya_w == 1):
        ax.axvspan(ta_w[s], ta_w[e - 1], color="red", alpha=0.2)
    ax.set_yscale("log")   # attack scores can be 10,000x normal; log keeps both visible
    ax.set_title("Anomaly score on the attack period (red = real attacks)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)

    # A few raw sensors with the same shading.
    shown = [f for f in ("LIT101", "FIT101", "MV101", "P101") if f in features] or features[:4]
    fig, axes = plt.subplots(len(shown), 1, figsize=(14, 2 * len(shown)), sharex=True)
    for ax, f in zip(np.atleast_1d(axes), shown):
        ax.plot(ta_w, frame[f], lw=0.6)
        for s, e in runs(ya_w == 1):
            ax.axvspan(ta_w[s], ta_w[e - 1], color="red", alpha=0.2)
        ax.set_ylabel(f)
    fig.tight_layout()
    fig.savefig(out / "sensors.png", dpi=150)

    print(f"\nSaved model, scores, alerts and plots to {out.resolve()}")


if __name__ == "__main__":
    main()
