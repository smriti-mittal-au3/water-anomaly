"""
Live replay dashboard for the water plant anomaly detector.

Run train.py first, then:
    streamlit run app.py
"""
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RESULTS = Path("results")
VIEW = 1500   # samples shown in the charts

st.set_page_config(page_title="Water Plant Anomaly Monitor", layout="wide")

if not (RESULTS / "test_windows.pkl").exists():
    st.error("No results found. Run train.py first.")
    st.stop()


@st.cache_data
def load():
    df = pd.read_pickle(RESULTS / "test_windows.pkl")
    alerts = pd.read_csv(RESULTS / "alerts.csv") if (RESULTS / "alerts.csv").stat().st_size > 1 else pd.DataFrame()
    metrics = pd.read_csv(RESULTS / "metrics.csv", index_col=0)
    attacks = (pd.read_csv(RESULTS / "attacks.csv", parse_dates=["start", "end"])
               if (RESULTS / "attacks.csv").exists() else pd.DataFrame())
    return df, alerts, metrics, attacks


df, alerts, metrics, attacks = load()
sensors = [c[len("err__"):] for c in df.columns if c.startswith("err__")]
n = len(df)
threshold = float(df["threshold"].iloc[0])

# ---------------------------------------------------------------- replay controls
# Advance before the slider is drawn, so the slider shows the new position.
st.session_state.setdefault("pos", min(VIEW, n - 1))
if st.session_state.get("playing"):
    st.session_state.pos = min(st.session_state.pos + st.session_state.get("speed", 20), n - 1)

st.sidebar.header("Replay")
st.sidebar.toggle("Play", key="playing")
st.sidebar.slider("Speed (samples per tick)", 1, 200, 20, key="speed")
st.sidebar.slider("Position", 0, n - 1, key="pos")
pos = st.session_state.pos

# ---------------------------------------------------------------- header
st.title("Water Plant Anomaly Monitor")
st.caption("Learns a treatment plant's normal behaviour and raises early warnings when it deviates. "
           "Replaying the SWaT water treatment testbed data.")

m = metrics.loc["autoencoder"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Attacks detected", m["attacks_detected"])
c2.metric("Median detection delay", f"{float(m['median_delay_s']):.0f} s" if pd.notna(m["median_delay_s"]) else "n/a")
c3.metric("False alarm events", int(m["false_alarm_events"]))
c4.metric("False alarm rate", f"{float(m['false_alarm_rate']):.2%}")

# ---------------------------------------------------------------- current status
row = df.iloc[pos]
err_now = pd.Series({s: row[f"err__{s}"] for s in sensors}).nlargest(3)
if row["alert"]:
    st.error(f"⚠️ ANOMALY at {row['time']} · score {row['score']:.4f} (threshold {threshold:.4f}) · "
             f"most abnormal: {', '.join(err_now.index)}")
else:
    st.success(f"✅ Normal operation at {row['time']} · score {row['score']:.4f}")

# What was really happening, from the dataset's attack list.
if len(attacks):
    now = pd.Timestamp(row["time"])
    live = attacks[(attacks["start"] <= now) & (attacks["end"] >= now)]
    if len(live):
        a = live.iloc[0]
        st.info(f"Ground truth: attack #{a['attack']} in progress: {a['action']} on {a['target']}. "
                f"Attacker intent: {a['intent']}")

# ---------------------------------------------------------------- score chart
view = df.iloc[max(0, pos - VIEW): pos + 1].reset_index(drop=True)
attack_spans = []
mask = np.concatenate([[0], view["label"].to_numpy(), [0]])
d = np.diff(mask)
for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
    attack_spans.append({"start": view["time"].iloc[s], "end": view["time"].iloc[e - 1]})

score_line = alt.Chart(view).mark_line(strokeWidth=1).encode(
    x=alt.X("time:T", title=None),
    y=alt.Y("score:Q", scale=alt.Scale(type="log"), title="Anomaly score (log)"),
)
thr = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
layers = [score_line, thr]
if attack_spans:
    layers.insert(0, alt.Chart(pd.DataFrame(attack_spans)).mark_rect(opacity=0.2, color="red")
                  .encode(x="start:T", x2="end:T"))
st.subheader("Anomaly score")
st.caption("Dashed line = alert threshold. Red bands = real attacks from the dataset's answer key.")
st.altair_chart(alt.layer(*layers).properties(height=250), width="stretch")

# ---------------------------------------------------------------- sensors
st.subheader("Sensor readings")
default = list(err_now.index) if row["alert"] else [s for s in ("LIT101", "FIT101", "P101") if s in sensors]
chosen = st.multiselect("Sensors", sensors, default=default[:4])
cols = st.columns(2)
for i, s in enumerate(chosen):
    chart = alt.Chart(view).mark_line(strokeWidth=1).encode(
        x=alt.X("time:T", title=None), y=alt.Y(f"{s}:Q", scale=alt.Scale(zero=False)))
    cols[i % 2].altair_chart(chart.properties(height=160, title=s), width="stretch")

# ---------------------------------------------------------------- alert log
st.subheader("Alert log")
if len(alerts):
    seen = alerts[alerts["start_idx"] <= pos].iloc[::-1]
    shown = ["start", "duration_s", "peak_score", "top_sensors", "real_attack"]
    if "matched_attack" in seen.columns:
        shown.append("matched_attack")
    st.dataframe(seen[shown], width="stretch", hide_index=True)
else:
    st.write("No alerts.")

with st.expander("How it works"):
    st.markdown(
        "- An autoencoder is trained **only on normal plant data**. No attack labels are used for training.\n"
        "- It learns to reproduce normal sensor patterns. When the plant behaves abnormally, it reproduces "
        "them badly, and the reconstruction error (the anomaly score) rises.\n"
        "- The threshold is set from held-out normal data, not tuned on attacks.\n"
        "- The sensors with the largest error tell operators **where** to look.\n"
        "- The anomalies in this dataset are cyber-physical attacks on a real testbed. The same method flags "
        "any deviation from normal: sensor faults, equipment problems or quality excursions.\n"
        "\nBaseline comparison:"
    )
    st.dataframe(metrics, width="stretch")

if st.session_state.playing and pos < n - 1:
    time.sleep(0.3)
    st.rerun()
