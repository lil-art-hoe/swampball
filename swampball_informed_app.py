from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date as DateType
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Helpers: parse dataset
# -----------------------------
def parse_numbers(s: str) -> Tuple[List[int], int]:
    nums = [int(x) for x in str(s).split()]
    if len(nums) < 6:
        raise ValueError(f"Bad Winning Numbers row: {s}")
    return nums[:5], nums[5]


def load_powerball_csv(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    if "Draw Date" not in df.columns or "Winning Numbers" not in df.columns:
        raise ValueError("CSV must contain columns: 'Draw Date' and 'Winning Numbers'.")

    df = df.copy()
    df["DrawDate"] = pd.to_datetime(df["Draw Date"], errors="coerce")
    df = df.dropna(subset=["DrawDate"])

    parsed = df["Winning Numbers"].apply(parse_numbers)
    df["W"] = parsed.apply(lambda x: x[0])
    df["PB"] = parsed.apply(lambda x: x[1])
    df["MaxW"] = df["W"].apply(max)

    # Filter to modern game format: white 1–69 and PB 1–26
    # (Older era had different ranges; this dataset includes those.)
    df = df[(df["PB"] <= 26) & (df["MaxW"] <= 69)].copy()

    # Expand whites into columns
    w = np.vstack(df["W"].to_list())
    df["W1"], df["W2"], df["W3"], df["W4"], df["W5"] = w.T
    df["WhiteSum"] = w.sum(axis=1)
    df["OddCount"] = (w % 2).sum(axis=1)
    df["LowCount"] = (w <= 12).sum(axis=1)
    df["RunMax"] = [longest_consecutive_run(row.tolist()) for row in w]

    return df.sort_values("DrawDate")


# -----------------------------
# Date gematria / numerology
# -----------------------------
def digitsum(n: int) -> int:
    n = abs(int(n))
    return sum(int(c) for c in str(n))


def digital_root(n: int) -> int:
    n = abs(int(n))
    while n >= 10:
        n = digitsum(n)
    return n


def is_leap_year(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def days_in_year(y: int) -> int:
    return 366 if is_leap_year(y) else 365


def day_of_year(d: DateType) -> int:
    return int(d.strftime("%j"))


def date_features(d: DateType) -> Dict[str, int]:
    y, m, dd = d.year, d.month, d.day
    ymd = y * 10000 + m * 100 + dd
    doy = day_of_year(d)
    dleft = days_in_year(y) - doy

    mmdd = m * 100 + dd
    ddmm = dd * 100 + m

    feats = {
        "year": y,
        "month": m,
        "day": dd,
        "yyyymmdd": ymd,
        "ds_yyyymmdd": digitsum(ymd),
        "dr_yyyymmdd": digital_root(ymd),
        "mmdd": mmdd,
        "ddmm": ddmm,
        "ds_mmdd": digitsum(mmdd),
        "dr_mmdd": digital_root(mmdd),
        "doy": doy,
        "ds_doy": digitsum(doy),
        "dr_doy": digital_root(doy),
        "days_left": dleft,
        "ds_days_left": digitsum(dleft),
        "dr_days_left": digital_root(dleft),
        "md": m + dd,
        "ds_md": digitsum(m + dd),
        "dr_md": digital_root(m + dd),
        "m_times_d": m * dd,
        "ds_m_times_d": digitsum(m * dd),
        "dr_m_times_d": digital_root(m * dd),
    }
    return feats


# -----------------------------
# Structural constraints
# -----------------------------
@dataclass
class Constraints:
    odd_min: int = 1
    odd_max: int = 4
    sum_min: int = 110
    sum_max: int = 225
    low_max: int = 3
    run_max: int = 2


def longest_consecutive_run(nums: List[int]) -> int:
    s = sorted(nums)
    best = run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def passes_constraints(whites: List[int], c: Constraints) -> bool:
    odd = sum(n % 2 for n in whites)
    if not (c.odd_min <= odd <= c.odd_max):
        return False
    s = sum(whites)
    if not (c.sum_min <= s <= c.sum_max):
        return False
    low = sum(1 for n in whites if n <= 12)
    if low > c.low_max:
        return False
    if longest_consecutive_run(whites) > c.run_max:
        return False
    return True


# -----------------------------
# Frequency + scoring
# -----------------------------
def frequency_tables(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    whites = np.vstack(df[["W1", "W2", "W3", "W4", "W5"]].to_numpy())
    pb = df["PB"].to_numpy()

    wc = pd.Series(whites.flatten()).value_counts().sort_index()
    pc = pd.Series(pb).value_counts().sort_index()

    wc = wc.reindex(range(1, 70), fill_value=0)
    pc = pc.reindex(range(1, 27), fill_value=0)
    return wc, pc


def make_weights(counts: pd.Series, alpha: float = 1.0) -> np.ndarray:
    x = counts.to_numpy(dtype=float) + 1.0  # +1 smoothing
    x = np.power(x, alpha)
    return x / x.sum()


def gematria_match_score(whites: List[int], pb: int, feats: Dict[str, int]) -> float:
    raw_vals = [
        feats["dr_yyyymmdd"],
        feats["ds_yyyymmdd"],
        feats["dr_mmdd"],
        feats["ds_mmdd"],
        feats["doy"],
        feats["dr_doy"],
        feats["md"],
        feats["dr_md"],
        feats["m_times_d"],
        feats["dr_m_times_d"],
    ]

    reduced_white = [(v - 1) % 69 + 1 for v in raw_vals]
    reduced_pb = [(v - 1) % 26 + 1 for v in raw_vals]

    white_set = set(whites)
    score = 0.0

    score += 1.5 * sum(1 for v in reduced_white if v in white_set)
    score += 1.0 if pb in set(reduced_pb) else 0.0
    score += 0.75 if feats["dr_yyyymmdd"] in white_set else 0.0

    return score


def structural_score(whites: List[int], c: Constraints) -> float:
    s = sum(whites)
    odd = sum(n % 2 for n in whites)
    low = sum(1 for n in whites if n <= 12)
    run = longest_consecutive_run(whites)

    sum_center = (c.sum_min + c.sum_max) / 2.0
    sum_span = max(1.0, (c.sum_max - c.sum_min) / 2.0)
    sum_score = 1.0 - min(1.0, abs(s - sum_center) / (sum_span * 1.25))

    odd_center = (c.odd_min + c.odd_max) / 2.0
    odd_span = max(1.0, (c.odd_max - c.odd_min) / 2.0)
    odd_score = 1.0 - min(1.0, abs(odd - odd_center) / (odd_span * 1.25))

    low_score = 1.0 - min(1.0, max(0, low - 2) / 3.0)
    run_score = 1.0 - min(1.0, max(0, run - 2) / 3.0)

    return 2.0 * sum_score + 1.0 * odd_score + 0.5 * low_score + 0.5 * run_score


def generate_informed_sets(
    df: pd.DataFrame,
    target_date: DateType,
    n_sets: int,
    candidates: int,
    c: Constraints,
    alpha_white: float,
    alpha_pb: float,
    w_freq: float,
    w_struct: float,
    w_gem: float,
    seed_salt: str,
) -> Tuple[pd.DataFrame, Dict[str, int], pd.Series, pd.Series]:
    wc, pc = frequency_tables(df)
    w_weights = make_weights(wc, alpha=alpha_white)
    pb_weights = make_weights(pc, alpha=alpha_pb)

    feats = date_features(target_date)

    blob = f"{target_date.isoformat()}|{seed_salt}|{alpha_white}|{alpha_pb}|{w_freq}|{w_struct}|{w_gem}|{c}".encode("utf-8")
    seed = int(hashlib.sha256(blob).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)

    w_logp = np.log(w_weights + 1e-12)
    pb_logp = np.log(pb_weights + 1e-12)

    seen = set()
    rows = []

    for _ in range(candidates):
        whites = rng.choice(np.arange(1, 70), size=5, replace=False, p=w_weights).tolist()
        whites.sort()

        if not passes_constraints(whites, c):
            continue

        pb = int(rng.choice(np.arange(1, 27), p=pb_weights))

        key = (tuple(whites), pb)
        if key in seen:
            continue
        seen.add(key)

        freq_score = float(sum(w_logp[n - 1] for n in whites) + pb_logp[pb - 1])
        struct_score = float(structural_score(whites, c))
        gem_score = float(gematria_match_score(whites, pb, feats))

        total = w_freq * freq_score + w_struct * struct_score + w_gem * gem_score

        rows.append({
            "Set": 0,  # filled later
            "Whites": " ".join(f"{n:02d}" for n in whites),
            "Powerball": f"{pb:02d}",
            "WhiteSum": sum(whites),
            "OddCount": sum(n % 2 for n in whites),
            "FreqScore": freq_score,
            "StructScore": struct_score,
            "GematriaScore": gem_score,
            "TotalScore": total,
        })

    if not rows:
        raise RuntimeError("No candidates passed constraints. Loosen constraints or increase candidates.")

    out = pd.DataFrame(rows).sort_values("TotalScore", ascending=False).head(n_sets).reset_index(drop=True)
    out["Set"] = np.arange(1, len(out) + 1)
    return out, feats, wc, pc


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Powerball Informed Builder", layout="wide")
st.title("Powerball Informed Builder (Frequency + Structure + Date Gematria)")

st.caption(
    "This app does *not* predict lottery outcomes. It generates consistent, date-shaped picks using historical frequency + realistic draw structure + date-gematria alignment."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Powerball history CSV", type=["csv"])
    st.write("Expected columns: **Draw Date**, **Winning Numbers** (5 whites + PB).")

    st.divider()
    st.header("Target")
    target_date = st.date_input("Target date", value=dt.date(2025, 12, 17))
    n_sets = st.slider("How many sets to output?", 5, 50, 20, 1)
    candidates = st.slider("How many candidates to search?", 500, 50000, 12000, 500)

    seed_salt = st.text_input("Optional salt (keeps runs stable)", value="lil_art_hoe")

    st.divider()
    st.header("Bias controls")
    alpha_white = st.slider("White frequency emphasis", 0.2, 3.0, 1.1, 0.1)
    alpha_pb = st.slider("Powerball frequency emphasis", 0.2, 3.0, 1.1, 0.1)

    w_freq = st.slider("Weight: frequency", 0.0, 5.0, 1.2, 0.1)
    w_struct = st.slider("Weight: structure", 0.0, 5.0, 2.0, 0.1)
    w_gem = st.slider("Weight: date gematria", 0.0, 5.0, 1.0, 0.1)

    st.divider()
    st.header("Structural constraints")
    c = Constraints(
        odd_min=st.slider("Odd min", 0, 5, 1),
        odd_max=st.slider("Odd max", 0, 5, 4),
        sum_min=st.slider("White sum min", 50, 260, 110, 1),
        sum_max=st.slider("White sum max", 80, 300, 225, 1),
        low_max=st.slider("Max count ≤12", 0, 5, 3),
        run_max=st.slider("Max consecutive run", 1, 5, 2),
    )

    run_btn = st.button("Generate", type="primary")

if not uploaded:
    st.info("Upload your CSV (like the one you already have) to enable generation.")
    st.stop()

df = load_powerball_csv(uploaded)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Modern-format dataset summary")
    st.write(f"Rows (modern 69/26 era): **{len(df):,}**")
    st.write(f"Date range: **{df['DrawDate'].min().date()} → {df['DrawDate'].max().date()}**")
    st.dataframe(df[["Draw Date", "Winning Numbers", "PB", "WhiteSum", "OddCount"]].tail(10), use_container_width=True)

with col2:
    st.subheader("Typical structure (from your dataset)")
    st.write("These guide the default constraints:")
    st.write(f"- Median white sum: **{int(df['WhiteSum'].median())}**")
    st.write(f"- 10th–90th percentile white sum: **{int(df['WhiteSum'].quantile(0.10))} – {int(df['WhiteSum'].quantile(0.90))}**")
    st.write(f"- Most common odd counts: **{df['OddCount'].value_counts().head(3).to_dict()}**")

if run_btn:
    try:
        out, feats, wc, pc = generate_informed_sets(
            df=df,
            target_date=target_date,
            n_sets=n_sets,
            candidates=candidates,
            c=c,
            alpha_white=alpha_white,
            alpha_pb=alpha_pb,
            w_freq=w_freq,
            w_struct=w_struct,
            w_gem=w_gem,
            seed_salt=seed_salt.strip(),
        )

        st.divider()
        a, b = st.columns([1.3, 1])

        with a:
            st.subheader(f"Top {len(out)} informed sets for {target_date.isoformat()}")
            st.dataframe(out, use_container_width=True, hide_index=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download sets as CSV",
                data=csv_bytes,
                file_name=f"powerball_informed_{target_date.isoformat()}.csv",
                mime="text/csv",
            )

        with b:
            st.subheader("Date gematria features (curated)")
            show = ["yyyymmdd", "ds_yyyymmdd", "dr_yyyymmdd", "mmdd", "dr_mmdd", "doy", "dr_doy", "days_left", "md", "dr_md", "m_times_d", "dr_m_times_d"]
            st.json({k: feats[k] for k in show})

            st.subheader("Frequency snapshot (top 10)")
            topw = wc.sort_values(ascending=False).head(10).rename("WhiteCount").reset_index().rename(columns={"index": "White"})
            topp = pc.sort_values(ascending=False).head(10).rename("PBCount").reset_index().rename(columns={"index": "PB"})
            st.write("White balls:")
            st.dataframe(topw, use_container_width=True, hide_index=True)
            st.write("Powerballs:")
            st.dataframe(topp, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(str(e))
        st.info("Try loosening constraints (sum range / run limit) or increasing candidate search.")
else:
    st.write("Upload your CSV, set your options in the sidebar, then click **Generate**.")
    st.caption("Tip: Same settings => same output.")
