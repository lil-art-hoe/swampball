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
# Load + parse Powerball history
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

    # Keep modern 69/26 era only
    df = df[(df["PB"] <= 26) & (df["MaxW"] <= 69)].copy()

    w = np.vstack(df["W"].to_list())
    df["W1"], df["W2"], df["W3"], df["W4"], df["W5"] = w.T
    return df.sort_values("DrawDate")


def frequency_tables(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    whites = np.vstack(df[["W1", "W2", "W3", "W4", "W5"]].to_numpy())
    pb = df["PB"].to_numpy()

    wc = pd.Series(whites.flatten()).value_counts().sort_index().reindex(range(1, 70), fill_value=0)
    pc = pd.Series(pb).value_counts().sort_index().reindex(range(1, 27), fill_value=0)
    return wc, pc


def make_weights(counts: pd.Series, alpha: float = 1.0) -> np.ndarray:
    # alpha > 1 pushes "hot" numbers, alpha < 1 flattens
    x = counts.to_numpy(dtype=float) + 1.0  # smoothing
    x = np.power(x, alpha)
    return x / x.sum()


# -----------------------------
# Date gematria (matches your screenshot layout)
# Example for 12/20/2025:
# (12)+(20)+(20)+(25)=77
# (12)+(20)+2+0+2+5=41
# 1+2+2+0+2+0+2+5=14
# (12)+(20)+(25)=57
# 1+2+2+0+2+5=12
# Day of Year=354, Days Left=11
# (12)+(20)=32
# 1+2+2+0+(20)+(25)=50
# (12)+(20)+2+5=39
# 1+2+2+0+(25)=30
# 1×2×2×2×2×5=80
# 1×2×2×2×5=40
# -----------------------------
def digitsum_from_str(s: str) -> int:
    return sum(int(ch) for ch in s if ch.isdigit())


def product_nonzero_digits_from_str(s: str) -> int:
    prod = 1
    any_digit = False
    for ch in s:
        if ch.isdigit():
            d = int(ch)
            if d != 0:
                prod *= d
                any_digit = True
    return prod if any_digit else 0


def is_leap_year(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def days_in_year(y: int) -> int:
    return 366 if is_leap_year(y) else 365


def day_of_year(d: DateType) -> int:
    # Jan 1 = 1, Dec 31 = 365/366
    return int(d.strftime("%j"))


def date_gematria_layout(d: DateType) -> pd.DataFrame:
    m = d.month
    dd = d.day
    y = d.year
    century = y // 100          # e.g. 20
    yy = y % 100                # e.g. 25

    mm_str = f"{m:02d}"
    dd_str = f"{dd:02d}"
    yyyy_str = f"{y:04d}"
    yy_str = f"{yy:02d}"

    mmddyyyy = f"{mm_str}{dd_str}{yyyy_str}"  # MMDDYYYY
    mmddyy = f"{mm_str}{dd_str}{yy_str}"      # MMDDYY

    # Values (right column in your screenshot)
    v1 = m + dd + century + yy
    v2 = m + dd + digitsum_from_str(yyyy_str)
    v3 = digitsum_from_str(mmddyyyy)
    v4 = m + dd + yy
    v5 = digitsum_from_str(mmddyy)

    doy = day_of_year(d)
    dleft = days_in_year(y) - doy

    v6 = m + dd
    v7 = digitsum_from_str(mm_str) + digitsum_from_str(dd_str) + century + yy
    v8 = m + dd + digitsum_from_str(yy_str)
    v9 = digitsum_from_str(mm_str) + digitsum_from_str(dd_str) + yy

    v10 = product_nonzero_digits_from_str(mmddyyyy)
    v11 = product_nonzero_digits_from_str(mmddyy)

    rows = [
        {"Label": f"({m}) + ({dd}) + ({century}) + ({yy})", "Value": v1},
        {"Label": f"({m}) + ({dd}) + " + " + ".join(list(yyyy_str)), "Value": v2},
        {"Label": " + ".join(list(mmddyyyy)), "Value": v3},
        {"Label": f"({m}) + ({dd}) + ({yy})", "Value": v4},
        {"Label": " + ".join(list(mmddyy)), "Value": v5},
        {"Label": f"Day of Year: ({d.strftime('%b')}-{dd_str})", "Value": doy},
        {"Label": f"Days Left in Year: ({d.strftime('%b')}-{dd_str})", "Value": dleft},
        {"Label": f"({m}) + ({dd})", "Value": v6},
        {"Label": " + ".join(list(mm_str)) + " + " + " + ".join(list(dd_str)) + f" + ({century}) + ({yy})", "Value": v7},
        {"Label": f"({m}) + ({dd}) + " + " + ".join(list(yy_str)), "Value": v8},
        {"Label": " + ".join(list(mm_str)) + " + " + " + ".join(list(dd_str)) + f" + ({yy})", "Value": v9},
        {"Label": " × ".join([ch for ch in mmddyyyy if ch.isdigit() and ch != "0"]), "Value": v10},
        {"Label": " × ".join([ch for ch in mmddyy if ch.isdigit() and ch != "0"]), "Value": v11},
    ]
    return pd.DataFrame(rows)


def date_features_compact(d: DateType) -> Dict[str, int]:
    # Keep a compact dict for seeding + display.
    m = d.month
    dd = d.day
    y = d.year
    century = y // 100
    yy = y % 100
    mmddyyyy = f"{m:02d}{dd:02d}{y:04d}"
    mmddyy = f"{m:02d}{dd:02d}{yy:02d}"
    doy = day_of_year(d)
    dleft = days_in_year(y) - doy
    return {
        "month": m,
        "day": dd,
        "year": y,
        "century": century,
        "yy": yy,
        "mmddyyyy_digitsum": digitsum_from_str(mmddyyyy),
        "mmddyy_digitsum": digitsum_from_str(mmddyy),
        "day_of_year": doy,
        "days_left_in_year": dleft,
    }


# -----------------------------
# Constraints (kept simple)
# -----------------------------
@dataclass
class Constraints:
    odd_min: int = 1
    odd_max: int = 4
    sum_min: int = 105
    sum_max: int = 230
    low_max: int = 3          # count <= 12
    run_max: int = 2          # max consecutive run


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
# Generator (simple + "informed")
# -----------------------------
def stable_seed(target_date: DateType, salt: str = "") -> int:
    feats = date_features_compact(target_date)
    blob = f"{target_date.isoformat()}|{salt}|{feats}".encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def generate_sets(
    df_history: pd.DataFrame,
    target_date: DateType,
    n_sets: int,
    alpha_white: float,
    alpha_pb: float,
    salt: str,
    max_attempts: int = 200000,
) -> pd.DataFrame:
    wc, pc = frequency_tables(df_history)
    w_weights = make_weights(wc, alpha=alpha_white)
    pb_weights = make_weights(pc, alpha=alpha_pb)

    seed = stable_seed(target_date, salt=salt)
    rng = np.random.default_rng(seed)

    seen = set()
    rows = []
    attempts = 0
    c = Constraints()

    while len(rows) < n_sets and attempts < max_attempts:
        attempts += 1

        whites = rng.choice(np.arange(1, 70), size=5, replace=False, p=w_weights).tolist()
        whites.sort()

        if not passes_constraints(whites, c):
            continue

        pb = int(rng.choice(np.arange(1, 27), p=pb_weights))

        key = (tuple(whites), pb)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "Set": len(rows) + 1,
            "Whites": " ".join(f"{n:02d}" for n in whites),
            "Powerball": f"{pb:02d}",
        })

    if len(rows) < n_sets:
        raise RuntimeError("Couldn’t generate enough sets under constraints. Try again or lower set count.")

    return pd.DataFrame(rows)


def rank_numbers_from_output(out: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    white_counts = {}
    pb_counts = {}

    for _, r in out.iterrows():
        whites = [int(x) for x in str(r["Whites"]).split()]
        pb = int(r["Powerball"])

        for w in whites:
            white_counts[w] = white_counts.get(w, 0) + 1
        pb_counts[pb] = pb_counts.get(pb, 0) + 1

    wdf = (pd.DataFrame({"White": list(white_counts.keys()), "Count": list(white_counts.values())})
             .sort_values(["Count", "White"], ascending=[False, True])
             .reset_index(drop=True))

    pdf = (pd.DataFrame({"Powerball": list(pb_counts.keys()), "Count": list(pb_counts.values())})
             .sort_values(["Count", "Powerball"], ascending=[False, True])
             .reset_index(drop=True))

    return wdf, pdf


# -----------------------------
# Streamlit UI (simple)
# -----------------------------
st.set_page_config(page_title="Swampball Simple", layout="centered")
st.title("Swampball Simple")
st.caption(
    "Pick a date → generate 20/50/100 informed sets. "
    "Then optionally scrub the output to see which numbers showed up most in *this batch*."
)

uploaded = st.file_uploader("Upload Powerball history CSV", type=["csv"])
if not uploaded:
    st.info("Upload your Powerball history CSV first.")
    st.stop()

df = load_powerball_csv(uploaded)

target_date = st.date_input("Target date", value=dt.date(2025, 12, 17))
n_sets = st.radio("How many sets?", options=[20, 50, 100], horizontal=True)
salt = st.text_input("Optional salt (keeps results stable)", value="lil_art_hoe")

with st.expander("Optional tuning (leave as-is if you want it simple)"):
    alpha_white = st.slider("White frequency bias", 0.2, 3.0, 1.1, 0.1)
    alpha_pb = st.slider("Powerball frequency bias", 0.2, 3.0, 1.1, 0.1)

gen = st.button("Generate", type="primary")

if gen:
    out = generate_sets(
        df_history=df,
        target_date=target_date,
        n_sets=int(n_sets),
        alpha_white=float(alpha_white),
        alpha_pb=float(alpha_pb),
        salt=salt.strip(),
    )

    st.subheader(f"Sets for {target_date.isoformat()}")
    st.dataframe(out, hide_index=True, use_container_width=True)

    st.download_button(
        "Download sets (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"swampball_sets_{target_date.isoformat()}_{n_sets}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Scrub / Rank numbers in this output")

    scrub = st.toggle("Show ranked counts from generated output", value=True)
    if scrub:
        wdf, pdf = rank_numbers_from_output(out)

        st.write("Most common **white balls** in this generated batch:")
        st.dataframe(wdf, hide_index=True, use_container_width=True)

        st.write("Most common **Powerballs** in this generated batch:")
        st.dataframe(pdf, hide_index=True, use_container_width=True)

        ranked = pd.concat(
            [
                wdf.assign(Type="White").rename(columns={"White": "Number"}),
                pdf.assign(Type="Powerball").rename(columns={"Powerball": "Number"}),
            ],
            ignore_index=True,
        )[["Type", "Number", "Count"]]

        st.download_button(
            "Download ranked counts (CSV)",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name=f"swampball_ranked_{target_date.isoformat()}_{n_sets}.csv",
            mime="text/csv",
        )

    with st.expander("Date gematria (matches your reference layout)"):
        layout_df = date_gematria_layout(target_date)
        st.dataframe(layout_df, hide_index=True, use_container_width=True)
