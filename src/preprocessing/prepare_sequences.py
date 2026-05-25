import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/clickstream_data.csv"
PROCESSED_DIR = "data/processed"

WINDOW_SIZE        = 8
MAX_SEQ_LEN        = WINDOW_SIZE
MIN_SESSION_LENGTH = WINDOW_SIZE + 1

TEST_SIZE    = 0.2
RANDOM_STATE = 42

SELF_LOOP_PAGES = {"Exit", "Home", "Support"}
TERMINAL_PAGES  = {"Exit"}

CTX_COLUMNS = {
    "persona_type"  : "persona",
    "device_type"   : "device",
    "traffic_source": "traffic",
    "funnel_stage"  : "funnel_stage",
}

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    required = {
        "session_id",
        "step",
        "current_page",
        "time_spent"
    } | set(CTX_COLUMNS.keys())

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
        )

    print(
        f"      Rows: {len(df):,}  |  "
        f"Sessions: {df['session_id'].nunique():,}"
    )

    return df

def bucket_time(t):
    if t < 30:
        return "short"
    elif t < 90:
        return "medium"
    else:
        return "long"


def build_page_sequences(df: pd.DataFrame) -> list:

    df_sorted = df.sort_values(["session_id", "step"])

    sessions = []

    for sid, group in df_sorted.groupby("session_id"):
        
        page_tokens = [
            f"{page}_{bucket_time(time_spent)}"
            for page, time_spent in zip(
                group["current_page"],
                group["time_spent"]
            )
        ]

        if len(page_tokens) < 2:
            continue

        persona      = group["persona_type"].iloc[0]
        device       = group["device_type"].iloc[0]
        traffic      = group["traffic_source"].iloc[0]
        funnel_stage = group["funnel_stage"].iloc[0]

        sessions.append((
            sid,
            page_tokens,
            persona,
            device,
            traffic,
            funnel_stage
        ))

    print(f"      Raw sessions: {len(sessions):,}")

    return sessions


def collapse_consecutive_duplicates(pages: list) -> list:
    if not pages:
        return pages

    collapsed = [pages[0]]

    for p in pages[1:]:
        if p != collapsed[-1]:
            collapsed.append(p)

    return collapsed


def is_trivial_sequence(pages: list) -> bool:

    if len(pages) < 2:
        return True

    for i in range(len(pages) - 1):
        if not (
            pages[i] == pages[i + 1]
            and pages[i] in SELF_LOOP_PAGES
        ):
            return False

    return True


def has_navigation_loop(
    pages: list,
    max_revisits: int = 3
) -> bool:

    from collections import Counter

    return any(
        v > max_revisits
        for v in Counter(pages).values()
    )


def strip_terminal_tail(pages: list) -> list:

    result = list(pages)

    while (
        len(result) > 1
        and result[-1] in TERMINAL_PAGES
        and result[-2] in TERMINAL_PAGES
    ):
        result.pop()

    return result


def filter_sequences(
    sessions: list,
    min_len: int
) -> tuple:

    print(
        f"      MIN_SESSION_LENGTH : {min_len}  "
        f"(= WINDOW_SIZE {WINDOW_SIZE} + 1)"
    )

    n_orig = len(sessions)

    removed_trivial = 0
    removed_loop    = 0
    removed_short   = 0

    trans_before = 0
    trans_after  = 0

    clean = []

    for sid, pages, persona, device, traffic, funnel_stage in sessions:

        trans_before += max(0, len(pages) - 1)

        pages = collapse_consecutive_duplicates(pages)

        pages = strip_terminal_tail(pages)

        if is_trivial_sequence(pages):
            removed_trivial += 1
            continue

        if has_navigation_loop(pages):
            removed_loop += 1
            continue

        if len(pages) < min_len:
            removed_short += 1
            continue

        trans_after += max(0, len(pages) - 1)

        clean.append((
            pages,
            persona,
            device,
            traffic,
            funnel_stage
        ))

    n_ret = len(clean)
    n_rem = n_orig - n_ret

    stats = dict(
        sessions_before=n_orig,
        sessions_retained=n_ret,
        sessions_removed=n_rem,
        removed_trivial=removed_trivial,
        removed_loop=removed_loop,
        removed_too_short=removed_short,
        transitions_before=trans_before,
        transitions_after=trans_after,
    )

    return clean, stats


def build_tokenizer(clean_sessions: list) -> tuple:

    unique_pages = sorted({
        p
        for pages, *_ in clean_sessions
        for p in pages
    })

    page_to_idx = {
        p: i + 1
        for i, p in enumerate(unique_pages)
    }

    idx_to_page = {
        i: p
        for p, i in page_to_idx.items()
    }

    return page_to_idx, idx_to_page


def build_feature_encoders(clean_sessions: list) -> dict:

    persona_vals = sorted({s[1] for s in clean_sessions})

    device_vals = sorted({s[2] for s in clean_sessions})

    traffic_vals = sorted({s[3] for s in clean_sessions})

    funnel_stage_vals = sorted({s[4] for s in clean_sessions})

    def make_encoder(values: list) -> dict:
        return {v: i + 1 for i, v in enumerate(values)}

    encoders = {
        "persona": make_encoder(persona_vals),
        "device": make_encoder(device_vals),
        "traffic": make_encoder(traffic_vals),
        "funnel_stage": make_encoder(funnel_stage_vals),
    }

    for name, enc in encoders.items():
        print(
            f"      {name:15s} "
            f"({len(enc)} categories): {enc}"
        )

    return encoders


def encode_sequences(
    clean_sessions: list,
    page_to_idx: dict,
    encoders: dict
) -> list:

    encoded = []

    for (
        pages,
        persona,
        device,
        traffic,
        funnel_stage
    ) in clean_sessions:

        enc_pages = [
            page_to_idx.get(p, 0)
            for p in pages
        ]

        enc_persona = encoders["persona"].get(persona, 0)

        enc_device = encoders["device"].get(device, 0)

        enc_traffic = encoders["traffic"].get(traffic, 0)

        enc_funnel_stage = encoders[
            "funnel_stage"
        ].get(funnel_stage, 0)

        encoded.append((
            enc_pages,
            enc_persona,
            enc_device,
            enc_traffic,
            enc_funnel_stage
        ))

    return encoded


def create_sliding_windows(
    encoded_sessions: list,
    window_size: int = WINDOW_SIZE
) -> tuple:

    X_raw, y_raw = [], []

    persona_raw = []
    device_raw = []
    traffic_raw = []
    funnel_stage_raw = []

    samples_per_session = []

    for (
        enc_pages,
        enc_persona,
        enc_device,
        enc_traffic,
        enc_funnel_stage
    ) in encoded_sessions:

        if len(enc_pages) <= window_size:
            continue

        count = 0

        for i in range(len(enc_pages) - window_size):

            X_raw.append(
                enc_pages[i : i + window_size]
            )

            y_raw.append(
                enc_pages[i + window_size]
            )

            persona_raw.append(enc_persona)
            device_raw.append(enc_device)
            traffic_raw.append(enc_traffic)
            funnel_stage_raw.append(enc_funnel_stage)

            count += 1

        samples_per_session.append(count)

    avg_s = (
        float(np.mean(samples_per_session))
        if samples_per_session
        else 0.0
    )

    return (
        X_raw,
        y_raw,
        persona_raw,
        device_raw,
        traffic_raw,
        funnel_stage_raw
    )


def pad_sequences_custom(
    sequences: list,
    maxlen: int
) -> np.ndarray:

    padded = []

    for seq in sequences:

        if len(seq) > maxlen:
            seq = seq[-maxlen:]

        else:
            seq = [0] * (
                maxlen - len(seq)
            ) + list(seq)

        padded.append(seq)

    return np.array(padded, dtype="int32")


def pad_input_sequences(
    X_raw: list,
    max_len: int = MAX_SEQ_LEN
) -> np.ndarray:

    X_padded = pad_sequences_custom(X_raw, max_len)

    print(f"      X shape : {X_padded.shape}")

    print(f"      Sample  : {X_padded[0]}")

    return X_padded


def split_dataset(
    X,
    y,
    persona,
    device,
    traffic,
    funnel_stage,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
):

    n = len(X)

    rng = np.random.default_rng(random_state)

    idx = rng.permutation(n)

    split = int(n * (1 - test_size))

    tr_idx, te_idx = idx[:split], idx[split:]

    def s(arr):
        return arr[tr_idx], arr[te_idx]

    X_tr, X_te = s(X)
    y_tr, y_te = s(y)

    persona_tr, persona_te = s(persona)

    device_tr, device_te = s(device)

    traffic_tr, traffic_te = s(traffic)

    funnel_stage_tr, funnel_stage_te = s(funnel_stage)

    return (
        X_tr, X_te,
        y_tr, y_te,
        persona_tr, persona_te,
        device_tr, device_te,
        traffic_tr, traffic_te,
        funnel_stage_tr, funnel_stage_te
    )


if __name__ == "__main__":

    print(" PREPROCESSING CLICKSTREAM DATA:")

    df = load_data(RAW_DATA_PATH)

    sessions = build_page_sequences(df)

    clean_sessions, filter_stats = filter_sequences(
        sessions,
        MIN_SESSION_LENGTH
    )

    page_to_idx, idx_to_page = build_tokenizer(
        clean_sessions
    )

    encoders = build_feature_encoders(
        clean_sessions
    )

    encoded_sessions = encode_sequences(
        clean_sessions,
        page_to_idx,
        encoders
    )

    (
        X_raw,
        y_raw,
        persona_raw,
        device_raw,
        traffic_raw,
        funnel_stage_raw
    ) = create_sliding_windows(
        encoded_sessions,
        window_size=WINDOW_SIZE
    )

    X = pad_input_sequences(
        X_raw,
        max_len=MAX_SEQ_LEN
    )

    y = np.array(y_raw, dtype="int32")

    persona = np.array(persona_raw, dtype="int32")

    device = np.array(device_raw, dtype="int32")

    traffic = np.array(traffic_raw, dtype="int32")

    funnel_stage = np.array(
        funnel_stage_raw,
        dtype="int32"
    )

    (
        X_tr, X_te,
        y_tr, y_te,
        persona_tr, persona_te,
        device_tr, device_te,
        traffic_tr, traffic_te,
        funnel_stage_tr, funnel_stage_te
    ) = split_dataset(
        X,
        y,
        persona,
        device,
        traffic,
        funnel_stage
    )
    
    print(f"X_train shape : {X_tr.shape}")
    print(f"X_test shape  : {X_te.shape}")

    print(f"Vocabulary size : {len(page_to_idx)+1}")