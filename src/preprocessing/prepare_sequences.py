import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/clickstream_data.csv"
PROCESSED_DIR = "data/processed"

WINDOW_SIZE        = 5       
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

    required = {"session_id", "step", "current_page"} | set(CTX_COLUMNS.keys())
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
        )

    print(f"      Rows: {len(df):,}  |  Sessions: {df['session_id'].nunique():,}")
    return df

def build_page_sequences(df: pd.DataFrame) -> list:
    df_sorted = df.sort_values(["session_id", "step"])
    sessions  = []

    for sid, group in df_sorted.groupby("session_id"):
        pages = group["current_page"].tolist()
        if len(pages) < 2:
            continue

        persona      = group["persona_type"].iloc[0]
        device       = group["device_type"].iloc[0]
        traffic      = group["traffic_source"].iloc[0]
        funnel_stage = group["funnel_stage"].iloc[0]   # ← NEW

        sessions.append((sid, pages, persona, device, traffic, funnel_stage))

    print(f"      Raw sessions (>= 2 clicks): {len(sessions):,}")
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
        if not (pages[i] == pages[i + 1] and pages[i] in SELF_LOOP_PAGES):
            return False
    return True


def has_navigation_loop(pages: list, max_revisits: int = 3) -> bool:
    from collections import Counter
    return any(v > max_revisits for v in Counter(pages).values())


def strip_terminal_tail(pages: list) -> list:
    result = list(pages)
    while (len(result) > 1
           and result[-1] in TERMINAL_PAGES
           and result[-2] in TERMINAL_PAGES):
        result.pop()
    return result


def filter_sequences(sessions: list, min_len: int) -> tuple:
    print(f"      MIN_SESSION_LENGTH : {min_len}  (= WINDOW_SIZE {WINDOW_SIZE} + 1)")

    n_orig = len(sessions)
    removed_trivial = removed_loop = removed_short = 0
    trans_before = trans_after = 0
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
        clean.append((pages, persona, device, traffic, funnel_stage))

    n_ret  = len(clean)
    n_rem  = n_orig - n_ret
    t_rem  = trans_before - trans_after

    stats = dict(
        sessions_before=n_orig, sessions_retained=n_ret, sessions_removed=n_rem,
        removed_trivial=removed_trivial, removed_loop=removed_loop,
        removed_too_short=removed_short,
        transitions_before=trans_before, transitions_after=trans_after,
    )
    return clean, stats

def build_tokenizer(clean_sessions: list) -> tuple:
    unique_pages = sorted({p for pages, *_ in clean_sessions for p in pages})
    page_to_idx  = {p: i + 1 for i, p in enumerate(unique_pages)}
    idx_to_page  = {i: p for p, i in page_to_idx.items()}
    return page_to_idx, idx_to_page

def build_feature_encoders(clean_sessions: list) -> dict:
    persona_vals      = sorted({s[1] for s in clean_sessions})
    device_vals       = sorted({s[2] for s in clean_sessions})
    traffic_vals      = sorted({s[3] for s in clean_sessions})
    funnel_stage_vals = sorted({s[4] for s in clean_sessions}) 

    def make_encoder(values: list) -> dict:
        return {v: i + 1 for i, v in enumerate(values)}

    encoders = {
        "persona"     : make_encoder(persona_vals),
        "device"      : make_encoder(device_vals),
        "traffic"     : make_encoder(traffic_vals),
        "funnel_stage": make_encoder(funnel_stage_vals),
    }

    for name, enc in encoders.items():
        print(f"      {name:15s} ({len(enc)} categories): {enc}")

    return encoders

def encode_sequences(clean_sessions: list, page_to_idx: dict,
                     encoders: dict) -> list:

    encoded = []
    for pages, persona, device, traffic, funnel_stage in clean_sessions:
        enc_pages        = [page_to_idx.get(p, 0) for p in pages]
        enc_persona      = encoders["persona"].get(persona,           0)
        enc_device       = encoders["device"].get(device,             0)
        enc_traffic      = encoders["traffic"].get(traffic,           0)
        enc_funnel_stage = encoders["funnel_stage"].get(funnel_stage, 0)  
        encoded.append((enc_pages, enc_persona, enc_device,
                        enc_traffic, enc_funnel_stage))

    orig = clean_sessions[0]
    enc  = encoded[0]
    return encoded
def create_sliding_windows(encoded_sessions: list,
                            window_size: int = WINDOW_SIZE) -> tuple:

    X_raw, y_raw                      = [], []
    persona_raw, device_raw           = [], []
    traffic_raw, funnel_stage_raw     = [], []
    samples_per_session               = []

    for enc_pages, enc_persona, enc_device, enc_traffic, enc_funnel_stage \
            in encoded_sessions:

        if len(enc_pages) <= window_size:
            continue

        count = 0
        for i in range(len(enc_pages) - window_size):
            X_raw.append(enc_pages[i : i + window_size])
            y_raw.append(enc_pages[i + window_size])
            persona_raw.append(enc_persona)
            device_raw.append(enc_device)
            traffic_raw.append(enc_traffic)
            funnel_stage_raw.append(enc_funnel_stage) 
            count += 1

        samples_per_session.append(count)

    total = len(X_raw)
    avg_s = float(np.mean(samples_per_session)) if samples_per_session else 0.0
    return X_raw, y_raw, persona_raw, device_raw, traffic_raw, funnel_stage_raw

def pad_sequences_custom(sequences: list, maxlen: int) -> np.ndarray:
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            seq = seq[-maxlen:]
        else:
            seq = [0] * (maxlen - len(seq)) + list(seq)
        padded.append(seq)
    return np.array(padded, dtype="int32")


def pad_input_sequences(X_raw: list, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    X_padded = pad_sequences_custom(X_raw, max_len)
    print(f"      X shape : {X_padded.shape}")
    print(f"      Sample  : {X_padded[0]}")
    return X_padded

def split_dataset(X, y, persona, device, traffic, funnel_stage,
                  test_size=TEST_SIZE, random_state=RANDOM_STATE):

    n = len(X)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)

    split = int(n * (1 - test_size))
    tr_idx, te_idx = idx[:split], idx[split:]

    def s(arr):
        return arr[tr_idx], arr[te_idx]

    X_tr,            X_te            = s(X)
    y_tr,            y_te            = s(y)
    persona_tr,      persona_te      = s(persona)
    device_tr,       device_te       = s(device)
    traffic_tr,      traffic_te      = s(traffic)
    funnel_stage_tr, funnel_stage_te = s(funnel_stage)   # ← NEW

    print(f"      X_train : {X_tr.shape}   X_test  : {X_te.shape}")
    print(f"      y_train : {y_tr.shape}   y_test  : {y_te.shape}")
    print(f"      persona_train/test      shapes : {persona_tr.shape} / {persona_te.shape}")
    print(f"      device_train/test       shapes : {device_tr.shape}  / {device_te.shape}")
    print(f"      traffic_train/test      shapes : {traffic_tr.shape} / {traffic_te.shape}")
    print(f"      funnel_stage_train/test shapes : "          # ← NEW
          f"{funnel_stage_tr.shape} / {funnel_stage_te.shape}")

    return (X_tr, X_te, y_tr, y_te,
            persona_tr,      persona_te,
            device_tr,       device_te,
            traffic_tr,      traffic_te,
            funnel_stage_tr, funnel_stage_te)

def save_processed_data(X_tr, X_te, y_tr, y_te,
                         persona_tr,      persona_te,
                         device_tr,       device_te,
                         traffic_tr,      traffic_te,
                         funnel_stage_tr, funnel_stage_te,   # ← NEW
                         page_to_idx, idx_to_page,
                         encoders: dict,
                         output_dir: str = PROCESSED_DIR) -> None:
    
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_te)
    np.save(os.path.join(output_dir, "y_train.npy"), y_tr)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_te)

    np.save(os.path.join(output_dir, "persona_train.npy"),      persona_tr)
    np.save(os.path.join(output_dir, "persona_test.npy"),       persona_te)
    np.save(os.path.join(output_dir, "device_train.npy"),       device_tr)
    np.save(os.path.join(output_dir, "device_test.npy"),        device_te)
    np.save(os.path.join(output_dir, "traffic_train.npy"),      traffic_tr)
    np.save(os.path.join(output_dir, "traffic_test.npy"),       traffic_te)
    np.save(os.path.join(output_dir, "funnel_stage_train.npy"), funnel_stage_tr)
    np.save(os.path.join(output_dir, "funnel_stage_test.npy"),  funnel_stage_te)

    with open(os.path.join(output_dir, "page_to_idx.json"), "w") as f:
        json.dump(page_to_idx, f, indent=2)

    idx_to_page_str = {str(k): v for k, v in idx_to_page.items()}
    with open(os.path.join(output_dir, "idx_to_page.json"), "w") as f:
        json.dump(idx_to_page_str, f, indent=2)

    for name, enc in encoders.items():
        fname = f"{name}_encoder.json"
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(enc, f, indent=2)
        print(f"      Saved: {fname}")

    config = {
        "window_size"             : WINDOW_SIZE,
        "max_seq_len"             : MAX_SEQ_LEN,
        "min_session_length"      : MIN_SESSION_LENGTH,
        "test_size"               : TEST_SIZE,
        "random_state"            : RANDOM_STATE,
        "context_features"        : list(encoders.keys()),
        "persona_vocab_size"      : max(encoders["persona"].values())      + 1,
        "device_vocab_size"       : max(encoders["device"].values())       + 1,
        "traffic_vocab_size"      : max(encoders["traffic"].values())      + 1,
        "funnel_stage_vocab_size" : max(encoders["funnel_stage"].values()) + 1,  # ← NEW
    }
    with open(os.path.join(output_dir, "preprocessing_config.json"), "w") as f:
        json.dump(config, f, indent=2)
def print_summary(X_tr, X_te, y_tr, y_te,
                  persona_tr, device_tr, traffic_tr, funnel_stage_tr,
                  page_to_idx, encoders, filter_stats,
                  window_size, max_len) -> None:

    vocab_size    = len(page_to_idx) + 1
    total_samples = len(X_tr) + len(X_te)
    non_pad       = (X_tr != 0).sum(axis=1)

    print("\n" + "=" * 60)
    print("           PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Context window (WINDOW_SIZE)      : {window_size}")
    print(f"  Padded input length (MAX_SEQ_LEN) : {max_len}")
    print(f"  Page vocabulary size (+ padding)  : {vocab_size}")
    print(f"  {'-'*46}")
    print(f"  Sessions before filtering         : {filter_stats['sessions_before']:>8,}")
    print(f"  Sessions retained                 : {filter_stats['sessions_retained']:>8,}")
    print(f"  {'-'*46}")
    print(f"  Total (X, y, ctx) samples         : {total_samples:>8,}")
    print(f"  Training samples                  : {len(X_tr):>8,}")
    print(f"  Test samples                      : {len(X_te):>8,}")
    print(f"  {'-'*46}")
    print(f"  Avg non-padding seq length        : {non_pad.mean():>8.2f}")
    print(f"  X_train shape                     : {X_tr.shape}")
    print(f"  X_test  shape                     : {X_te.shape}")
    print(f"  {'-'*46}")
    print("  CONTEXTUAL FEATURE SIZES")

    feat_arrays = {
        "persona"     : persona_tr,
        "device"      : device_tr,
        "traffic"     : traffic_tr,
        "funnel_stage": funnel_stage_tr, 
    }
    for name, enc in encoders.items():
        arr         = feat_arrays[name]
        unique_vals = np.unique(arr)
        print(f"    {name:15s}: {len(enc)} categories  "
              f"(vocab size = {max(enc.values())+1})  "
              f"sample values: {unique_vals[:5].tolist()}")

    print("=" * 60)
    print(f"  Current: WINDOW_SIZE={window_size}")


def main():
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING CONFIGURATIONS")
    print(f"{'='*60}")
    print(f"  WINDOW_SIZE        = {WINDOW_SIZE}")
    print(f"  MAX_SEQ_LEN        = {MAX_SEQ_LEN}  (auto)")
    print(f"  MIN_SESSION_LENGTH = {MIN_SESSION_LENGTH}  (auto = WINDOW_SIZE + 1)")
    print(f"  Context features   = {list(CTX_COLUMNS.keys())}")
    print(f"{'='*60}")

    df = load_data(RAW_DATA_PATH)

    sessions = build_page_sequences(df)

    clean_sessions, filter_stats = filter_sequences(sessions, MIN_SESSION_LENGTH)

    page_to_idx, idx_to_page = build_tokenizer(clean_sessions)

    encoders = build_feature_encoders(clean_sessions)

    encoded_sessions = encode_sequences(clean_sessions, page_to_idx, encoders)

    X_raw, y_raw, persona_raw, device_raw, traffic_raw, funnel_stage_raw = \
        create_sliding_windows(encoded_sessions, window_size=WINDOW_SIZE)
    
    X            = pad_input_sequences(X_raw, max_len=MAX_SEQ_LEN)
    y            = np.array(y_raw,            dtype="int32")
    persona      = np.array(persona_raw,      dtype="int32")
    device       = np.array(device_raw,       dtype="int32")
    traffic      = np.array(traffic_raw,      dtype="int32")
    funnel_stage = np.array(funnel_stage_raw, dtype="int32")

    (X_tr, X_te, y_tr, y_te,
     persona_tr,      persona_te,
     device_tr,       device_te,
     traffic_tr,      traffic_te,
     funnel_stage_tr, funnel_stage_te) = split_dataset(
        X, y, persona, device, traffic, funnel_stage
    )
    save_processed_data(
        X_tr, X_te, y_tr, y_te,
        persona_tr,      persona_te,
        device_tr,       device_te,
        traffic_tr,      traffic_te,
        funnel_stage_tr, funnel_stage_te,  
        page_to_idx, idx_to_page,
        encoders,
        output_dir=PROCESSED_DIR,
    )

    print_summary(
        X_tr, X_te, y_tr, y_te,
        persona_tr, device_tr, traffic_tr, funnel_stage_tr,
        page_to_idx, encoders, filter_stats,
        window_size=WINDOW_SIZE,
        max_len=MAX_SEQ_LEN,
    )


if __name__ == "__main__":
    main()