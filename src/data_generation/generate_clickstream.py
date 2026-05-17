import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

RANDOM_SEED  = 42
NUM_SESSIONS = 50000
NUM_USERS    = 2000
START_DATE   = datetime(2024, 1, 1)
END_DATE     = datetime(2024, 6, 30)

OUTPUT_CSV = "data/raw/clickstream_data.csv"
CHART_DIR  = "outputs/charts"

PAGES = ["Home", "Search", "Product", "Cart",
         "Checkout", "Payment", "Profile", "Support", "Exit"]

DEVICE_TYPES    = ["desktop", "mobile", "tablet"]
DEVICE_WEIGHTS  = [0.50, 0.40, 0.10]

TRAFFIC_SOURCES = ["organic", "paid", "social", "email", "direct", "referral"]
TRAFFIC_WEIGHTS = [0.30, 0.20, 0.20, 0.10, 0.15, 0.05]

TIME_SPENT_PARAMS = {
    "Home":     (45,  20),
    "Search":   (60,  25),
    "Product":  (90,  40),
    "Cart":     (50,  20),
    "Checkout": (120, 45),
    "Payment":  (100, 35),
    "Profile":  (70,  30),
    "Support":  (180, 60),
    "Exit":     (5,   2),
}

FUNNELS = {
    "Buyer": {
        "weight":       0.25,
        "steps": [
            ("Home",     "discovery"),
            ("Search",   "discovery"),
            ("Product",  "evaluation"),
            ("Cart",     "evaluation"),
            ("Checkout", "purchase"),
            ("Payment",  "purchase"),
            ("Exit",     "exit"),
        ],
        "advance_prob": 0.68,
        "loop_prob": 0.08,
        "backstep_prob": 0.10,
        "noise_prob": 0.05,
    },

    "Casual Browser": {
        "weight":       0.30,
        "steps": [
            ("Home",    "discovery"),
            ("Search",  "discovery"),
            ("Product", "evaluation"),
            ("Exit",    "exit"),
        ],
        "advance_prob": 0.60,
        "loop_prob": 0.15,
        "backstep_prob": 0.12,
        "noise_prob": 0.05,
    },

    "Returning User": {
        "weight":       0.20,
        "steps": [
            ("Home",     "discovery"),
            ("Profile",  "discovery"),
            ("Product",  "evaluation"),
            ("Cart",     "evaluation"),
            ("Checkout", "purchase"),
            ("Exit",     "exit"),
        ],
       "advance_prob": 0.70,
        "loop_prob": 0.06,
        "backstep_prob": 0.08,
        "noise_prob": 0.05,
    },

    "Support-Seeking User": {
        "weight":       0.13,
        "steps": [
            ("Home",    "discovery"),
            ("Product", "evaluation"),
            ("Support", "support"),
            ("Exit",    "exit"),
        ],
        "advance_prob": 0.65,
        "loop_prob": 0.10,
        "backstep_prob": 0.10,
        "noise_prob": 0.05,
    },

    "Frustrated User": {
        "weight":       0.12,
        "steps": [
            ("Home",    "discovery"),
            ("Product", "evaluation"),
            ("Home",    "discovery"), 
            ("Search",  "discovery"),
            ("Exit",    "exit"),
        ],
        "advance_prob": 0.55,
        "loop_prob": 0.15,
        "backstep_prob": 0.15,
        "noise_prob": 0.08,
    },
}

_total_weight = sum(f["weight"] for f in FUNNELS.values())
assert abs(_total_weight - 1.0) < 1e-6, (
    f"Persona weights sum to {_total_weight:.6f}, expected 1.0"
)

_PERSONA_NAMES   = list(FUNNELS.keys())
_PERSONA_WEIGHTS = [FUNNELS[p]["weight"] for p in _PERSONA_NAMES]

def get_funnel_stage(persona_name: str, funnel_pos: int) -> str:
    steps = FUNNELS[persona_name]["steps"]
    pos   = min(funnel_pos, len(steps) - 1)
    return steps[pos][1]


def get_funnel_page(persona_name: str, funnel_pos: int) -> str:
    steps = FUNNELS[persona_name]["steps"]
    pos   = min(funnel_pos, len(steps) - 1)
    return steps[pos][0]

def _pick_action(persona_name: str, funnel_pos: int, max_funnel_pos: int) -> str:
    cfg          = FUNNELS[persona_name]
    depth_ratio  = funnel_pos / max(max_funnel_pos, 1)
    exit_bonus   = depth_ratio * 0.04         

    adv  = cfg["advance_prob"]
    loop = cfg["loop_prob"]
    back = cfg["backstep_prob"]
    noi  = cfg["noise_prob"]
    base_exit = max(0.0, 1.0 - adv - loop - back - noi)
    ext  = base_exit + exit_bonus

    total = adv + loop + back + noi + ext
    r = random.random() * total

    if r < adv:
        return "advance"
    r -= adv
    if r < loop:
        return "loop"
    r -= loop
    if r < back:
        return "backstep"
    r -= back
    if r < noi:
        return "noise"
    return "exit"


def select_next_page(current_page: str, persona_name: str,
                     funnel_pos: int) -> tuple[str, int]:
    steps         = FUNNELS[persona_name]["steps"]
    max_pos       = len(steps) - 1
    action        = _pick_action(persona_name, funnel_pos, max_pos)

    if action == "advance":
        new_pos   = min(funnel_pos + 1, max_pos)
        next_page = steps[new_pos][0]

    elif action == "loop":
        new_pos   = funnel_pos
        next_page = current_page    

    elif action == "backstep":
        new_pos   = max(funnel_pos - 1, 0)
        next_page = steps[new_pos][0]

    elif action == "noise":
        noise_pool = [p for p in PAGES if p != "Exit"]
        next_page  = random.choice(noise_pool)
        new_pos    = funnel_pos     

    else:
        next_page = "Exit"
        new_pos   = max_pos    

    return next_page, new_pos


def sample_time_spent(page: str) -> float:
    mu, sigma = TIME_SPENT_PARAMS[page]
    return max(2.0, round(np.random.normal(mu, sigma), 1))

def assign_persona() -> str:
    return random.choices(_PERSONA_NAMES, weights=_PERSONA_WEIGHTS, k=1)[0]

def generate_session(session_id: int, user_id: int,
                     session_start: datetime) -> list[dict]:
    persona_name   = assign_persona()
    device_type    = random.choices(DEVICE_TYPES,   weights=DEVICE_WEIGHTS,   k=1)[0]
    traffic_source = random.choices(TRAFFIC_SOURCES, weights=TRAFFIC_WEIGHTS, k=1)[0]

    steps_list      = FUNNELS[persona_name]["steps"]
    current_page    = steps_list[0][0] 
    funnel_pos      = 0
    current_time    = session_start
    step            = 1
    rows            = []
    max_steps       = 30             
    reached_payment = False

    while step <= max_steps:
        next_page, new_funnel_pos = select_next_page(
            current_page, persona_name, funnel_pos
        )
        time_spent   = sample_time_spent(current_page)
        funnel_stage = get_funnel_stage(persona_name, funnel_pos)
        bounce_flag  = 1 if (step == 1 and next_page == "Exit") else 0

        if current_page == "Payment":
            reached_payment = True

        rows.append({
            "session_id":      session_id,
            "user_id":         user_id,
            "timestamp":       current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "step":            step,
            "current_page":    current_page,
            "next_page":       next_page,
            "time_spent":      time_spent,
            "device_type":     device_type,
            "traffic_source":  traffic_source,
            "persona_type":    persona_name,
            "funnel_stage":    funnel_stage,  
            "bounce_flag":     bounce_flag,
            "conversion_flag": 0,             
        })

        current_time += timedelta(seconds=time_spent)
        step         += 1
        funnel_pos    = new_funnel_pos

        if next_page == "Exit":
            break

        current_page = next_page

    if reached_payment:
        for row in rows:
            row["conversion_flag"] = 1

    return rows

def create_dataset(num_sessions: int, num_users: int) -> pd.DataFrame:
    all_rows = []
    user_ids = list(range(1, num_users + 1))

    for session_id in range(1, num_sessions + 1):
        user_id = random.choice(user_ids)

        delta_seconds = int((END_DATE - START_DATE).total_seconds())
        session_start = START_DATE + timedelta(
            seconds=random.randint(0, delta_seconds)
        )

        session_rows = generate_session(session_id, user_id, session_start)
        all_rows.extend(session_rows)

    df = pd.DataFrame(all_rows)
    df.sort_values(["session_id", "step"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def plot_session_length_distribution(df: pd.DataFrame, save_dir: str) -> None:
    session_lengths = df.groupby("session_id")["step"].max()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(session_lengths,
            bins=range(1, int(session_lengths.max()) + 2),
            color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title("Session Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Steps in Session")
    ax.set_ylabel("Number of Sessions")
    ax.set_xlim(left=1)

    mean_len = session_lengths.mean()
    ax.axvline(mean_len, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_len:.1f} steps")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "session_length_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_conversion_distribution(df: pd.DataFrame, save_dir: str) -> None:
    conv_per_session = df.groupby("session_id")["conversion_flag"].max()
    counts = conv_per_session.value_counts()
    labels = ["Not Converted", "Converted"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ["#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=140, textprops={"fontsize": 12})
    ax.set_title("Session Conversion Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "conversion_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_top_navigation_paths(df: pd.DataFrame, save_dir: str,
                               top_n: int = 15) -> None:
    df2 = df.copy()
    df2["path"] = df2["current_page"] + "  ->  " + df2["next_page"]
    path_counts = Counter(df2["path"])
    top_paths   = path_counts.most_common(top_n)
    paths, counts = zip(*top_paths)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(paths[::-1], counts[::-1],
                   color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(f"Top {top_n} Navigation Paths", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency")

    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 20,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "top_navigation_paths.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_persona_distribution(df: pd.DataFrame, save_dir: str) -> None:
    session_df = df.groupby("session_id").agg(
        persona_type=("persona_type", "first"),
        converted=("conversion_flag", "max"),
    ).reset_index()

    persona_counts    = session_df.groupby("persona_type").size()
    persona_converted = session_df.groupby("persona_type")["converted"].sum()
    persona_names     = list(FUNNELS.keys())

    counts    = [persona_counts.get(p, 0)    for p in persona_names]
    converted = [persona_converted.get(p, 0) for p in persona_names]
    not_conv  = [c - cv for c, cv in zip(counts, converted)]

    x     = np.arange(len(persona_names))
    width = 0.55

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, not_conv,  width, label="Not Converted", color="#DD8452", alpha=0.85)
    ax.bar(x, converted, width, bottom=not_conv,
           label="Converted", color="#55A868", alpha=0.85)

    for i, (nc, cv) in enumerate(zip(not_conv, converted)):
        total = nc + cv
        if total > 0:
            rate = cv / total * 100
            ax.text(i, total + 15, f"{rate:.1f}%",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(persona_names, fontsize=9)
    ax.set_title("Session Count and Conversion Rate by Persona",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Sessions")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "persona_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()

def _build_funnel_session_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("session_id").agg(
        persona_type=("persona_type", "first"),
        converted=("conversion_flag", "max"),
        max_step=("step", "max"),
    ).reset_index()


def plot_funnel_completion_rates(df: pd.DataFrame, save_dir: str) -> None:
    stages = ["discovery", "evaluation", "purchase", "support", "exit"]
    stage_colors = {
        "discovery":  "#4C72B0",
        "evaluation": "#DD8452",
        "purchase":   "#55A868",
        "support":    "#C44E52",
        "exit":       "#8172B2",
    }

    persona_names = list(FUNNELS.keys())
    n_personas    = len(persona_names)

    stage_reached = df.groupby(["session_id", "persona_type"])["funnel_stage"].apply(
        lambda s: set(s)
    ).reset_index()
    stage_reached.columns = ["session_id", "persona_type", "stages_reached"]

    fig, ax = plt.subplots(figsize=(13, 6))
    bar_width = 0.15
    x = np.arange(n_personas)

    for si, stage in enumerate(stages):
        rates = []
        for persona in persona_names:
            subset = stage_reached[stage_reached["persona_type"] == persona]
            if len(subset) == 0:
                rates.append(0)
                continue
            reached = subset["stages_reached"].apply(lambda s: stage in s).sum()
            rates.append(reached / len(subset) * 100)

        offset = (si - len(stages) / 2 + 0.5) * bar_width
        ax.bar(x + offset, rates, bar_width,
               label=stage, color=stage_colors[stage], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(persona_names, fontsize=9)
    ax.set_ylabel("Sessions Reaching Stage (%)")
    ax.set_title("Funnel Stage Completion Rate by Persona",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(title="Funnel Stage", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    path = os.path.join(save_dir, "funnel_completion_rates.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_funnel_dropout_heatmap(df: pd.DataFrame, save_dir: str) -> None:
    persona_names = list(FUNNELS.keys())
    pages_no_exit = [p for p in PAGES if p != "Exit"]

    matrix = np.zeros((len(persona_names), len(pages_no_exit)))

    for pi, persona in enumerate(persona_names):
        pdata = df[df["persona_type"] == persona]
        for ci, page in enumerate(pages_no_exit):
            page_rows = pdata[pdata["current_page"] == page]
            if len(page_rows) == 0:
                matrix[pi, ci] = np.nan
                continue
            dropout = (page_rows["next_page"] == "Exit").sum()
            matrix[pi, ci] = dropout / len(page_rows) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    masked = np.ma.masked_invalid(matrix)
    cmap   = plt.cm.YlOrRd
    cmap.set_bad(color="#eeeeee")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=80)
    ax.set_xticks(range(len(pages_no_exit)))
    ax.set_xticklabels(pages_no_exit, fontsize=10)
    ax.set_yticks(range(len(persona_names)))
    ax.set_yticklabels(persona_names, fontsize=10)
    ax.set_title("Dropout Rate by Persona x Page",
                 fontsize=14, fontweight="bold")

    for pi in range(len(persona_names)):
        for ci in range(len(pages_no_exit)):
            val = matrix[pi, ci]
            if not np.isnan(val):
                ax.text(ci, pi, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if val > 45 else "black")

    plt.colorbar(im, ax=ax, label="Dropout Rate (%)")
    plt.tight_layout()

    path = os.path.join(save_dir, "funnel_dropout_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_avg_funnel_depth(df: pd.DataFrame, save_dir: str) -> None:
    session_df = df.groupby("session_id").agg(
        persona_type=("persona_type", "first"),
        converted=("conversion_flag", "max"),
        depth=("step", "max"),
    ).reset_index()

    persona_names = list(FUNNELS.keys())
    conv_means    = []
    noconv_means  = []

    for persona in persona_names:
        sub  = session_df[session_df["persona_type"] == persona]
        conv = sub[sub["converted"] == 1]["depth"].mean()
        nc   = sub[sub["converted"] == 0]["depth"].mean()
        conv_means.append(conv if not np.isnan(conv) else 0)
        noconv_means.append(nc  if not np.isnan(nc)  else 0)

    y = np.arange(len(persona_names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y + height / 2, conv_means,   height,
            label="Converted",     color="#55A868", alpha=0.85)
    ax.barh(y - height / 2, noconv_means, height,
            label="Not Converted", color="#DD8452", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(persona_names, fontsize=10)
    ax.set_xlabel("Average Funnel Depth (steps)")
    ax.set_title("Average Funnel Depth by Persona and Conversion",
                 fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "avg_funnel_depth.png")
    plt.savefig(path, dpi=150)
    plt.close()

def print_summary(df: pd.DataFrame) -> None:
    n_sessions  = df["session_id"].nunique()
    n_users     = df["user_id"].nunique()
    n_rows      = len(df)
    conv_rate   = df.groupby("session_id")["conversion_flag"].max().mean() * 100
    bounce_rate = df.groupby("session_id")["bounce_flag"].max().mean() * 100
    avg_len     = df.groupby("session_id")["step"].max().mean()

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows       : {n_rows:,}")
    print(f"  Sessions         : {n_sessions:,}")
    print(f"  Unique users     : {n_users:,}")
    print(f"  Avg session len  : {avg_len:.2f} steps")
    print(f"  Conversion rate  : {conv_rate:.2f}%")
    print(f"  Bounce rate      : {bounce_rate:.2f}%")

    print("\n  USER PERSONA SUMMARY")
    print("  " + "-" * 56)
    print(f"  {'Persona':<22} {'Sessions':>9} {'Conv':>7} {'Conv%':>7} {'AvgDepth':>9}")
    print("  " + "-" * 56)

    session_df = df.groupby("session_id").agg(
        persona_type=("persona_type", "first"),
        converted=("conversion_flag", "max"),
        depth=("step", "max"),
    )

    for persona in FUNNELS:
        sub   = session_df[session_df["persona_type"] == persona]
        n     = len(sub)
        conv  = int(sub["converted"].sum())
        pct   = conv / n * 100 if n > 0 else 0.0
        depth = sub["depth"].mean()
        print(f"  {persona:<22} {n:>9,} {conv:>7,} {pct:>6.1f}% {depth:>9.2f}")

    print("=" * 60)
    print("\n  FUNNEL STAGE COVERAGE")
    print("  " + "-" * 40)
    stage_counts = df["funnel_stage"].value_counts()
    for stage, cnt in stage_counts.items():
        pct = cnt / len(df) * 100
        print(f"  {stage:<14} {cnt:>9,}  ({pct:.1f}%)")
    print("=" * 60)

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df = create_dataset(NUM_SESSIONS, NUM_USERS)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset saved: ({len(df):,} rows)")

    print_summary(df)

    os.makedirs(CHART_DIR, exist_ok=True)

    plot_session_length_distribution(df, CHART_DIR)
    plot_conversion_distribution(df, CHART_DIR)
    plot_top_navigation_paths(df, CHART_DIR)
    plot_persona_distribution(df, CHART_DIR)

    plot_funnel_completion_rates(df, CHART_DIR)
    plot_funnel_dropout_heatmap(df, CHART_DIR)
    plot_avg_funnel_depth(df, CHART_DIR)


if __name__ == "__main__":
    main()