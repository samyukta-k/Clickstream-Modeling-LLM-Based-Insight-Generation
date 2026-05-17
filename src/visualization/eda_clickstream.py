import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/clickstream_data.csv"
CHART_DIR = "outputs/charts"
os.makedirs(CHART_DIR, exist_ok=True)

PAGES_ORDER = ["Home", "Search", "Product", "Cart", "Checkout", "Payment", "Profile", "Support", "Exit"]

PALETTE_MAIN   = "#2D6A9F"
PALETTE_ACCENT = "#E07B39"
PALETTE_GREEN  = "#3AAB5C"
PALETTE_RED    = "#D94F4F"
PALETTE_SEQ    = "Blues"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titleweight":  "bold",
    "axes.titlesize":    13,
})

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df

def save_fig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_top_visited_pages(df: pd.DataFrame) -> None:
    counts = df["current_page"].value_counts().reindex(PAGES_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE_MAIN, edgecolor="white", width=0.65)
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
    ax.set_title("Top Visited Pages")
    ax.set_xlabel("Page")
    ax.set_ylabel("Visit Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save_fig(fig, "eda_01_top_visited_pages.png")

def plot_device_distribution(df: pd.DataFrame) -> None:
    per_session = df.drop_duplicates("session_id")[["session_id", "device_type"]]
    counts = per_session["device_type"].value_counts()

    colours = [PALETTE_MAIN, PALETTE_ACCENT, PALETTE_GREEN]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index,
        colors=colours[:len(counts)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"width": 0.55, "edgecolor": "white"},
        textprops={"fontsize": 11},
    )
    ax.set_title("Device Type Distribution\n(per session)", pad=20)
    save_fig(fig, "eda_02_device_distribution.png")

def plot_traffic_source(df: pd.DataFrame) -> None:
    per_session = df.drop_duplicates("session_id")[["session_id", "traffic_source"]]
    counts = per_session["traffic_source"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = sns.color_palette("Blues_r", len(counts))
    bars = ax.barh(counts.index, counts.values, color=colours, edgecolor="white")
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
    ax.set_title("Traffic Source Distribution")
    ax.set_xlabel("Number of Sessions")
    ax.invert_yaxis()
    save_fig(fig, "eda_03_traffic_source_distribution.png")

def plot_conversion_funnel(df: pd.DataFrame) -> None:
    funnel_pages = ["Home", "Search", "Product", "Cart", "Checkout", "Payment"]

    sessions_reached = {}
    for page in funnel_pages:
        n = df[df["current_page"] == page]["session_id"].nunique()
        sessions_reached[page] = n

    pages  = list(sessions_reached.keys())
    values = list(sessions_reached.values())
    top    = values[0]
    pcts   = [v / top * 100 for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = sns.color_palette("Blues", len(pages))[::-1]
    bars = ax.barh(pages[::-1], values[::-1], color=colours, edgecolor="white", height=0.55)

    for bar, pct, val in zip(bars, pcts[::-1], values[::-1]):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_title("Conversion Funnel  (sessions that reached each stage)")
    ax.set_xlabel("Unique Sessions")
    ax.set_xlim(0, max(values) * 1.22)
    save_fig(fig, "eda_04_conversion_funnel.png")

    print("Funnel Drop offs:")
    for i in range(1, len(pages)):
        drop = (values[i - 1] - values[i]) / values[i - 1] * 100
        print(f"     {pages[i-1]:10s} -> {pages[i]:10s}  drop-off: {drop:.1f}%")
    print()

def plot_bounce_rate(df: pd.DataFrame) -> None:
    per_session = df.groupby("session_id").agg(
        traffic_source  = ("traffic_source",  "first"),
        bounce_flag     = ("bounce_flag",      "max"),
        conversion_flag = ("conversion_flag",  "max"),
    ).reset_index()

    summary = per_session.groupby("traffic_source").agg(
        bounce_rate     = ("bounce_flag",      "mean"),
        conversion_rate = ("conversion_flag",  "mean"),
    ).reset_index().sort_values("bounce_rate", ascending=False)

    x   = np.arange(len(summary))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, summary["bounce_rate"]     * 100, width=w,
           label="Bounce Rate",     color=PALETTE_RED,   edgecolor="white", alpha=0.88)
    ax.bar(x + w / 2, summary["conversion_rate"] * 100, width=w,
           label="Conversion Rate", color=PALETTE_GREEN, edgecolor="white", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["traffic_source"], rotation=20, ha="right")
    ax.set_title("Bounce Rate vs Conversion Rate by Traffic Source")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    save_fig(fig, "eda_05_bounce_rate_by_source.png")

def plot_transition_heatmap(df: pd.DataFrame) -> None:
    transitions = df[df["next_page"] != "Exit"].copy()
    pivot = (
        transitions
        .groupby(["current_page", "next_page"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=PAGES_ORDER, columns=PAGES_ORDER, fill_value=0)
    )

    log_pivot = np.log1p(pivot)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        log_pivot, annot=pivot.values, fmt=",",
        cmap="Blues", linewidths=0.5, linecolor="#ddd",
        ax=ax, cbar_kws={"label": "log(1 + count)"},
        annot_kws={"size": 8},
    )
    ax.set_title("Page Transition Heatmap")
    ax.set_xlabel("Next Page")
    ax.set_ylabel("Current Page")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    save_fig(fig, "eda_06_transition_heatmap.png")

def plot_session_length(df: pd.DataFrame) -> None:
    session_len = df.groupby("session_id")["step"].max().reset_index()
    session_len.columns = ["session_id", "steps"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(session_len["steps"], bins=range(1, session_len["steps"].max() + 2),
                 color=PALETTE_MAIN, edgecolor="white", alpha=0.85, ax=ax, stat="count")

    mean_steps = session_len["steps"].mean()
    ax.axvline(mean_steps, color=PALETTE_ACCENT, linestyle="--", linewidth=2,
               label=f"Mean: {mean_steps:.1f} steps")

    ax.set_title("Session Length Distribution")
    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Sessions")
    ax.legend()
    save_fig(fig, "eda_07_session_length_distribution.png")
    
def plot_avg_time_per_page(df: pd.DataFrame) -> None:
    avg_time = (
        df[df["current_page"] != "Exit"]
        .groupby("current_page")["time_spent"]
        .mean()
        .reindex([p for p in PAGES_ORDER if p != "Exit"])
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = sns.color_palette("Blues", len(avg_time))
    bars = ax.barh(avg_time.index, avg_time.values, color=colours, edgecolor="white")
    ax.bar_label(bars, fmt="{:.1f}s", padding=4, fontsize=9)
    ax.set_title("Average Time Spent per Page")
    ax.set_xlabel("Avg Time (seconds)")
    save_fig(fig, "eda_08_avg_time_per_page.png")

    top_page = avg_time.idxmax()
    print(f"Highest dwell time: {top_page} ({avg_time[top_page]:.1f}s avg)\n")

def plot_conversion_paths(df: pd.DataFrame) -> None:
    converted_sessions = df[df["conversion_flag"] == 1]["session_id"].unique()
    conv_df = df[df["session_id"].isin(converted_sessions)].copy()

    conv_df = conv_df.sort_values(["session_id", "step"])
    paths = []
    for _, grp in conv_df.groupby("session_id"):
        pages = grp["current_page"].tolist()
        for i in range(len(pages) - 2):
            paths.append((pages[i], pages[i + 1], pages[i + 2]))

    from collections import Counter
    path_counts = Counter(paths)
    top_paths   = path_counts.most_common(20)

    all_nodes = list(PAGES_ORDER)
    node_idx  = {p: i for i, p in enumerate(all_nodes)}

    link_source, link_target, link_value = [], [], []

    pair_counts: dict = {}
    for (a, b, c), cnt in top_paths:
        pair_counts[(a, b)] = pair_counts.get((a, b), 0) + cnt
        pair_counts[(b, c)] = pair_counts.get((b, c), 0) + cnt

    for (src, tgt), val in pair_counts.items():
        if src in node_idx and tgt in node_idx:
            link_source.append(node_idx[src])
            link_target.append(node_idx[tgt])
            link_value.append(val)

    colour_nodes = [
        "#2D6A9F", "#3788C2", "#57A0D3", "#5BA85A", "#E07B39",
        "#C0392B", "#8E44AD", "#E67E22", "#95A5A6",
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=22,
            line=dict(color="white", width=0.5),
            label=all_nodes,
            color=colour_nodes,
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            color="rgba(45,106,159,0.25)",
        ),
    ))
    fig.update_layout(
        title_text="Top Navigation Paths in Converted Sessions",
        font_size=12,
        width=900, height=520,
        paper_bgcolor="white",
    )

    path = os.path.join(CHART_DIR, "eda_09_conversion_paths.png")
    try:
        fig.write_image(path)
    except Exception as e:
        html_path = path.replace(".png", ".html")
        fig.write_html(html_path)
        print(f"kaleido not available - saved as HTML: {html_path}  ({e})")

def print_ux_insights(df: pd.DataFrame) -> None:
    per_session = df.groupby("session_id").agg(
        conversion = ("conversion_flag", "max"),
        bounce     = ("bounce_flag",     "max"),
        steps      = ("step",            "max"),
        device     = ("device_type",     "first"),
        source     = ("traffic_source",  "first"),
    ).reset_index()

    overall_cvr    = per_session["conversion"].mean() * 100
    overall_bounce = per_session["bounce"].mean()     * 100

    device_bounce = (per_session.groupby("device")["bounce"]
                     .mean() * 100).sort_values(ascending=False)
    
    source_cvr = (per_session.groupby("source")["conversion"]
                  .mean() * 100).sort_values(ascending=False)
    
    exit_df = df[df["next_page"] == "Exit"]
    drop_off = (exit_df.groupby("current_page").size() /
                df.groupby("current_page").size() * 100).sort_values(ascending=False)
    drop_off = drop_off[drop_off.index != "Exit"].dropna()

    print("=" * 58)
    print("  UX INSIGHTS")
    print("=" * 58)
    print(f"\nOverall conversion rate : {overall_cvr:.2f}%")
    print(f"Overall bounce rate     : {overall_bounce:.2f}%\n")

    print("Bounce Rate by Device:")
    for dev, rate in device_bounce.items():
        print(f"     {dev:8s}  ->  {rate:.1f}%")

    print(f"\nBest-converting traffic sources:")
    for src, rate in source_cvr.head(3).items():
        print(f"     {src:10s}  ->  {rate:.1f}% CVR")

    print(f"\n  High Drop-off Pages (% of visits that exit):")
    for page, rate in drop_off.head(5).items():
        flag = "ATTENTION" if rate > 40 else ""
        print(f"     {page:10s}  ->  {rate:.1f}%{flag}")

    print(f"\nConversion Funnel – sessions that reached Payment:")
    payment_sessions = df[df["current_page"] == "Payment"]["session_id"].nunique()
    total_sessions   = df["session_id"].nunique()
    print(f"     {payment_sessions:,} / {total_sessions:,} sessions  "
          f"({payment_sessions/total_sessions*100:.2f}%)")

    print("\n" + "=" * 58 + "\n")

def main():
    df = load_data(DATA_PATH)

    print_ux_insights(df)

    plot_top_visited_pages(df)
    plot_device_distribution(df)
    plot_traffic_source(df)
    plot_conversion_funnel(df)
    plot_bounce_rate(df)
    plot_transition_heatmap(df)
    plot_session_length(df)
    plot_avg_time_per_page(df)
    plot_conversion_paths(df)


if __name__ == "__main__":
    main()