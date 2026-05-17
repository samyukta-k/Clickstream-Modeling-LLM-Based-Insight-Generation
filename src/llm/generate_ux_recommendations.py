import json
import os
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import InferenceClient

DATA_DIR        = Path("data/processed")
MODEL_PATH      = Path("outputs/models/lstm_next_click.pt")
PROMPT_TEMPLATE = Path("prompts/ux_prompt.txt")
REPORT_PATH     = Path("outputs/reports/ux_recommendations.txt")

HF_TOKEN = os.environ.get("HF_TOKEN")

HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct:cerebras"

NUM_SAMPLES        = 3 
TOP_K              = 3
LLM_MAX_NEW_TOKENS = 512

PAGE_EMBEDDING_DIM         = 32
HIDDEN_DIM                 = 64
NUM_LAYERS                 = 1
DROPOUT                    = 0.5
PERSONA_EMBEDDING_DIM      = 8
DEVICE_EMBEDDING_DIM       = 8
TRAFFIC_EMBEDDING_DIM      = 8
FUNNEL_STAGE_EMBEDDING_DIM = 8

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

class BiLSTMNextClickWithContext(nn.Module):
    def __init__(self, page_vocab_size, persona_vocab_size, device_vocab_size,
                 traffic_vocab_size, funnel_stage_vocab_size,
                 page_emb_dim=32, hidden_dim=64, num_layers=1,
                 persona_emb_dim=8, device_emb_dim=8,
                 traffic_emb_dim=8, funnel_stage_emb_dim=8, dropout=0.5):
        super().__init__()
        self.page_embedding = nn.Embedding(page_vocab_size, page_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=page_emb_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.bilstm_output_dim = hidden_dim * 2
        self.persona_embedding      = nn.Embedding(persona_vocab_size,      persona_emb_dim,      padding_idx=0)
        self.device_embedding       = nn.Embedding(device_vocab_size,       device_emb_dim,       padding_idx=0)
        self.traffic_embedding      = nn.Embedding(traffic_vocab_size,      traffic_emb_dim,      padding_idx=0)
        self.funnel_stage_embedding = nn.Embedding(funnel_stage_vocab_size, funnel_stage_emb_dim, padding_idx=0)
        combined_dim = (self.bilstm_output_dim + persona_emb_dim
                        + device_emb_dim + traffic_emb_dim + funnel_stage_emb_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_dim, page_vocab_size)

    def forward(self, x_pages, x_persona, x_device, x_traffic, x_funnel_stage):
        page_emb    = self.page_embedding(x_pages)
        lstm_out, _ = self.lstm(page_emb)
        last_hidden = lstm_out[:, -1, :]
        p_emb = self.persona_embedding(x_persona)
        d_emb = self.device_embedding(x_device)
        t_emb = self.traffic_embedding(x_traffic)
        f_emb = self.funnel_stage_embedding(x_funnel_stage)
        combined = torch.cat([last_hidden, p_emb, d_emb, t_emb, f_emb], dim=1)
        return self.classifier(self.dropout(combined))

def load_mappings(data_dir):
    def jload(name):
        with open(data_dir / f"{name}.json") as f:
            return json.load(f)

    page_to_idx          = jload("page_to_idx")
    idx_to_page          = {int(k): v for k, v in jload("idx_to_page").items()}
    persona_encoder      = jload("persona_encoder")
    device_encoder       = jload("device_encoder")
    traffic_encoder      = jload("traffic_encoder")
    funnel_stage_encoder = jload("funnel_stage_encoder")

    mappings = {
        "page_to_idx": page_to_idx, "idx_to_page": idx_to_page,
        "persona_encoder": persona_encoder, "device_encoder": device_encoder,
        "traffic_encoder": traffic_encoder, "funnel_stage_encoder": funnel_stage_encoder,
        "rev_persona"     : {v: k for k, v in persona_encoder.items()},
        "rev_device"      : {v: k for k, v in device_encoder.items()},
        "rev_traffic"     : {v: k for k, v in traffic_encoder.items()},
        "rev_funnel_stage": {v: k for k, v in funnel_stage_encoder.items()},
        "vocab_size"              : len(page_to_idx) + 1,
        "persona_vocab_size"      : max(persona_encoder.values())      + 1,
        "device_vocab_size"       : max(device_encoder.values())       + 1,
        "traffic_vocab_size"      : max(traffic_encoder.values())      + 1,
        "funnel_stage_vocab_size" : max(funnel_stage_encoder.values()) + 1,
    }
    print(f"  Page vocab: {mappings['vocab_size']}  "
          f"Persona: {mappings['persona_vocab_size']}  "
          f"Device: {mappings['device_vocab_size']}  "
          f"Traffic: {mappings['traffic_vocab_size']}  "
          f"Funnel: {mappings['funnel_stage_vocab_size']}")
    return mappings


def load_test_samples(data_dir, n):
    def npy(name): return np.load(data_dir / f"{name}.npy")[:n]
    return {
        "X": npy("X_test"), "y": npy("y_test"),
        "persona": npy("persona_test"), "device": npy("device_test"),
        "traffic": npy("traffic_test"), "funnel_stage": npy("funnel_stage_test"),
    }

def load_bilstm(model_path, mappings, device):
    model = BiLSTMNextClickWithContext(
        page_vocab_size=mappings["vocab_size"],
        persona_vocab_size=mappings["persona_vocab_size"],
        device_vocab_size=mappings["device_vocab_size"],
        traffic_vocab_size=mappings["traffic_vocab_size"],
        funnel_stage_vocab_size=mappings["funnel_stage_vocab_size"],
        page_emb_dim=PAGE_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, persona_emb_dim=PERSONA_EMBEDDING_DIM,
        device_emb_dim=DEVICE_EMBEDDING_DIM, traffic_emb_dim=TRAFFIC_EMBEDDING_DIM,
        funnel_stage_emb_dim=FUNNEL_STAGE_EMBEDDING_DIM, dropout=DROPOUT,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  BiLSTM loaded — {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def run_bilstm_inference(model, samples, mappings, device, top_k=3):
    X_t  = torch.tensor(samples["X"],            dtype=torch.long).to(device)
    P_t  = torch.tensor(samples["persona"],      dtype=torch.long).to(device)
    D_t  = torch.tensor(samples["device"],       dtype=torch.long).to(device)
    Tr_t = torch.tensor(samples["traffic"],      dtype=torch.long).to(device)
    F_t  = torch.tensor(samples["funnel_stage"], dtype=torch.long).to(device)

    with torch.inference_mode():
        probs = torch.softmax(model(X_t, P_t, D_t, Tr_t, F_t), dim=1)
        topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=1)

    results = []
    for i in range(len(samples["X"])):
        page_seq = [mappings["idx_to_page"].get(int(v), f"<page_{v}>")
                    for v in samples["X"][i] if int(v) != 0]
        preds = [{"page": mappings["idx_to_page"].get(int(topk_idx[i,j].item()), f"<page_{int(topk_idx[i,j].item())}>"),
                  "probability": round(topk_probs[i,j].item() * 100, 2)}
                 for j in range(top_k)]
        true_idx  = int(samples["y"][i])
        results.append({
            "page_sequence" : page_seq,
            "true_next"     : mappings["idx_to_page"].get(true_idx, f"<page_{true_idx}>"),
            "predictions"   : preds,
            "persona_type"  : mappings["rev_persona"].get(int(samples["persona"][i]),           "unknown"),
            "device_type"   : mappings["rev_device"].get(int(samples["device"][i]),             "unknown"),
            "traffic_source": mappings["rev_traffic"].get(int(samples["traffic"][i]),           "unknown"),
            "funnel_stage"  : mappings["rev_funnel_stage"].get(int(samples["funnel_stage"][i]), "unknown"),
        })
    return results

def print_predictions(results):
    print("  BiLSTM NEXT-CLICK PREDICTIONS")
    for i, r in enumerate(results, 1):
        print(f"\n  Sample {i}")
        print(f"  {'─'*60}")
        print(f"  Context : persona={r['persona_type']}  device={r['device_type']}  "
              f"traffic={r['traffic_source']}  funnel={r['funnel_stage']}")
        print(f"  Journey : {' -> '.join(r['page_sequence']) if r['page_sequence'] else '(empty)'}")
        top = r["predictions"][0]
        match = "(SELECTED)" if top["page"] == r["true_next"] else ""
        print(f"  Predicted next : {top['page']:40s}  {top['probability']:.2f}%  {match}")
        print(f"  Ground truth   : {r['true_next']}")

def load_prompt_template(prompt_path):
    with open(prompt_path) as f:
        template = f.read()
    print(f"  Template loaded - {len(template)} characters")
    return template


def build_prompt(template, result):
    seq_str    = " -> ".join(result["page_sequence"]) if result["page_sequence"] else "(no pages recorded)"
    top_pred   = result["predictions"][0]
    return template.format(
        persona_type         = result["persona_type"],
        device_type          = result["device_type"],
        traffic_source       = result["traffic_source"],
        funnel_stage         = result["funnel_stage"],
        page_sequence        = seq_str,
        predicted_next_click = top_pred["page"],
        confidence_score     = f"{top_pred['probability']:.2f}",
    )

def load_llm(hf_token, model_id):
    if not hf_token:
        raise ValueError("\nNo HF token found.\n")
    
    client = InferenceClient(provider="auto", api_key=hf_token)
    return client, model_id

def generate_recommendation(prompt, client, model_id, max_new_tokens=LLM_MAX_NEW_TOKENS):
    completion = client.chat.completions.create(
        model    = model_id,
        messages = [{"role": "user", "content": prompt}],
        max_tokens = max_new_tokens,
    )
    return completion.choices[0].message.content.strip()

def clean_recommendation(text):
    import re
    lines = text.splitlines()
    cleaned = []
    skip_phrases = [
        "based on the session data",
        "here are my recommendations",
        "here are the recommendations",
    ]
    for line in lines:
        if any(line.strip().lower().startswith(p) for p in skip_phrases):
            continue
        line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        line = re.sub(r'^(\s*)\* ', r'\1- ', line)
        cleaned.append(line)
    return "\n".join(cleaned)

def format_report_entry(sample_num, result, recommendation):
    lines = []
    sep = "═" * 72

    lines += [sep, f"  SESSION {sample_num}  -  CONVERSION RECOMMENDATION", sep, ""]

    lines += [
        "  SESSION CONTEXT",
        f"  {'─'*60}",
        f"  Persona type   : {result['persona_type']}",
        f"  Device type    : {result['device_type']}",
        f"  Traffic source : {result['traffic_source']}",
        f"  Funnel stage   : {result['funnel_stage']}",
        "",
    ]

    journey = result["page_sequence"] or ["(empty)"]
    journey_str = " -> ".join(journey)
    lines += ["  PAGE JOURNEY", f"  {'─'*60}", f"  {journey_str}", ""]

    top = result["predictions"][0]
    lines += [
        "  PREDICTED NEXT CLICK",
        f"  {'─'*60}",
        f"  {top['page']:45s}  {top['probability']:.2f}%  (SELECTED)" if top["page"] == result["true_next"]
        else f"  {top['page']:45s}  {top['probability']:.2f}%",
        f"  Ground truth: {result['true_next']}",
        "",
    ]

    lines += ["  CONVERSION RECOMMENDATIONS", f"  {'─'*60}"]
    cleaned = clean_recommendation(recommendation)
    for line in cleaned.splitlines():
        lines.append(f"  {line}")
    lines.append("")

    return "\n".join(lines)


def save_report(entries, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "CONVERSION RECOMMENDATION REPORT\n"
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "═" * 72 + "\n\n"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header)
        for entry in entries:
            f.write(entry + "\n\n")

def main():
    device          = get_device()
    mappings        = load_mappings(DATA_DIR)
    samples         = load_test_samples(DATA_DIR, n=NUM_SAMPLES)
    bilstm_model    = load_bilstm(MODEL_PATH, mappings, device)
    results         = run_bilstm_inference(bilstm_model, samples, mappings, device, top_k=TOP_K)
    print_predictions(results)
    prompt_template = load_prompt_template(PROMPT_TEMPLATE)
    client, model_id = load_llm(HF_TOKEN, HF_MODEL_ID)

    report_entries = []
    for i, result in enumerate(results, 1):
        filled_prompt  = build_prompt(prompt_template, result)
        recommendation = generate_recommendation(filled_prompt, client, model_id)

        print(f"\n{'─'*72}")
        print(f"  SESSION {i} — HOW TO CONVERT THIS USER")
        print(f"{'─'*72}")
        for line in recommendation.splitlines():
            print(textwrap.fill(line, width=78, subsequent_indent="    ") if line.strip() else "")

        report_entries.append(format_report_entry(i, result, recommendation))

    save_report(report_entries, REPORT_PATH)

    print("\n" + "=" * 72)
    print(f"  Processed : {len(results)} sessions")
    print(f"  Report    : {REPORT_PATH}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()