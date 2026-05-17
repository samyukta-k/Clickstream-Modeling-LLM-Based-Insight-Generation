import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR   = "data/processed"
OUT_MODELS = "outputs/models"
OUT_CHARTS = "outputs/charts"

PAGE_EMBEDDING_DIM = 32    
HIDDEN_DIM         = 64    
NUM_LAYERS         = 1     
DROPOUT            = 0.5  

PERSONA_EMBEDDING_DIM      = 8
DEVICE_EMBEDDING_DIM       = 8
TRAFFIC_EMBEDDING_DIM      = 8
FUNNEL_STAGE_EMBEDDING_DIM = 8

BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 50
PATIENCE      = 5
WEIGHT_DECAY  = 1e-4     
CHECKPOINT_ON_ACCURACY = True  

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def load_data(data_dir: str) -> dict:
    def npy(name):
        return np.load(os.path.join(data_dir, f"{name}.npy"))

    def jload(name):
        with open(os.path.join(data_dir, f"{name}.json")) as f:
            return json.load(f)

    d = {}

    d["X_train"] = npy("X_train")
    d["X_test"]  = npy("X_test")
    d["y_train"] = npy("y_train")
    d["y_test"]  = npy("y_test")

    for feat in ("persona", "device", "traffic", "funnel_stage"):
        d[f"{feat}_train"] = npy(f"{feat}_train")
        d[f"{feat}_test"]  = npy(f"{feat}_test")

    d["page_to_idx"] = jload("page_to_idx")
    idx_to_page      = jload("idx_to_page")
    d["idx_to_page"] = {int(k): v for k, v in idx_to_page.items()}

    for feat in ("persona", "device", "traffic", "funnel_stage"):
        d[f"{feat}_encoder"] = jload(f"{feat}_encoder")

    d["vocab_size"]              = len(d["page_to_idx"]) + 1  
    d["persona_vocab_size"]      = max(d["persona_encoder"].values())      + 1
    d["device_vocab_size"]       = max(d["device_encoder"].values())       + 1
    d["traffic_vocab_size"]      = max(d["traffic_encoder"].values())      + 1
    d["funnel_stage_vocab_size"] = max(d["funnel_stage_encoder"].values()) + 1

    print(f"  X_train           : {d['X_train'].shape}   y_train : {d['y_train'].shape}")
    print(f"  X_test            : {d['X_test'].shape}    y_test  : {d['y_test'].shape}")
    print(f"  persona_train     : {d['persona_train'].shape}")
    print(f"  device_train      : {d['device_train'].shape}")
    print(f"  traffic_train     : {d['traffic_train'].shape}")
    print(f"  funnel_stage_train: {d['funnel_stage_train'].shape}")
    print(f"  Page vocab size         : {d['vocab_size']}")
    print(f"  Persona vocab size      : {d['persona_vocab_size']}")
    print(f"  Device vocab size       : {d['device_vocab_size']}")
    print(f"  Traffic vocab size      : {d['traffic_vocab_size']}")
    print(f"  Funnel stage vocab size : {d['funnel_stage_vocab_size']}")

    return d

def make_dataloaders(d: dict, batch_size: int) -> tuple:
    def tensors(split):
        return (
            torch.tensor(d[f"X_{split}"],              dtype=torch.long),
            torch.tensor(d[f"persona_{split}"],        dtype=torch.long),
            torch.tensor(d[f"device_{split}"],         dtype=torch.long),
            torch.tensor(d[f"traffic_{split}"],        dtype=torch.long),
            torch.tensor(d[f"funnel_stage_{split}"],   dtype=torch.long),
            torch.tensor(d[f"y_{split}"],              dtype=torch.long),
        )

    train_ds = TensorDataset(*tensors("train"))
    test_ds  = TensorDataset(*tensors("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader

class BiLSTMNextClickWithContext(nn.Module):
    def __init__(self,
                 page_vocab_size,         persona_vocab_size,
                 device_vocab_size,       traffic_vocab_size,
                 funnel_stage_vocab_size,
                 page_emb_dim=32,         hidden_dim=64,   num_layers=1,
                 persona_emb_dim=8,       device_emb_dim=8,
                 traffic_emb_dim=8,       funnel_stage_emb_dim=8,
                 dropout=0.5):
        super().__init__()
        
        self.page_embedding = nn.Embedding(
            page_vocab_size, page_emb_dim, padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size   = page_emb_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,                           
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        self.bilstm_output_dim = hidden_dim * 2

        self.persona_embedding = nn.Embedding(
            persona_vocab_size, persona_emb_dim, padding_idx=0
        )
        self.device_embedding = nn.Embedding(
            device_vocab_size, device_emb_dim, padding_idx=0
        )
        self.traffic_embedding = nn.Embedding(
            traffic_vocab_size, traffic_emb_dim, padding_idx=0
        )
        self.funnel_stage_embedding = nn.Embedding(
            funnel_stage_vocab_size, funnel_stage_emb_dim, padding_idx=0
        )

        combined_dim = (self.bilstm_output_dim
                        + persona_emb_dim
                        + device_emb_dim
                        + traffic_emb_dim
                        + funnel_stage_emb_dim)

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(combined_dim, page_vocab_size)

    def forward(self, x_pages, x_persona, x_device, x_traffic, x_funnel_stage,
                debug: bool = False):
        page_emb = self.page_embedding(x_pages)      
        lstm_out, _ = self.lstm(page_emb)
        last_hidden = lstm_out[:, -1, :]

        p_emb = self.persona_embedding(x_persona)          
        d_emb = self.device_embedding(x_device)          
        t_emb = self.traffic_embedding(x_traffic)         
        f_emb = self.funnel_stage_embedding(x_funnel_stage)

        combined = torch.cat([last_hidden, p_emb, d_emb, t_emb, f_emb], dim=1)
        out    = self.dropout(combined)
        logits = self.classifier(out)

        return logits


def build_model(d: dict, device) -> nn.Module:
    bilstm_output_dim = HIDDEN_DIM * 2
    combined_dim = (bilstm_output_dim
                    + PERSONA_EMBEDDING_DIM
                    + DEVICE_EMBEDDING_DIM
                    + TRAFFIC_EMBEDDING_DIM
                    + FUNNEL_STAGE_EMBEDDING_DIM)

    model = BiLSTMNextClickWithContext(
        page_vocab_size         = d["vocab_size"],
        persona_vocab_size      = d["persona_vocab_size"],
        device_vocab_size       = d["device_vocab_size"],
        traffic_vocab_size      = d["traffic_vocab_size"],
        funnel_stage_vocab_size = d["funnel_stage_vocab_size"],
        page_emb_dim            = PAGE_EMBEDDING_DIM,
        hidden_dim              = HIDDEN_DIM,
        num_layers              = NUM_LAYERS,
        persona_emb_dim         = PERSONA_EMBEDDING_DIM,
        device_emb_dim          = DEVICE_EMBEDDING_DIM,
        traffic_emb_dim         = TRAFFIC_EMBEDDING_DIM,
        funnel_stage_emb_dim    = FUNNEL_STAGE_EMBEDDING_DIM,
        dropout                 = DROPOUT,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device,
                    debug_first_batch: bool = False) -> tuple:
    model.train()
    total_loss = correct = total = 0
    first = True

    for X, persona, dev, traffic, funnel_stage, y in loader:
        X, persona, dev, traffic, funnel_stage, y = (
            X.to(device), persona.to(device), dev.to(device),
            traffic.to(device), funnel_stage.to(device), y.to(device)
        )

        optimizer.zero_grad()
        logits = model(X, persona, dev, traffic, funnel_stage,
                       debug=(debug_first_batch and first))
        first = False

        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds    = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += X.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, persona, dev, traffic, funnel_stage, y in loader:
            X, persona, dev, traffic, funnel_stage, y = (
                X.to(device), persona.to(device), dev.to(device),
                traffic.to(device), funnel_stage.to(device), y.to(device)
            )

            logits = model(X, persona, dev, traffic, funnel_stage)
            loss   = criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            preds    = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += X.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    return total_loss / total, correct / total, all_preds, all_labels

def train_model(model, train_loader, test_loader,
                num_epochs, patience, learning_rate,
                model_save_path, device,
                checkpoint_on_accuracy: bool = True) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode    = "max" if checkpoint_on_accuracy else "min",
        factor  = 0.5,
        patience= 3,
    )

    history          = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_metric  = -float("inf") if checkpoint_on_accuracy else float("inf")
    patience_counter = 0
    metric_label     = "Val Acc" if checkpoint_on_accuracy else "Val Loss"
    debug_done       = False

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            debug_first_batch=(not debug_done)
        )
        debug_done = True

        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{train_acc:>9.4f}  {val_acc:>9.4f}  {elapsed:>5.1f}s")

        val_metric = val_acc if checkpoint_on_accuracy else -val_loss
        scheduler.step(val_metric)

        if val_metric > best_val_metric:
            best_val_metric  = val_metric
            patience_counter = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"         Best model saved  ({metric_label}={best_val_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    print("=" * 72)
    return history

def predict(model, X, persona, device_arr, traffic, funnel_stage,
            idx_to_page: dict, device, top_k: int = 3) -> list:
    model.eval()
    Xt  = torch.tensor(X,            dtype=torch.long).to(device)
    Pt  = torch.tensor(persona,      dtype=torch.long).to(device)
    Dt  = torch.tensor(device_arr,   dtype=torch.long).to(device)
    Tt  = torch.tensor(traffic,      dtype=torch.long).to(device)
    Ft  = torch.tensor(funnel_stage, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(Xt, Pt, Dt, Tt, Ft)
        probs  = torch.softmax(logits, dim=1)
        topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=1)

    results = []
    for i in range(len(X)):
        seq_pages = [
            idx_to_page.get(int(v), f"<{v}>")
            for v in X[i] if v != 0
        ]
        preds = [
            (
                idx_to_page.get(int(topk_idx[i, j].item()), f"<{topk_idx[i,j].item()}>"),
                round(topk_probs[i, j].item(), 4),
            )
            for j in range(top_k)
        ]
        results.append({"sequence_pages": seq_pages, "top_k_predictions": preds})

    return results

def _save_line_chart(values_dict, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for label, vals in values_dict.items():
        plt.plot(range(1, len(vals) + 1), vals, marker="o", markersize=3, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_loss(history, charts_dir):
    _save_line_chart({"Train Loss": history["train_loss"]},
                     "Training Loss per Epoch", "Cross-Entropy Loss",
                     os.path.join(charts_dir, "training_loss.png"))


def plot_validation_loss(history, charts_dir):
    _save_line_chart({"Validation Loss": history["val_loss"]},
                     "Validation Loss per Epoch", "Cross-Entropy Loss",
                     os.path.join(charts_dir, "validation_loss.png"))


def plot_accuracy(history, charts_dir):
    _save_line_chart(
        {"Train Accuracy": history["train_acc"], "Val Accuracy": history["val_acc"]},
        "Accuracy per Epoch", "Accuracy",
        os.path.join(charts_dir, "accuracy.png"))


def plot_confusion_matrix(all_labels, all_preds, idx_to_page, charts_dir,
                           max_classes=20):
    unique, counts = np.unique(all_labels, return_counts=True)
    top_idx        = sorted(unique[np.argsort(-counts)[:max_classes]])
    mask           = np.isin(all_labels, top_idx)
    cm             = confusion_matrix(
        np.array(all_labels)[mask], np.array(all_preds)[mask], labels=top_idx
    )
    labels_str = [idx_to_page.get(int(i), str(i))[-20:] for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_str, yticklabels=labels_str,
                linewidths=0.5, ax=ax)
    ax.set_title(f"Confusion Matrix — BiLSTM (top-{len(top_idx)} classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    save_path = os.path.join(charts_dir, "confusion_matrix.png")
    os.makedirs(charts_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    d = load_data(DATA_DIR)
    train_loader, test_loader = make_dataloaders(d, batch_size=BATCH_SIZE)
    model = build_model(d, DEVICE)

    model_save_path = os.path.join(OUT_MODELS, "lstm_next_click.pt")
    history = train_model(
        model                  = model,
        train_loader           = train_loader,
        test_loader            = test_loader,
        num_epochs             = NUM_EPOCHS,
        patience               = PATIENCE,
        learning_rate          = LEARNING_RATE,
        model_save_path        = model_save_path,
        device                 = DEVICE,
        checkpoint_on_accuracy = CHECKPOINT_ON_ACCURACY,
    )
    
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n{'='*48}")
    print(f"  Final Test Loss     : {test_loss:.4f}")
    print(f"  Final Test Accuracy : {test_acc * 100:.2f}%")
    print(f"{'='*48}\n")
    plot_training_loss(history, OUT_CHARTS)
    plot_validation_loss(history, OUT_CHARTS)
    plot_accuracy(history, OUT_CHARTS)
    plot_confusion_matrix(all_labels, all_preds, d["idx_to_page"], OUT_CHARTS)

    n = 5
    preds_out = predict(
        model,
        X            = d["X_test"][:n],
        persona      = d["persona_test"][:n],
        device_arr   = d["device_test"][:n],
        traffic      = d["traffic_test"][:n],
        funnel_stage = d["funnel_stage_test"][:n],
        idx_to_page  = d["idx_to_page"],
        device       = DEVICE,
        top_k        = 3,
    )
    rev_persona      = {v: k for k, v in d["persona_encoder"].items()}
    rev_device       = {v: k for k, v in d["device_encoder"].items()}
    rev_traffic      = {v: k for k, v in d["traffic_encoder"].items()}
    rev_funnel_stage = {v: k for k, v in d["funnel_stage_encoder"].items()}

    for i, (pred_info, true_idx) in enumerate(zip(preds_out, d["y_test"][:n])):
        true_page    = d["idx_to_page"].get(int(true_idx), f"<{true_idx}>")
        persona_l    = rev_persona.get(int(d["persona_test"][i]),           "?")
        device_l     = rev_device.get(int(d["device_test"][i]),             "?")
        traffic_l    = rev_traffic.get(int(d["traffic_test"][i]),           "?")
        funnel_l     = rev_funnel_stage.get(int(d["funnel_stage_test"][i]), "?")

if __name__ == "__main__":
    main()