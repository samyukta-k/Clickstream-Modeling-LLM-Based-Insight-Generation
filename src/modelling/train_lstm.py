import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR   = "data/processed"
OUT_MODELS = "outputs/models"
OUT_CHARTS = "outputs/charts"

PAGE_EMBEDDING_DIM = 32
HIDDEN_DIM         = 64
NUM_LAYERS         = 1
DROPOUT            = 0.3

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

    for feat in (
        "persona",
        "device",
        "traffic",
        "funnel_stage"
    ):
        d[f"{feat}_train"] = npy(f"{feat}_train")
        d[f"{feat}_test"]  = npy(f"{feat}_test")

    d["page_to_idx"] = jload("page_to_idx")

    idx_to_page = jload("idx_to_page")

    d["idx_to_page"] = {
        int(k): v
        for k, v in idx_to_page.items()
    }

    for feat in (
        "persona",
        "device",
        "traffic",
        "funnel_stage"
    ):
        d[f"{feat}_encoder"] = jload(f"{feat}_encoder")

    d["vocab_size"] = len(d["page_to_idx"]) + 1

    d["persona_vocab_size"] = (
        max(d["persona_encoder"].values()) + 1
    )

    d["device_vocab_size"] = (
        max(d["device_encoder"].values()) + 1
    )

    d["traffic_vocab_size"] = (
        max(d["traffic_encoder"].values()) + 1
    )

    d["funnel_stage_vocab_size"] = (
        max(d["funnel_stage_encoder"].values()) + 1
    )

    print(f"X_train : {d['X_train'].shape}")
    print(f"X_test  : {d['X_test'].shape}")

    print(f"y_train : {d['y_train'].shape}")
    print(f"y_test  : {d['y_test'].shape}")

    print(f"Vocabulary size : {d['vocab_size']}")

    return d


def make_dataloaders(
    d: dict,
    batch_size: int
):

    def tensors(split):
        return (
            torch.tensor(
                d[f"X_{split}"],
                dtype=torch.long
            ),

            torch.tensor(
                d[f"persona_{split}"],
                dtype=torch.long
            ),

            torch.tensor(
                d[f"device_{split}"],
                dtype=torch.long
            ),

            torch.tensor(
                d[f"traffic_{split}"],
                dtype=torch.long
            ),

            torch.tensor(
                d[f"funnel_stage_{split}"],
                dtype=torch.long
            ),

            torch.tensor(
                d[f"y_{split}"],
                dtype=torch.long
            ),
        )

    train_ds = TensorDataset(*tensors("train"))

    test_ds = TensorDataset(*tensors("test"))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# =========================================================
# BiLSTM + Attention Model
# =========================================================
class BiLSTMNextClickWithContext(nn.Module):

    def __init__(
        self,
        page_vocab_size,
        persona_vocab_size,
        device_vocab_size,
        traffic_vocab_size,
        funnel_stage_vocab_size,
        page_emb_dim=32,
        hidden_dim=64,
        num_layers=1,
        persona_emb_dim=8,
        device_emb_dim=8,
        traffic_emb_dim=8,
        funnel_stage_emb_dim=8,
        dropout=0.5
    ):

        super().__init__()

        # =================================================
        # PAGE EMBEDDING
        # =================================================
        self.page_embedding = nn.Embedding(
            page_vocab_size,
            page_emb_dim,
            padding_idx=0
        )

        # =================================================
        # BiLSTM
        # =================================================
        self.lstm = nn.LSTM(
            input_size=page_emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.bilstm_output_dim = hidden_dim * 2

        # =================================================
        # ATTENTION LAYER
        # =================================================
        self.attention = nn.Linear(
            self.bilstm_output_dim,
            1
        )

        # =================================================
        # CONTEXT EMBEDDINGS
        # =================================================
        self.persona_embedding = nn.Embedding(
            persona_vocab_size,
            persona_emb_dim,
            padding_idx=0
        )

        self.device_embedding = nn.Embedding(
            device_vocab_size,
            device_emb_dim,
            padding_idx=0
        )

        self.traffic_embedding = nn.Embedding(
            traffic_vocab_size,
            traffic_emb_dim,
            padding_idx=0
        )

        self.funnel_stage_embedding = nn.Embedding(
            funnel_stage_vocab_size,
            funnel_stage_emb_dim,
            padding_idx=0
        )

        combined_dim = (
            self.bilstm_output_dim
            + persona_emb_dim
            + device_emb_dim
            + traffic_emb_dim
            + funnel_stage_emb_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            combined_dim,
            page_vocab_size
        )

    def forward(
        self,
        x_pages,
        x_persona,
        x_device,
        x_traffic,
        x_funnel_stage
    ):

        # =================================================
        # PAGE EMBEDDINGS
        # =================================================
        page_emb = self.page_embedding(x_pages)

        # =================================================
        # BiLSTM OUTPUT
        # =================================================
        lstm_out, _ = self.lstm(page_emb)

        # =================================================
        # ATTENTION
        # =================================================
        attention_scores = self.attention(lstm_out)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context_vector = torch.sum(
            attention_weights * lstm_out,
            dim=1
        )

        # =================================================
        # CONTEXT FEATURES
        # =================================================
        p_emb = self.persona_embedding(x_persona)

        d_emb = self.device_embedding(x_device)

        t_emb = self.traffic_embedding(x_traffic)

        f_emb = self.funnel_stage_embedding(x_funnel_stage)

        # =================================================
        # CONCATENATE FEATURES
        # =================================================
        combined = torch.cat([
            context_vector,
            p_emb,
            d_emb,
            t_emb,
            f_emb
        ], dim=1)

        combined = self.dropout(combined)

        logits = self.classifier(combined)

        return logits


def build_model(d: dict, device):

    model = BiLSTMNextClickWithContext(
        page_vocab_size=d["vocab_size"],
        persona_vocab_size=d["persona_vocab_size"],
        device_vocab_size=d["device_vocab_size"],
        traffic_vocab_size=d["traffic_vocab_size"],
        funnel_stage_vocab_size=d["funnel_stage_vocab_size"],
        page_emb_dim=PAGE_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        persona_emb_dim=PERSONA_EMBEDDING_DIM,
        device_emb_dim=DEVICE_EMBEDDING_DIM,
        traffic_emb_dim=TRAFFIC_EMBEDDING_DIM,
        funnel_stage_emb_dim=FUNNEL_STAGE_EMBEDDING_DIM,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Total parameters : {total_params:,}")

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for (
        X,
        persona,
        dev,
        traffic,
        funnel_stage,
        y
    ) in loader:

        X = X.to(device)

        persona = persona.to(device)

        dev = dev.to(device)

        traffic = traffic.to(device)

        funnel_stage = funnel_stage.to(device)

        y = y.to(device)

        optimizer.zero_grad()

        logits = model(
            X,
            persona,
            dev,
            traffic,
            funnel_stage
        )

        loss = criterion(logits, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * X.size(0)

        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()

        total += X.size(0)

    return total_loss / total, correct / total


def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for (
            X,
            persona,
            dev,
            traffic,
            funnel_stage,
            y
        ) in loader:

            X = X.to(device)

            persona = persona.to(device)

            dev = dev.to(device)

            traffic = traffic.to(device)

            funnel_stage = funnel_stage.to(device)

            y = y.to(device)

            logits = model(
                X,
                persona,
                dev,
                traffic,
                funnel_stage
            )

            loss = criterion(logits, y)

            total_loss += loss.item() * X.size(0)

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()

            total += X.size(0)

            all_preds.extend(
                preds.cpu().numpy().tolist()
            )

            all_labels.extend(
                y.cpu().numpy().tolist()
            )

    return (
        total_loss / total,
        correct / total,
        all_preds,
        all_labels
    )


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs,
    patience,
    learning_rate,
    model_save_path,
    device
):

    criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(1, num_epochs + 1):

        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Train Acc={train_acc:.4f} | "
            f"Val Acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            patience_counter = 0

            os.makedirs(
                os.path.dirname(model_save_path),
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                model_save_path
            )

            print(
                f"Best model saved "
                f"(Val Acc={best_val_acc:.4f})"
            )

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return history


def main():

    d = load_data(DATA_DIR)

    train_loader, test_loader = make_dataloaders(
        d,
        batch_size=BATCH_SIZE
    )

    model = build_model(d, DEVICE)

    model_save_path = os.path.join(
        OUT_MODELS,
        "bilstm_attention_next_click.pt"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        model_save_path=model_save_path,
        device=DEVICE
    )

    model.load_state_dict(
        torch.load(
            model_save_path,
            map_location=DEVICE
        )
    )

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, _, _ = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print("\n" + "=" * 50)

    print(f"Final Test Loss     : {test_loss:.4f}")

    print(
        f"Final Test Accuracy : "
        f"{test_acc * 100:.2f}%"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()