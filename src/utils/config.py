from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = OUTPUTS_DIR / "logs"

PROMPTS_DIR = ROOT_DIR / "prompts"

PAGES = [
    "home",
    "search",
    "product",
    "cart",
    "checkout",
    "profile",
    "help",
    "login",
    "logout",
]
N_USERS = 500
SESSIONS_PER_USER = (1, 5)      
EVENTS_PER_SESSION = (3, 20)    
RANDOM_SEED = 42

SEQ_LEN = 5                      
EMBED_DIM = 32
LSTM_UNITS = 64
EPOCHS = 5
BATCH_SIZE = 64
TEST_SIZE = 0.2

LLM_MODEL_NAME = "google/flan-t5-base"
LLM_MAX_NEW_TOKENS = 200
