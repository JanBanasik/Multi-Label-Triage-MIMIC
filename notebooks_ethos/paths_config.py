"""
Ścieżki dla notebooków 01–07. Uruchamiaj notebooki z katalogu notebooks_ethos (cwd = ten folder).
"""
from pathlib import Path

# Katalog roboczy = notebooks_ethos
NOTEBOOKS_DIR = Path(__file__).resolve().parent

# Dane przetworzone (output z 01, 02, 03, 05)
DATA_PROCESSED = NOTEBOOKS_DIR / "data" / "processed"

# Modele
MODELS_MLM = NOTEBOOKS_DIR / "models" / "mlm"      # medical_bert_mlm_* (notebook 04)
MODELS_TRIAGE = NOTEBOOKS_DIR / "models" / "triage"  # bert_triage_model_* (notebook 05)

# Wyniki (wykresy, tabele)
OUTPUTS_PLOTS = NOTEBOOKS_DIR / "outputs" / "plots"
OUTPUTS_TABLES = NOTEBOOKS_DIR / "outputs" / "tables"

def ensure_dirs():
    """Tworzy katalogi, jeśli nie istnieją."""
    for d in (DATA_PROCESSED, MODELS_MLM, MODELS_TRIAGE, OUTPUTS_PLOTS, OUTPUTS_TABLES):
        d.mkdir(parents=True, exist_ok=True)
