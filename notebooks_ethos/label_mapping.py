"""
Wspólny moduł mapowania labeli dla wszystkich notebooków.

Zapewnia spójność mapowania między:
- Notebook 05 (Fine_Tune_Model)
- Notebook 06 (Explainability)
- Notebook 07 (Advanced_Evaluation)
"""

# Krótkie nazwy klas (używane w wizualizacjach)
LABEL_NAMES = [
    'POZ', 
    'GASTROLOG', 
    'HEMATOLOG', 
    'NEFROLOG', 
    'SOR',
    'KARDIOLOG',   # NEW
    'PULMONOLOG',  # NEW
    'HEPATOLOG'    # NEW
]

# Pełne nazwy klas (używane w danych i modelu)
LABEL_NAMES_FULL = [
    'TARGET_POZ',
    'TARGET_SPEC_GASTROLOG',
    'TARGET_SPEC_HEMATOLOG',
    'TARGET_SPEC_NEFROLOG',
    'TARGET_SPEC_SOR',
    'TARGET_SPEC_KARDIOLOG',   # NEW
    'TARGET_SPEC_PULMONOLOG',  # NEW
    'TARGET_SPEC_HEPATOLOG'    # NEW
]

# Mapowanie: pełna nazwa -> ID (używane w treningu)
LABEL_TO_ID = {
    'TARGET_POZ': 0,
    'TARGET_SPEC_GASTROLOG': 1,
    'TARGET_SPEC_HEMATOLOG': 2,
    'TARGET_SPEC_NEFROLOG': 3,
    'TARGET_SPEC_SOR': 4,
    'TARGET_SPEC_KARDIOLOG': 5,  # NEW
    'TARGET_SPEC_PULMONOLOG': 6, # NEW
    'TARGET_SPEC_HEPATOLOG': 7   # NEW
}

# Mapowanie: ID -> pełna nazwa (używane w predykcjach)
ID_TO_LABEL = {
    0: 'TARGET_POZ',
    1: 'TARGET_SPEC_GASTROLOG',
    2: 'TARGET_SPEC_HEMATOLOG',
    3: 'TARGET_SPEC_NEFROLOG',
    4: 'TARGET_SPEC_SOR',
    5: 'TARGET_SPEC_KARDIOLOG',  # NEW
    6: 'TARGET_SPEC_PULMONOLOG', # NEW
    7: 'TARGET_SPEC_HEPATOLOG'   # NEW
}

# Mapowanie: ID -> krótka nazwa (używane w wizualizacjach)
ID_TO_LABEL_SHORT = {
    0: 'POZ',
    1: 'GASTROLOG',
    2: 'HEMATOLOG',
    3: 'NEFROLOG',
    4: 'SOR',
    5: 'KARDIOLOG',  # NEW
    6: 'PULMONOLOG', # NEW
    7: 'HEPATOLOG'   # NEW
}

# Dla kompatybilności wstecznej (notebook 05 używa LABEL_MAP)
LABEL_MAP = LABEL_TO_ID

# Dla kompatybilności wstecznej (notebook 07 używa TARGET_NAMES)
TARGET_NAMES = LABEL_NAMES

# Liczba klas
NUM_LABELS = len(LABEL_NAMES)
