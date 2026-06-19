import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json

# --- 1. TOKENIZER ---
class MedicalTokenizer:
    def __init__(self, vocab_path="vocab_mlm.json"):
        self.vocab_path = vocab_path
        try:
            with open(vocab_path, "r") as f:
                self.token2idx = json.load(f)
        except FileNotFoundError:
            # Fallback dla treningu od zera
            self.token2idx = {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2}
            
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
        
    def encode(self, text_seq):
        if isinstance(text_seq, str): text_seq = text_seq.split()
        return [self.token2idx.get(t, 2) for t in text_seq] # 2 = UNK

# --- 2. MODEL (ARCHITEKTURA) ---
class EthosClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(100, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dim_feedforward=embed_dim*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Głowica klasyfikacyjna
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        # x: [batch, seq_len]
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        
        # Padding mask (ignorujemy zera)
        padding_mask = (x.sum(dim=-1) == 0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Global Average Pooling (z pominięciem paddingu)
        mask = (~padding_mask).float().unsqueeze(-1)
        # Zabezpieczenie przed dzieleniem przez zero (gdyby cała sekwencja była paddingiem - mało prawdopodobne)
        sum_mask = mask.sum(dim=1)
        sum_mask[sum_mask == 0] = 1.0 
        
        x = (x * mask).sum(dim=1) / sum_mask
        
        logits = self.fc(x)
        return logits

    def predict(self, x_tensor):
        """Pomocnicza funkcja do inferencji"""
        self.eval()
        with torch.no_grad():
            logits = self(x_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item(), probs.cpu().numpy()[0]

# --- 3. DATASET ---
class EthosClassificationDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=50):
        self.X = []
        self.y = []
        self.max_len = max_len
        self.label_map = {}
        
        # Zbieranie labeli
        raw_labels = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                raw_labels.append(parts[-1])
        
        unique_labels = sorted(list(set(raw_labels)))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Kodowanie
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                
                text_tokens = parts[:-1]
                label_str = parts[-1]
                
                x_ids = tokenizer.encode(text_tokens)
                if len(x_ids) > max_len:
                    x_ids = x_ids[:max_len]
                else:
                    x_ids += [0] * (max_len - len(x_ids))
                
                self.X.append(torch.tensor(x_ids))
                self.y.append(self.label_map[label_str])

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], torch.tensor(self.y[idx])

# --- 4. SAFETY GUARD (DETERMINISTIC) ---
class SafetyGuard:
    def __init__(self):
        # Trigger tokens (oparte na kwantylach _VERY_LOW / _VERY_HIGH)
        self.critical_triggers = {
            "HGB_VERY_LOW", "PLT_VERY_LOW", "WBC_VERY_HIGH", 
            "CREATININE_VERY_HIGH", "TROPONIN_HIGH", "HGB_CRITICAL_LOW"
        }

    def check(self, tokens_list):
        detected = [t for t in tokens_list if t in self.critical_triggers]
        if detected:
            return True, f"CRITICAL RULE: Found {detected}"
        return False, None