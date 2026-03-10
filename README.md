# 🎬 IMDB Sentiment Analysis — BiLSTM + Attention

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Accuracy](https://img.shields.io/badge/Accuracy-89.38%25-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)

Binary sentiment classification (Positive/Negative) on the IMDB Large Movie 
Review Dataset using a BiLSTM model with Bahdanau Attention mechanism.

---

## 📊 Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 89.38% |
| Precision | 0.89   |
| Recall    | 0.89   |
| F1 Score  | 0.89   |

---

## 🏗️ Model Architecture
```
Embedding(25002, 100) — GloVe.6B.100d pretrained
        ↓
SpatialDropout(0.2)
        ↓
BiLSTM(128 units × 2 directions, 2 layers)
        ↓
Bahdanau Attention (softmax-weighted context vector)
        ↓
Dropout(0.5)
        ↓
Dense(1, Sigmoid) → Positive / Negative
```

---

## 📁 Repository Structure
```
sentiment-analysis-bilstm/
│
├── model/
│   └── sentiment_model.pt       # trained model weights
├── notebook/
│   └── sentiment_analysis.ipynb # full training notebook
├── src/
│   └── model.py                 # model architecture class
│   └── predict.py               # inference script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation
```bash
git clone https://github.com/AkhileshS0ni/sentiment-analysis-bilstm.git
cd sentiment-analysis-bilstm
pip install -r requirements.txt
```

---

## 🚀 Quick Inference
```python
import torch
from src.model import BiLSTMAttention

# Load model
model = BiLSTMAttention(vocab_size=25002, embed_dim=100, 
                         hidden_dim=128, num_layers=2)
model.load_state_dict(torch.load('model/sentiment_model.pt'))
model.eval()

# Predict
text = "This movie was absolutely amazing!"
# (add your tokenization steps here)
output = model(input_tensor)
sentiment = "Positive" if output > 0.5 else "Negative"
print(sentiment)
```

---

## 📦 Dataset

- **IMDB Large Movie Review Dataset**
- 25,000 training samples + 25,000 test samples
- Binary labels: Positive / Negative

---

## 🔗 Links

- 📓 [Colab Notebook](#)        ← add your colab link
- 🤗 [HuggingFace Model](#)     ← add your HF link
- 👤 [My GitHub](https://github.com/AkhileshS0ni)

---

## 📋 Requirements
```
torch>=2.0.0
numpy
pandas
```

---

## 👤 Author

**Akhilesh Soni**  
M.Tech Computer Engineering | NIT Kurukshetra  
[LinkedIn](https://linkedin.com/in/akhileshs0ni) | 
[GitHub](https://github.com/AkhileshS0ni)
```

---

### Files You Should Push to GitHub

| File | What it is |
|---|---|
| `README.md` | Above file |
| `sentiment_model.pt` | Your trained weights |
| `sentiment_analysis.ipynb` | Your Colab notebook |
| `model.py` | Your BiLSTM class definition |
| `requirements.txt` | Dependencies |

---

### requirements.txt — Create This Too
```
torch>=2.0.0
numpy
pandas
torchtext
