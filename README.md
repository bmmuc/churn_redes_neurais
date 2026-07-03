# Germano — Telco Customer Churn Prediction

Comparação de modelos para predição de churn no dataset [Telco Customer Churn (IBM)](https://www.kaggle.com/blastchar/telco-customer-churn).

## Modelos

| Notebook | Modelo |
|---|---|
| `preprocessing.ipynb` | Pré-processamento + RandomOverSampler |
| `gradient_boosting.ipynb` | XGBoost com Optuna |
| `mlp.ipynb` | MLP (PyTorch) |
| `kan.ipynb` | Kolmogorov-Arnold Network |
| `tabpfn.ipynb` | TabPFN v2 (in-context learning) |
| `tabkan.ipynb` | TabKAN (ChebyshevKAN) |
| `mitra.ipynb` | MITRA |

> Todos os modelos usam mesmo pipeline: split 70/15/15 estratificado, mesma codificação, mesmo balanceamento (RandomOverSampler apenas no treino) e mesma função `compute_metrics`.

## Resultados (test set)

| Modelo | Accuracy | Bal. Accuracy | Precision | Recall | F1 | AUROC | KS |
|---|---|---|---|---|---|---|---|
| STAB | 0.706 | **0.763** | 0.471 | **0.886** | 0.615 | 0.848 | 0.545 |
| MLP | 0.728 | 0.756 | 0.491 | 0.818 | 0.614 | 0.844 | 0.540 |
| KAN | 0.742 | 0.755 | 0.508 | 0.782 | 0.616 | 0.846 | 0.537 |
| TabKAN | 0.748 | 0.771 | 0.516 | 0.818 | 0.633 | **0.855** | **0.552** |
| XGBoost | 0.769 | 0.776 | 0.544 | 0.789 | **0.644** | 0.847 | **0.554** |
| TabPFN | 0.793 | 0.707 | **0.631** | 0.525 | 0.573 | 0.836 | 0.540 |
| MITRA | **0.801** | 0.695 | 0.682 | 0.468 | 0.555 | 0.845 | 0.555 |

> Melhor por coluna em **negrito**. Modelos ordenados por accuracy crescente.

## Requisitos

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- GPU CUDA 12.4 recomendada (CPU funciona, mais lento)

## Setup local

```bash
git clone https://github.com/bmmuc/churn_redes_neurais.git
cd churn_redes_neurais

uv sync

source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

jupyter lab
```

## Ordem de execução

1. **`preprocessing.ipynb`** — obrigatório primeiro. Gera os arquivos em `processed/`
2. Qualquer notebook de modelo em qualquer ordem

## Estrutura

```
germano/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # dataset original
├── model_utils.py                          # funções compartilhadas
├── processed/                              # gerado pelo preprocessing.ipynb
│   ├── X_train_resampled.csv
│   ├── X_val.csv, X_test.csv
│   └── y_*.csv
└── results/                                # gerado por cada modelo
    ├── gradient_boosting/
    ├── mlp/
    ├── kan/
    ├── tabpfn/
    ├── tabkan/
    └── mitra/
```

---

## Rodar no Google Colab

### 1. Upload dos arquivos

```python
# Opção A: mount Drive (recomendado)
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/germano

# Opção B: clonar direto no Colab
!git clone https://github.com/bmmuc/churn_redes_neurais.git
%cd churn_redes_neurais
```

### 2. Instalar dependências

```python
!pip install -q \
    numpy pandas scipy scikit-learn imbalanced-learn \
    xgboost optuna matplotlib seaborn joblib tqdm \
    ipywidgets

# TabPFN v2
!pip install -q "tabpfn>=2.0,<3.0"

# TabKAN (ChebyshevKAN)
!pip install tabkan

# KAN
!pip install -q git+https://github.com/Blealtan/efficient-kan.git

# PyTorch — Colab já tem, mas para garantir CUDA:
# !pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Verificar GPU

```python
import torch
print(torch.cuda.is_available())        # deve retornar True
print(torch.cuda.get_device_name(0))    # ex: Tesla T4
```

Para usar GPU no Colab: `Runtime → Change runtime type → T4 GPU`.

### 4. Ajustes de path

```python
import os
os.chdir('/content/churn_redes_neurais')  # ou o caminho do seu Drive
```

---

## Métricas reportadas

| Métrica | Descrição |
|---|---|
| `accuracy` | Acurácia global |
| `balanced_accuracy` | Média do recall por classe |
| `precision` | TP / (TP + FP) na classe churn |
| `recall` | TP / (TP + FN) na classe churn |
| `f1` | Média harmônica precision/recall |
| `auroc` | Área sob curva ROC |
| `ks` | Kolmogorov-Smirnov statistic |

Resultados salvos em `results/<modelo>/scores.csv` e plots em `results/<modelo>/plots/`.
