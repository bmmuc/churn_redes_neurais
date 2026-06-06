# Germano — Telco Customer Churn Prediction

Comparação de modelos para predição de churn no dataset [Telco Customer Churn (IBM)](https://www.kaggle.com/blastchar/telco-customer-churn).

## Modelos

| Notebook | Modelo | Origem |
|---|---|---|
| `preprocessing.ipynb` | Pré-processamento + RandomOverSampler | adaptado de `Churn_Prediction.ipynb` |
| `gradient_boosting.ipynb` | XGBoost com Optuna | adaptado de `Churn_Prediction.ipynb` |
| `mlp.ipynb` | MLP (PyTorch) | adaptado de `Churn_Prediction.ipynb` |
| `kan.ipynb` | Kolmogorov-Arnold Network | — |
| `stab.ipynb` | STAB (Sparse Tabular Attention Boosting) | — |
| `tabpfn.ipynb` | TabPFN v2 (in-context learning) | — |
| `Churn_Prediction.ipynb` | Análise exploratória consolidada (referência) | — |

> `preprocessing.ipynb`, `mlp.ipynb` e `gradient_boosting.ipynb` foram adaptados de `Churn_Prediction.ipynb` para manter congruência de pipeline entre todos os experimentos: mesmo split (70/15/15 estratificado), mesma codificação, mesmo balanceamento (RandomOverSampler apenas no treino) e mesma função `compute_metrics`.

## Requisitos

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- GPU CUDA 12.4 recomendada (CPU funciona, mais lento)

## Setup local

```bash
# Clonar e entrar no diretório
git clone https://github.com/bmmuc/churn_redes_neurais.git
cd churn_redes_neurais

# Criar ambiente e instalar dependências
uv sync

# Ativar ambiente
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Iniciar Jupyter
jupyter lab
```

## Ordem de execução

1. **`preprocessing.ipynb`** — obrigatório primeiro. Gera os arquivos em `processed/`
2. Qualquer notebook de modelo em qualquer ordem
3. **`Churn_Prediction.ipynb`** — análise comparativa (requer todos os modelos rodados)

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
    ├── stab/
    └── tabpfn/
```

---

## Rodar no Google Colab

### 1. Upload dos arquivos

No Colab, faça upload ou monte o Google Drive:

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

Substitua o `uv sync` por `pip install` no Colab (adicione uma célula no início de cada notebook):

```python
# Instalar dependências base
!pip install -q \
    numpy pandas scipy scikit-learn imbalanced-learn \
    xgboost optuna matplotlib seaborn joblib tqdm \
    ipywidgets

# TabPFN v2
!pip install -q "tabpfn>=2.0,<3.0"

# KAN (apenas para kan.ipynb)
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

Os notebooks usam caminhos relativos (`processed/`, `results/`). No Colab, garanta que o working directory está correto:

```python
import os
os.chdir('/content/churn_redes_neurais')  # ou o caminho do seu Drive
```

Ou adicione no topo do notebook:

```python
import sys, os
# Ajuste conforme seu ambiente
BASE = '/content/churn_redes_neurais'          # Colab
# BASE = '/content/drive/MyDrive/churn_redes_neurais'  # Colab + Drive
os.chdir(BASE)
sys.path.insert(0, BASE)
```

### 5. Executar notebooks no Colab

O Colab suporta execução direta de `.ipynb`. Faça upload do arquivo ou abra via Drive e execute célula por célula. Comece sempre pelo `preprocessing.ipynb`.

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
