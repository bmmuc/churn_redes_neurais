import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as skm

PROCESSED = 'processed'
RESULTS   = 'results'


def load_data():
    X_train = pd.read_csv(f'{PROCESSED}/X_train_resampled.csv')
    y_train = pd.read_csv(f'{PROCESSED}/y_train_resampled.csv').squeeze()
    X_val   = pd.read_csv(f'{PROCESSED}/X_val.csv')
    y_val   = pd.read_csv(f'{PROCESSED}/y_val.csv').squeeze()
    X_test  = pd.read_csv(f'{PROCESSED}/X_test.csv')
    y_test  = pd.read_csv(f'{PROCESSED}/y_test.csv').squeeze()
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    fpr, tpr, _ = skm.roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    return {
        'accuracy':          float(skm.accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(skm.balanced_accuracy_score(y_true, y_pred)),
        'precision':         float(skm.precision_score(y_true, y_pred, zero_division=0)),
        'recall':            float(skm.recall_score(y_true, y_pred, zero_division=0)),
        'f1':                float(skm.f1_score(y_true, y_pred, zero_division=0)),
        'auroc':             float(skm.roc_auc_score(y_true, y_prob)),
        'ks':                ks,
        'mse':               float(np.mean((y_prob - y_true) ** 2)),
    }


def save_results(model_name, y_true, y_pred, y_prob, scores):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    out = os.path.join(RESULTS, model_name)
    os.makedirs(f'{out}/plots', exist_ok=True)

    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}) \
      .to_csv(f'{out}/predictions_test.csv', index=False)

    pd.DataFrame([{'model': model_name, **scores}]) \
      .to_csv(f'{out}/scores.csv', index=False)

    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
    pd.DataFrame([{'model': model_name, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}]) \
      .to_csv(f'{out}/confusion_matrix.csv', index=False)

    _plot_ks(y_true, y_prob, model_name, f'{out}/plots/ks_curve.png')
    _plot_roc(y_true, y_prob, model_name, f'{out}/plots/roc_curve.png')
    _plot_cm(y_true, y_pred, model_name, f'{out}/plots/confusion_matrix.png')
    print(f'Saved → {out}')


def print_scores(scores):
    for k, v in scores.items():
        print(f'  {k:<22} {v:.4f}')


def _plot_ks(y_true, y_prob, title, path):
    df = pd.DataFrame({'y': y_true, 'p': y_prob}).sort_values('p').reset_index(drop=True)
    n0 = (df['y'] == 0).sum()
    n1 = (df['y'] == 1).sum()
    cum0 = (df['y'] == 0).cumsum() / n0
    cum1 = (df['y'] == 1).cumsum() / n1
    diff = (cum1 - cum0).abs()
    ks_val   = diff.max()
    ks_idx   = diff.idxmax()
    ks_thr   = df.loc[ks_idx, 'p']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['p'].values, cum0.values, label='Class 0 (No Churn)', lw=2)
    ax.plot(df['p'].values, cum1.values, label='Class 1 (Churn)', lw=2)
    ax.axvline(ks_thr, color='red', ls='--', label=f'KS = {ks_val:.3f}  @ thr={ks_thr:.3f}')
    ax.set_xlabel('Probability Threshold')
    ax.set_ylabel('Cumulative Distribution')
    ax.set_title(f'KS Curve — {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(y_true, y_prob, title, path):
    fpr, tpr, _ = skm.roc_curve(y_true, y_prob)
    auc = skm.roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cm(y_true, y_pred, title, path):
    cm = skm.confusion_matrix(y_true, y_pred, normalize='true')
    disp = skm.ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=['No Churn', 'Churn'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, values_format='.2%')
    ax.set_title(f'Confusion Matrix — {title}')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
