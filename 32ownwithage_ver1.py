import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, roc_curve, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import warnings
import random

# --- 配置 ---
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class Config:
    LSTM_HIDDEN_SIZE = 64
    ATTENTION_SIZE = 32
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4

    # 保持较大的正则化，防止过拟合死记硬背
    WEIGHT_DECAY = 5e-3
    EPOCHS = 150
    PATIENCE = 15

    OUTER_FOLDS = 10
    INNER_FOLDS = 5
    K_RANGE = [10, 15, 20, 25, 30, 40, 50, 60]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RAW_FEATURES_PATH = r"E:\msb论文模型\数据\Output\fused_features_32sub_3s_60ch.npy"
    DEMO_DATA_PATH = r"E:\msb论文模型\数据\Input\32.xlsx"
    OUTPUT_DIR = r"E:\msb论文模型\数据\333"


# --- 数据集 ---
class FusionDataset(Dataset):
    def __init__(self, force_features, demographic_features, labels):
        self.force_features = torch.tensor(force_features, dtype=torch.float32)
        self.demographic_features = torch.tensor(demographic_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {'force': self.force_features[idx], 'demographic': self.demographic_features[idx],
                'label': self.labels[idx]}


# --- 模型架构 ---
class Attention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(Attention, self).__init__()
        self.W = nn.Linear(hidden_size, attention_size, bias=False)
        self.V = nn.Linear(attention_size, 1, bias=False)

    def forward(self, hidden_states):
        attn_scores = self.V(torch.tanh(self.W(hidden_states))).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(2) * hidden_states, dim=1)
        return context_vector, attn_weights


class FusionLSTMAttention(nn.Module):
    def __init__(self, input_size, demographic_size, num_classes=2):
        super(FusionLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=Config.LSTM_HIDDEN_SIZE, batch_first=True)
        self.attention = Attention(Config.LSTM_HIDDEN_SIZE + demographic_size, Config.ATTENTION_SIZE)

        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.LSTM_HIDDEN_SIZE + demographic_size, 32),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(32, num_classes))

    def forward(self, force, demographic):
        lstm_out, _ = self.lstm(force)
        if demographic.shape[1] > 0:
            demographic_sequence = demographic.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
            attention_input = torch.cat((lstm_out, demographic_sequence), dim=2)
        else:
            attention_input = lstm_out
        context_vector, _ = self.attention(attention_input)
        return self.classifier(context_vector)


# --- 训练与评估 ---
def train_model(model, train_loader, val_loader, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    train_labels = train_loader.dataset.labels
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = class_counts.sum() / (2.0 * class_counts.clamp_min(1.0))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        for batch in train_loader:
            force, demo, labels = batch['force'].to(device), batch['demographic'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            loss = criterion(model(force, demo), labels)
            loss.backward()
            optimizer.step()

        val_loss, _, _, _, _, _, _ = evaluate_model(model, val_loader, device, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE: break

    if best_model_state: model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, data_loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            force, demo, labels = batch['force'].to(device), batch['demographic'].to(device), batch['label'].to(device)
            outputs = model(force, demo)
            if criterion: total_loss += criterion(outputs, labels).item() * force.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels, all_preds, all_probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    auc_score = roc_auc_score(all_labels, all_probs[:, 1]) if len(np.unique(all_labels)) > 1 else 0.5
    avg_loss = total_loss / len(data_loader.dataset) if criterion else 0
    return avg_loss, accuracy, auc_score, precision, all_labels, all_preds, all_probs


def fit_transform_demographic(train_df, *evaluation_dfs):
    train_features = train_df.copy()
    evaluation_features = [df.copy() for df in evaluation_dfs]
    continuous_cols = ['age', 'height', 'weight', 'FMA', 'Brun']
    cols_to_scale = [col for col in continuous_cols if col in train_features.columns]
    if cols_to_scale:
        scaler = StandardScaler()
        train_features[cols_to_scale] = scaler.fit_transform(train_features[cols_to_scale])
        for features in evaluation_features:
            features[cols_to_scale] = scaler.transform(features[cols_to_scale])
    arrays = [train_features.values.astype(np.float32)]
    arrays.extend(features.values.astype(np.float32) for features in evaluation_features)
    return arrays


def rank_features_on_train_set(X_train, y_train):
    n_samples, n_time_steps, n_channels = X_train.shape
    X_transposed = X_train.transpose(0, 2, 1)
    X_reshaped = X_transposed.reshape(n_samples, -1)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_reshaped, y_train)

    importances = rf_model.feature_importances_
    channel_feature_importances = importances.reshape(n_channels, n_time_steps)
    channel_importances = channel_feature_importances.sum(axis=1)
    return np.argsort(channel_importances)[::-1]


# --- 核心主程序 ---
def run_nested_cv_experiment():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    print(f"[{Config.DEVICE}] 开始严格的嵌套交叉验证 (Nested CV)...")

    model_name = "Proposed_FusionLSTMAttention"
    txt_path = os.path.join(Config.OUTPUT_DIR, f"{model_name}_metrics.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {model_name} Nested CV Metrics ===\n\n")

    X_raw = np.load(Config.RAW_FEATURES_PATH)
    demo_df = pd.read_excel(Config.DEMO_DATA_PATH, index_col=0)

    # 【修正】：将标签顺序改回标准的：前14个是健康(0)，后18个是患者(1)
    labels = np.concatenate([np.zeros(14, dtype=int), np.ones(18, dtype=int)])

    outer_true_labels, outer_pred_labels, outer_pred_probs = [], [], []
    outer_accuracies, outer_aucs = [], []
    selected_k_history = []

    outer_kfold = StratifiedKFold(n_splits=Config.OUTER_FOLDS, shuffle=True, random_state=42)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X_raw, labels)):
        print(f"\n{'=' * 40}")
        print(f" Outer Fold {outer_fold + 1}/{Config.OUTER_FOLDS} 开始")
        print(f"{'=' * 40}")

        X_train_outer, X_test_outer = X_raw[train_idx], X_raw[test_idx]
        demo_train_outer = demo_df.iloc[train_idx].reset_index(drop=True)
        demo_test_outer = demo_df.iloc[test_idx].reset_index(drop=True)
        y_train_outer, y_test_outer = labels[train_idx], labels[test_idx]

        inner_kfold = StratifiedKFold(n_splits=Config.INNER_FOLDS, shuffle=True, random_state=42)
        best_k = None
        best_inner_score = -1

        for k in Config.K_RANGE:
            inner_aucs = []
            for inner_train_idx, inner_val_idx in inner_kfold.split(X_train_outer, y_train_outer):
                inner_rank = rank_features_on_train_set(
                    X_train_outer[inner_train_idx], y_train_outer[inner_train_idx]
                )
                selected_channels = inner_rank[:k]
                X_inner_t = X_train_outer[inner_train_idx][:, :, selected_channels]
                X_inner_v = X_train_outer[inner_val_idx][:, :, selected_channels]
                demo_inner_t, demo_inner_v = fit_transform_demographic(
                    demo_train_outer.iloc[inner_train_idx], demo_train_outer.iloc[inner_val_idx]
                )
                y_inner_t, y_inner_v = y_train_outer[inner_train_idx], y_train_outer[inner_val_idx]

                inner_train_loader = DataLoader(FusionDataset(X_inner_t, demo_inner_t, y_inner_t),
                                                batch_size=Config.BATCH_SIZE, shuffle=True)
                inner_val_loader = DataLoader(FusionDataset(X_inner_v, demo_inner_v, y_inner_v),
                                              batch_size=Config.BATCH_SIZE)

                model = FusionLSTMAttention(input_size=k, demographic_size=demo_inner_t.shape[1])
                trained_model = train_model(model, inner_train_loader, inner_val_loader, Config.DEVICE)

                _, _, auc_score, _, _, _, _ = evaluate_model(trained_model, inner_val_loader, Config.DEVICE)
                inner_aucs.append(auc_score)

            avg_inner_auc = np.mean(inner_aucs)
            if avg_inner_auc > best_inner_score:
                best_inner_score = avg_inner_auc
                best_k = k

        print(f"  -> 内层寻优结束。该折选定最佳 k = {best_k}")
        selected_k_history.append(best_k)

        outer_rank = rank_features_on_train_set(X_train_outer, y_train_outer)
        selected_channels = outer_rank[:best_k]
        X_train_final = X_train_outer[:, :, selected_channels]
        X_test_final = X_test_outer[:, :, selected_channels]

        fit_idx, val_idx = train_test_split(
            np.arange(len(y_train_outer)), test_size=0.2, stratify=y_train_outer,
            random_state=42 + outer_fold
        )
        demo_fit, demo_val, demo_test = fit_transform_demographic(
            demo_train_outer.iloc[fit_idx], demo_train_outer.iloc[val_idx], demo_test_outer
        )
        final_train_loader = DataLoader(FusionDataset(X_train_final[fit_idx], demo_fit,
                                                      y_train_outer[fit_idx]),
                                        batch_size=Config.BATCH_SIZE, shuffle=True)
        final_val_loader = DataLoader(FusionDataset(X_train_final[val_idx], demo_val,
                                                    y_train_outer[val_idx]),
                                      batch_size=Config.BATCH_SIZE)
        final_test_loader = DataLoader(FusionDataset(X_test_final, demo_test, y_test_outer),
                                       batch_size=Config.BATCH_SIZE)

        final_model = FusionLSTMAttention(input_size=best_k, demographic_size=demo_fit.shape[1])
        trained_final_model = train_model(final_model, final_train_loader, final_val_loader, Config.DEVICE)

        _, acc, auc_score, fold_prec, true, pred, probs = evaluate_model(trained_final_model, final_test_loader,
                                                                         Config.DEVICE)

        fold_recall = recall_score(true, pred, zero_division=0)
        fold_f1 = f1_score(true, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"  *** 外侧测试 -> ACC: {acc:.4f}, AUC: {auc_score:.4f} ***")

        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(
                f"Fold {outer_fold + 1} - ACC: {acc:.4f}, AUC: {auc_score:.4f}, Precision: {fold_prec:.4f}, Recall: {fold_recall:.4f}, Specificity: {fold_spec:.4f}, F1: {fold_f1:.4f}, Best K: {best_k}\n")

        outer_accuracies.append(acc)
        outer_aucs.append(auc_score)
        outer_true_labels.extend(true)
        outer_pred_labels.extend(pred)
        outer_pred_probs.extend(probs[:, 1])

    # ================= 实验总结 =================
    overall_auc = roc_auc_score(outer_true_labels, outer_pred_probs)
    overall_recall = recall_score(outer_true_labels, outer_pred_labels)
    overall_prec = precision_score(outer_true_labels, outer_pred_labels)
    overall_f1 = f1_score(outer_true_labels, outer_pred_labels)
    tn_all, fp_all, fn_all, tp_all = confusion_matrix(outer_true_labels, outer_pred_labels).ravel()
    overall_spec = tn_all / (tn_all + fp_all) if (tn_all + fp_all) > 0 else 0.0

    print("\n" + "=" * 50)
    print("      嵌套交叉验证 (Nested CV) 最终报告")
    print("=" * 50)

    # 按照要求的顺序打印
    print(f"ACC:       {np.mean(outer_accuracies):.4f} ± {np.std(outer_accuracies):.4f}")
    print(f"F1-score:  {overall_f1:.4f}")
    print(f"Recall:    {overall_recall:.4f}")
    print(f"Precision: {overall_prec:.4f}")
    print(f"AUC:       {overall_auc:.4f}")

    print(f"\n建议的最终 K 值: {np.median(selected_k_history)}")

    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Overall Results ===\n")
        f.write(f"ACC: {np.mean(outer_accuracies):.4f} ± {np.std(outer_accuracies):.4f}\n")
        f.write(f"F1-score: {overall_f1:.4f}\n")
        f.write(f"Recall: {overall_recall:.4f}\n")
        f.write(f"Precision: {overall_prec:.4f}\n")
        f.write(f"AUC: {overall_auc:.4f}\n")

    cm = confusion_matrix(outer_true_labels, outer_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Stroke'],
                yticklabels=['Healthy', 'Stroke'])
    plt.title('Nested CV Overall Confusion Matrix')
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

    fpr, tpr, _ = roc_curve(outer_true_labels, outer_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {overall_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Nested CV Overall ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f'{model_name}_roc_curve.png'))
    plt.close()

    roc_dict = {'fpr': fpr, 'tpr': tpr, 'auc': overall_auc}
    np.save(os.path.join(Config.OUTPUT_DIR, f"{model_name}_roc.npy"), roc_dict)
    print(f"\n所有图表已保存至: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    run_nested_cv_experiment()
