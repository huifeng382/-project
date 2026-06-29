import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from config import *
from src.utils import set_seed, split_by_circuit

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

set_seed(RANDOM_SEED)

# 创建输出目录
os.makedirs("data_quality_report", exist_ok=True)

# ---------- 数据加载与规范化 ----------
def normalize_dynamic(df):
    if 'circuit_id' not in df.columns:
        if 'candidate' in df.columns:
            df = df.rename(columns={'candidate': 'circuit_id'})
        elif 'candidate_id' in df.columns:
            df = df.rename(columns={'candidate_id': 'circuit_id'})
    df['circuit_id'] = df['circuit_id'].astype(str)
    if 'DELAY' not in df.columns:
        for col in ['delay', 'delay_s', 'Delay', 'delays']:
            if col in df.columns:
                df = df.rename(columns={col: 'DELAY'})
                break
    return df

def normalize_static(df):
    if 'circuit_id' not in df.columns:
        if 'candidate' in df.columns:
            df = df.rename(columns={'candidate': 'circuit_id'})
        elif 'candidate_id' in df.columns:
            df = df.rename(columns={'candidate_id': 'circuit_id'})
    df['circuit_id'] = df['circuit_id'].astype(str)
    return df

def safe_parse_loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return {}
    elif isinstance(x, dict):
        return x
    else:
        return {}

# 加载数据
static_parquets = glob.glob("data/batch_*/circuit_static.parquet")
dynamic_parquets = glob.glob("data/batch_*/timing_arcs.parquet")
if not static_parquets or not dynamic_parquets:
    raise FileNotFoundError("No Parquet files found.")

dynamic_dfs = []
for p in dynamic_parquets:
    df = pd.read_parquet(p)
    df = normalize_dynamic(df)
    dynamic_dfs.append(df)
dynamic_df = pd.concat(dynamic_dfs, ignore_index=True)
dynamic_df = dynamic_df.dropna(subset=['circuit_id', 'DELAY'])
dynamic_df = dynamic_df[dynamic_df['DELAY'] > 0]
dynamic_df = dynamic_df[(dynamic_df['DELAY'] > 1e-12) & (dynamic_df['DELAY'] < 1e-8)]

static_dfs = []
for p in static_parquets:
    df = pd.read_parquet(p)
    df = normalize_static(df)
    static_dfs.append(df)
static_df = pd.concat(static_dfs).drop_duplicates('circuit_id').set_index('circuit_id')

if 'pin_loads_json' in static_df.columns:
    static_df['pin_loads_dict'] = static_df['pin_loads_json'].apply(safe_parse_loads)
else:
    static_df['pin_loads_dict'] = [{}] * len(static_df)

# 划分数据集
circuit_ids = dynamic_df['circuit_id'].unique().tolist()
train_ids, val_ids, test_ids = split_by_circuit(circuit_ids, seed=RANDOM_SEED)
train_df = dynamic_df[dynamic_df['circuit_id'].isin(train_ids)]
val_df = dynamic_df[dynamic_df['circuit_id'].isin(val_ids)]
test_df = dynamic_df[dynamic_df['circuit_id'].isin(test_ids)]

print("="*70)
print("数据质量检验报告")
print("="*70)

# ----- 1. 目标变量分布 -----
def plot_target_dist(df, name):
    y = df['DELAY'].values
    log_y = np.log10(y + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y, bins=50, log=True)
    axes[0].set_title(f'{name} DELAY 分布 (线性)')
    axes[0].set_xlabel('DELAY (s)')
    axes[0].set_ylabel('频率')
    axes[1].hist(log_y, bins=50)
    axes[1].set_title(f'{name} log10(DELAY) 分布')
    axes[1].set_xlabel('log10(DELAY)')
    plt.tight_layout()
    plt.savefig(f'data_quality_report/target_dist_{name}.png', dpi=300)
    plt.close()

plot_target_dist(train_df, "训练集")
plot_target_dist(val_df, "验证集")
plot_target_dist(test_df, "测试集")

# ----- 2. 训练/验证/测试分布一致性 -----
log_train = np.log10(train_df['DELAY'].values + 1e-12)
log_val = np.log10(val_df['DELAY'].values + 1e-12)
log_test = np.log10(test_df['DELAY'].values + 1e-12)
ks_train_val = ks_2samp(log_train, log_val)
ks_train_test = ks_2samp(log_train, log_test)
print(f"\nK-S 检验: 训练 vs 验证 p-value = {ks_train_val.pvalue:.6f}")
print(f"K-S 检验: 训练 vs 测试 p-value = {ks_train_test.pvalue:.6f}")
if ks_train_val.pvalue < 0.05:
    print("  -> 警告：训练集与验证集分布显著不同")
if ks_train_test.pvalue < 0.05:
    print("  -> 警告：训练集与测试集分布显著不同")

# ----- 3. 特征与目标相关性 -----
pins = sorted({c.split('_')[1] for c in train_df.columns if c.startswith('slew_') and c != 'slew_s'})
if not pins:
    pins = sorted(train_df['switching_pin'].dropna().unique())
print(f"\n检测到引脚: {pins}")

X, y = [], []
for idx, row in train_df.iterrows():
    cid = row['circuit_id']
    pin_loads = static_df.loc[cid, 'pin_loads_dict']
    features = []
    for pin in pins:
        features.append(row.get(f'slew_{pin}', row.get('slew_s', 0.0)))
        features.append(pin_loads.get(pin, 0.0))
    features.append(row.get('output_load_f', 0.0))
    X.append(features)
    y.append(row['DELAY'])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float64)
log_y = np.log10(y + 1e-12)

corr_list = []
for i in range(X.shape[1]):
    p, _ = pearsonr(X[:, i], log_y)
    s, _ = spearmanr(X[:, i], log_y)
    corr_list.append((i, p, s))
corr_list.sort(key=lambda x: abs(x[1]), reverse=True)
print("\n特征与目标 (log10 DELAY) 皮尔逊相关性 Top 10:")
for idx, p, s in corr_list[:10]:
    print(f"  特征 {idx}: 皮尔逊 = {p:.4f}, 斯皮尔曼 = {s:.4f}")

# ----- 4. 电路级别的变异 -----
circuit_stats = train_df.groupby('circuit_id')['DELAY'].agg(['mean', 'std', 'count'])
print(f"\n电路内延迟变异（训练集）:")
print(f"  电路数: {len(circuit_stats)}")
print(f"  每个电路样本数: 最少 {circuit_stats['count'].min()}, 最多 {circuit_stats['count'].max()}, 平均 {circuit_stats['count'].mean():.1f}")
print(f"  电路内延迟标准差: 均值 {circuit_stats['std'].mean():.3e}, 中位数 {circuit_stats['std'].median():.3e}")
print(f"  电路间延迟均值: 均值 {circuit_stats['mean'].mean():.3e}, 标准差 {circuit_stats['mean'].std():.3e}")

# ----- 5. 特征方差分析 -----
variances = np.var(X, axis=0)
const_features = np.sum(variances < 1e-12)
print(f"\n常数特征数: {const_features} / {X.shape[1]}")
if const_features > 0:
    print("  -> 警告：存在常数特征，它们不提供任何信息")

# ----- 6. 特征缺失情况 -----
missing_cols = train_df.isnull().sum()
if missing_cols.sum() > 0:
    print("\n缺失值列：")
    print(missing_cols[missing_cols > 0])

# ----- 7. 目标值动态范围 -----
print(f"\n目标值动态范围（训练集）: {np.min(y):.3e} ~ {np.max(y):.3e}, 比值 {np.max(y)/np.min(y):.1f}")
if np.max(y)/np.min(y) > 100:
    print("  -> 动态范围过大，可能影响模型收敛")

# ----- 8. 综合结论 -----
print("\n" + "="*70)
print("结论与建议:")
issues = []
if ks_train_val.pvalue < 0.05 or ks_train_test.pvalue < 0.05:
    issues.append("训练/验证/测试集分布不一致，建议重新划分或使用分层采样。")
if const_features > 0:
    issues.append(f"存在 {const_features} 个常数特征，应删除或重新生成。")
if np.max(y)/np.min(y) > 100:
    issues.append("目标值动态范围过大，建议进一步截断或使用更强变换（如 Box-Cox）。")
if np.mean(circuit_stats['std']) < 1e-12:
    issues.append("电路内延迟几乎恒定，缺乏多样性，可能限制模型泛化。")
if any(abs(p) < 0.1 for _, p, _ in corr_list[:5]):
    issues.append("特征与目标相关性极低（<0.1），特征信息不足，建议增加更有效的特征。")

if issues:
    print("发现以下问题，建议重新生成数据或进行数据增强：")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("数据质量尚可，问题可能源于模型容量或损失函数，建议调整模型。")

print("\n详细图表已保存至 data_quality_report/ 目录。")