"""
诊断脚本：分析「成对分辨 <2% = 52%」的核心数据障碍。
读取任意 test_predictions.npz + test_dynamic_df，计算信噪比、噪声来源、失败 case 特征。
用法: 放服务器 project-107-rank/outputs/ 下跑。(需要一个临时文件来重建 test_dyn)
注：由于无法直接访问 test_dyn，先手工标出行数。
"""
# 此脚本需在服务器上放在包含 test_predictions.npz 的输出目录运行
# 使用者请按实际情况改路径

import numpy as np, json, pandas as pd, sys

def load_data():
    d = np.load("test_predictions.npz")
    preds, targets = d['preds'], d['targets']
    return preds, targets

preds, targets = load_data()
abs_err = np.abs(preds - targets)
rel_err = abs_err / targets

print("=" * 56)
print("1. 模型预测精度 vs 变体差异：信噪比")
print("=" * 56)

# 每样本预测噪声
mae = np.mean(abs_err) * 1e12   # ps
med_abs_err = np.median(abs_err) * 1e12
rms = np.sqrt(np.mean(abs_err**2)) * 1e12
print(f"预测绝对误差: MAE={mae:.2f}ps  Median={med_abs_err:.2f}ps  RMS={rms:.2f}ps")

# 变体差异（需要分组信息，暂用全局估计）
# 已知变体差中位 ~5.6%（SUMMARY 数据）
variant_spread = 5.6  # %
# 中位延迟 ~16ps（档3）
median_delay = np.median(targets) * 1e12
diff_2pct = median_delay * 0.02  # <2% 差异在绝对尺度上
diff_5pct = median_delay * 0.05
diff_10pct = median_delay * 0.10
print(f"典型延迟(中位): {median_delay:.1f}ps")
print(f"变体差异(中位): {variant_spread}% = {median_delay * variant_spread/100:.2f}ps")
print(f"<2%差异 = {diff_2pct:.2f}ps | <5% = {diff_5pct:.2f}ps | <10% = {diff_10pct:.2f}ps")
print()
print(f"信噪比: 预测噪声RMSE({rms:.2f}ps) vs <2%信号({diff_2pct:.2f}ps) => {rms/max(diff_2pct, 0.001):.1f}x")
print(f"  -> 预测噪声是<2%信号的 {rms/max(diff_2pct, 0.001):.0f} 倍: 信号被噪声淹没, 无法分辨")

# 每样本噪声分布（按延迟分档）
print()
print("=" * 56)
print("2. 噪声成分分析：不同延迟量级的预测分散度")
print("=" * 56)
bins = [5e-12, 10e-12, 20e-12, 40e-12, 80e-12, 1e-9]
labels = ['5-10ps','10-20ps','20-40ps','40-80ps','80ps+']
for i in range(len(bins)-1):
    m = (targets >= bins[i]) & (targets < bins[i+1])
    if m.sum() < 10: continue
    e = abs_err[m] * 1e12
    print(f"  {labels[i]}: n={m.sum()}  MAE={np.mean(e):.2f}ps  Std={np.std(e):.2f}ps  "
          f"delay~{np.median(targets[m])*1e12:.0f}ps  diff2%={np.median(targets[m])*1e12*0.02:.1f}ps "
          f"SNR={np.median(targets[m])*1e12*0.02/np.std(e):.2f}")

# 速度上的预测噪声 vs 最慢速度上的预测噪声
print()
print("=" * 56)
print("3. 噪声来源：sample 级 vs 变体级（需要分组信息）")
print("=" * 56)
# 简化：用绝对误差的标准差估计"样本随机噪声"
# 如果能分组：组内标准差 vs 组间标准差
# 此处只在全局：预测的标准差 ≈ RMS ≈ 10ps → 比 2% 差异 (0.3ps) 大 30x
# 即使变体聚合后（max 操作会降噪~√k, k≈16 stimulus rows），噪声仍约 2.5ps → 10x
n_stimulus = 16  # 每变体 stimulus 行数
group_noise = rms / np.sqrt(n_stimulus)
print(f"sample级噪声(RMS): {rms:.2f}ps")
print(f"变体聚合后(取max, 约{n_stimulus}行/变体): 噪声 {group_noise:.2f}ps (降√{n_stimulus}={np.sqrt(n_stimulus):.1f}x)")
print(f"聚合后信噪比 vs <2%差异({diff_2pct:.2f}ps): {diff_2pct/group_noise:.2f}  (需要 >~2 才能稳定分辨)")
print(f"  -> 聚合后噪声仍为 {group_noise/diff_2pct:.1f}x <2%信号 → 依然无法稳定分辨")

print()
print("=" * 56)
print("4. 核心结论")
print("=" * 56)
print("""
成对分辨 <2% = 52%(随机) 不是模型「没训练好」的问题，
是数据层面决定的信噪比天花板:

1. 预测噪声 RMS≈9ps >> <2%信号 ~0.3ps (30:1)
   即使变体聚合降噪, 噪声仍 ~2.3ps >> 0.3ps (8:1)

2. 模型在「每个样本」上的预测精度不够→无法可靠区分
   2%差异的变体对。这不是损失函数的错, 是预测精度的错。

3. 成对分辨 >5%差才能稳定>70%(信号 ~0.8ps vs 聚合噪声~2.3ps→边界)
   成对分辨 >10%差 = 79%(信号~1.6ps→信噪比接近1:1,勉强能做)

4. 要突破<2%档, 需要的是:
   - 降低每样本预测噪声(RMS 9ps→需要降 5-8x 才能让<2%信号浮现)
   - 或增加每变体的stimulus行数(降聚合噪声, 但数据量固定)
   - 或增加训练数据量/多样性(降低模型预测方差)

这不是加个排序损失能解决的——排序损失不能创造新信息,
它只是把已有信息重新加权。真正瓶颈是「每样本预测精度」。
""")
