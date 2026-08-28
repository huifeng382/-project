# V2 数据交付说明（满量版，2026-08-28）

> 对应 `DATA_SPEC_V2.md`（Rust 对齐版）全量交付。生成方提交：
> `8ca3004`（P9 形状全覆盖 batch_v2_io）、`6402ff7`（满量 batch_v2_rest + 三批重收集）、
> `41e3346`（`_check_v2_data.py` part 文件支持）。

## 一、交付批次总览

| 批次 | 电路数 | 行数 | expr 组数 | expr 编号 | 内容 |
|---|---|---|---|---|---|
| `batch_v2_full` | 8,679 | 69,432 | 591 | expr8000-8590 | 4 入 1 出（v2_exprs.txt 池） |
| `batch_v2_io` | 4,248 | 57,764 | 394 | expr9000-9393 | 任意 I/O 形状覆盖（46 个 Rust benchmark .tl 家族 + 合成） |
| `batch_v2_rest` | 35,183 | 568,090 | 3,013 | expr9500+ | 满量补齐（分桶合成 + v2_exprs 未覆盖部分） |
| **合计** | **48,110** | **695,286** | **3,981** | — | 单 corner `s02p0_l01p0` |

> `batch_v2_rest` 因 GitHub 100MB 单文件上限，`timing_arcs` 拆为
> `timing_arcs_part1~6.parquet`、`circuit_static` 为 `circuit_static_part1.parquet` 交付；
> `_check_v2_data.py` 已支持 part 文件（显式目录与自动发现均可）。

## 二、规格对照（DATA_SPEC_V2 满量）

| 规格项 | 要求 | 实测 | 状态 |
|---|---|---|---|
| 电路总数 | ~5 万 | 48,110 | ✅ |
| 数据行数 | ~60 万 | 695,286 | ✅ |
| expr 组数 | ~4,000 | 3,981 | ✅ |
| 输入分桶 1~2 | ~25% | 11,796（24.5%） | ✅ |
| 输入分桶 3~4 | ~25% | 12,270（25.5%） | ✅ |
| 输入分桶 5~8 | ~25% | 11,898（24.7%） | ✅ |
| 输入分桶 9~16 | ~25% | 12,146（25.2%） | ✅ |
| 多输出占比 | ≥20% | 10,995（22.9%，输出 2: 8,251 / 3: 2,744） | ✅ |
| corner | 单 `s02p0_l01p0` | 全部一致 | ✅ |
| 每电路行数 | 2 × N_in × M | 0 不匹配（P6 剔除后） | ✅ |
| sc_expansion | 全部 SC_ 名可展开 | 18,002 名，缺 0、展开为空 0（映射表 23,278 项） | ✅ |

## 三、验收结果

接收方脚本 `scripts/diag/_check_v2_data.py`（三批合并）：

```
PASS 46 | FAIL 0 | WARN 1
WARN: ids_charge ≠ ids_avg — 复制残留 34/269,302 激活管（0.013%）
```

**WARN 说明**：`ids_charge` 已按 P5 修复公式 `ids_peak × ids_rise_time / 1000`（fC）重算，
并统一为 **4 位小数**（2026-08-28 三批重收集，此前 2 位小数取整巧合占比 1.15% → FAIL）。
残留的 0.013% 为窄脉冲晶体管（仅在翻转窗口导通）数值上 `peak×rise/1000 ≈ avg` 的物理巧合，
非复制 bug（复检口径见 `DATA_SPEC_V2_ISSUES.md` R3，可忽略）。

其余全部 PASS：值级检查（direction/vector 切换位/大小写）、行数公式、vector 唯一性、
transistor_wave 7 字段 100%、parasitic_caps 子字段 100%、expr 不与 V1 重叠、批次间无重叠、
coverage_report 全部 100%。

## 四、生成侧要点（供复现/审查）

1. **任意 I/O / 多输出链路（Rust）**：新增 `tl_to_recexpr` example（46 个 benchmark .tl →
   s-expr）；`generate_candidates` 修复多输出 Concat 候选 joinize + 枚举空回退；
   `rules.rs` 恢复 `self-join-synthesis` 的 `is_not_concat` 守卫（否则多输出根 eclass 被污染）；
   `generate_templates_v2` 多输出 `output_pins` 覆盖。
2. **表达池**：`spec_pipeline/v2/synthesize_io_exprs.py`（分桶合成，弧完备性精确校验）+
   `build_rest_pool.py`（满量池，3,063 exprs，布局：depth-6 段 + 多输出 n≥5 的 depth-7 尾段）。
3. **collect/finalize**：`collect_v2.py` 每 tb 按 `out_NN` 目录定输出（多输出逐输出一行）；
   `finalize_v2.py` dedup 键含 `output`；`spice_utils.py` 修 PIN 测量硬编码 a/b/c/d（任意 I/O 必需）。
4. **覆盖报告**：`coverage_report_v2.json` 已含三个批次；`vector.one_per_group` 组键含 `output`
   （Rust break 语义为 per-(output,pin) 首个翻转行，多输出下 vector 逐输出不同是正确行为）。

## 五、需接收方知悉的事项

1. **`_check_v2_data.py` 两处配套修改**（已随 commit 推送）：
   - 行身份键（vector 唯一性 / 去重）加入 `output`——多输出每输出一行、vector 逐输出，规格 §三
     「多输出 per-output」的必然要求；
   - 显式目录分支支持 `*_partN.parquet`（batch_v2_rest 拆分交付）。
2. **tl 多输出 benchmark 电路本体（ADD4_OVF 等）不在数据中**：其输出不对所有输入翻转 →
     行数 < 2×N_in×M，被 P6 完整性过滤剔除（规格行数公式的必然结果）；对应的 **18 种 I/O 形状
     （含 9 入 6 出）由合成等效电路覆盖**。若需保留本体，需与规格确认行数公式口径。
3. **n≥9 多输出 expr 每组 1 个候选**（枚举深度限制回退）：约 10% 的组低于 10~15 变体范围
   （组规模检查 96.8% 合规，WARN 阈值内）。
4. `supply_noise_json` 全零为理想电源的合法结果（规格 §六 方案D，报告已注明）。
