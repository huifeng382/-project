//! gnn_shadow.rs — GNN 粗筛 shadow 并行评估（Rust 侧接入参考实现）
//!
//! 用途：Rust 贪心优化（NetlistOpt）并行跑 GNN 粗筛 + SPICE 仿真，积累 GNN vs SPICE
//! 对照数据，按 docs/GNN_RUST_DATA_DIFF.md 第十节的标准判定「GNN 是否可替换逐候选 SPICE 排序」。
//!
//! 接入步骤：
//!   1) 拷入 NetlistOpt/src/，在 lib.rs（或 main.rs）加 `mod gnn_shadow;`
//!   2) Cargo.toml [dependencies] 加：
//!        serde = { version = "1", features = ["derive"] }
//!        serde_json = "1"
//!   3) 在 tl_opt.rs 的 optimize_tl_module 里，当 env GNN_SHADOW=1 时用
//!      ShadowGnnTlEvaluator 包住 FullCircuitTlEvaluator（见文末「挂载示例」）。
//!
//! 行为：evaluate(candidate) = 先调 GNN /rank 得预测 avg_delay，再调内部 SPICE 得真值，
//! 记录 (gnn_pred, true_delay) 到 CSV，**返回 SPICE 真值**（贪心决策照旧，不改行为）。

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::design::Design;
use crate::spice::expr_to_hierarchical_spice;
use crate::tl_opt::{FullCircuitTlEvaluator, TlEvalMetrics, TlEvaluator, TlOptError};
use crate::utils::tl_graph::TlModule;

/// 发给 GNN serve.py 的候选（对应 scripts/diag/serve_http.py 的 POST /rank）
#[derive(Serialize)]
struct GnnCandidate {
    id: String,
    netlist: String,
    input_pins: Vec<String>,
    output_pins: Vec<String>,
}

#[derive(Deserialize)]
struct RankResponse {
    ranked: Vec<RankEntry>,
}
#[derive(Deserialize)]
struct RankEntry {
    id: String,
    avg_delay: Option<f64>,
}

/// 零依赖 HTTP 客户端：用 std::net::TcpStream 发原始 HTTP/1.1 POST。
pub struct GnnClient {
    host: String,
    port: u16,
    timeout_ms: u64,
}

impl GnnClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self { host: host.to_string(), port, timeout_ms: 15_000 }
    }

    /// 批量排候选，返回 id -> 预测 avg_delay。
    pub fn rank(&self, cands: &[GnnCandidate]) -> Result<HashMap<String, f64>, String> {
        let body = serde_json::to_vec(&json!({"candidates": cands})).map_err(|e| e.to_string())?;
        let mut stream =
            TcpStream::connect((self.host.as_str(), self.port)).map_err(|e| e.to_string())?;
        let to = std::time::Duration::from_millis(self.timeout_ms);
        let _ = stream.set_read_timeout(Some(to));
        let _ = stream.set_write_timeout(Some(to));
        let req = format!(
            "POST /rank HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.host, self.port, body.len()
        );
        stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
        stream.write_all(&body).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&buf);
        let payload = text.find("\r\n\r\n").map(|i| &text[i + 4..]).unwrap_or(&text);
        let resp: RankResponse = serde_json::from_str(payload).map_err(|e| format!("parse json: {e}; body={payload}"))?;
        Ok(resp.ranked.into_iter().map(|r| (r.id, r.avg_delay.unwrap_or(f64::NAN))).collect())
    }
}

/// TlModule -> GNN 候选（netlist 用 expr_to_hierarchical_spice，引脚用 inorder/outorder）。
fn module_to_candidate(module: &TlModule, id: &str) -> Result<GnnCandidate, TlOptError> {
    let expr = module.to_recexpr()?;
    let netlist = expr_to_hierarchical_spice(&expr, &module.outorder);
    Ok(GnnCandidate {
        id: id.to_string(),
        netlist,
        input_pins: module.inorder.clone(),
        output_pins: module.outorder.clone(),
    })
}

/// Shadow 并行评估：GNN 预测 + SPICE 真值都跑，记录对照，返回真值（决策不改）。
pub struct ShadowGnnTlEvaluator {
    inner: FullCircuitTlEvaluator,
    gnn: GnnClient,
    log: std::fs::File,
    debug: bool,
}

impl ShadowGnnTlEvaluator {
    pub fn new(
        design_template: &Design,
        host: &str,
        port: u16,
        log_path: &Path,
    ) -> Result<Self, TlOptError> {
        let mut inner = FullCircuitTlEvaluator::new(Design::new(design_template.rec_expr.clone()));
        inner.design_template.simulation_cfg = design_template.simulation_cfg.clone();
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .map_err(|e| TlOptError::Evaluation(e.to_string()))?;
        Ok(Self { inner, gnn: GnnClient::new(host, port), log, debug: false })
    }

    pub fn with_debug(mut self, on: bool) -> Self {
        self.debug = on;
        self
    }
}

impl TlEvaluator for ShadowGnnTlEvaluator {
    fn evaluate(
        &mut self,
        module: &TlModule,
        eval_idx: usize,
        iter_idx: usize,
    ) -> Result<TlEvalMetrics, TlOptError> {
        // 1) GNN 预测（失败不阻塞，pred=NaN 记录）
        let id = format!("e{eval_idx}");
        let pred = module_to_candidate(module, &id)
            .ok()
            .and_then(|c| self.gnn.rank(&[c]).ok())
            .and_then(|m| m.get(&id).copied())
            .unwrap_or(f64::NAN);
        // 2) SPICE 真值（决策照旧）
        let metrics = self.inner.evaluate(module, eval_idx, iter_idx)?;
        // 3) 记录对照
        let _ = writeln!(
            self.log,
            "eval_idx={eval_idx}, iter={iter_idx}, gnn_pred={pred:.6e}, true_delay={:.6e}, transistors={}",
            metrics.avg_delay, metrics.transistor_count
        );
        if self.debug {
            eprintln!("[gnn_shadow] e{eval_idx} gnn={pred:.6e} true={:.6e}", metrics.avg_delay);
        }
        Ok(metrics)
    }
}

// ===== 挂载示例（tl_opt.rs 的 optimize_tl_module 里，按 env GNN_SHADOW 切换）=====
// use crate::gnn_shadow::ShadowGnnTlEvaluator;
// let shadow = std::env::var("GNN_SHADOW").is_ok();
// let mut evaluator: Box<dyn TlEvaluator> = if shadow {
//     Box::new(ShadowGnnTlEvaluator::new(
//         design, "10.20.34.16", 8000,
//         &design.simulation_cfg.simulation_path.join("gnn_shadow.csv"),
//     )?)
// } else {
//     Box::new(FullCircuitTlEvaluator::new(Design::new(design.rec_expr.clone()))
//         .with_debug_logging(search_params.debug_logging))
// };
// evaluator.design_template.simulation_cfg = design.simulation_cfg.clone();  // 若字段公开
