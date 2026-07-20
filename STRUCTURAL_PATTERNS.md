# Structural Patterns for Delay-Optimal Circuit Variant Selection

> **Purpose**: Reference for a greedy circuit optimization algorithm. When applying
> equivalence-preserving transformations to a circuit, these empirically observed
> structural patterns indicate which transformations are most likely to reduce
> propagation delay. The patterns are derived from statistical analysis of 1,005
> circuits (archive_v13.1 dataset), comparing the fastest vs. slowest functionally
> equivalent variants within each expression group.
>
> **Methodology**: For each `(expr, corner)` group, we computed the worst-case
> delay (max over switching pins, directions, and input vectors) for every variant.
> Among 2,559 variant groups (median intra-group spread = 6.6%), 1,028 groups
> (40%) had a spread exceeding 10% — meaning the best variant was at least 10%
> faster than the worst. Within these high-spread groups, we compared structural
> features of the fastest vs. slowest variants to identify recurring patterns.
>
> **Caveat**: These are statistical associations observed in the current dataset,
> not deterministic physical rules. A pattern that holds in 83% of cases may not
> hold in the remaining 17%. Use these as *priorities for transformation attempts*,
> not as hard constraints that override simulation or model predictions.
>
> **Update policy**: Append new observations when additional data becomes
> available. Do not delete old entries unless proven wrong by new evidence.

---

## 1. Transistor Count Is the Strongest Signal

**Observation**: In 83% of high-spread groups, the fastest variant has fewer
transistors than the slowest variant. The median difference is 8 transistors.
Gate count, however, is nearly identical (median difference = 0).

| Metric | Value |
|---|---|
| Fraction where fastest has fewer transistors | 83% |
| Median transistor difference (worst − best) | 8 |
| Median gate difference (best − worst) | 0 |

**Interpretation**: Structural optimizations that reduce transistor count —
while maintaining the same gate-level function — are the single most reliable
predictor of lower delay. The gate count stays similar because the optimization
typically replaces complex multi-transistor gates with simpler equivalents
rather than eliminating logic stages.

**Greedy algorithm guidance**: When choosing between candidate transformations,
prefer those that reduce total transistor count. This is not a guarantee, but it
is the single strongest statistical predictor available.

---

## 2. Gate Type Preference: `SC_INV_WIRE` Over `SC_AND`

**Observation**: Certain gate types appear disproportionately in the fastest
variants (BEST+) or the slowest variants (WORST+) across high-spread groups.
The numbers below are occurrence counts across all high-spread groups.

### Gate types strongly associated with LOWER delay (BEST+)

| Gate Type | Best Count | Worst Count | Delta |
|---|---|---|---|
| `SC_INV_WIRE` | 352 | 252 | **+100** |
| `SC_JOIN_OR_OR` | 180 | 128 | **+52** |
| `SC_JOIN_v1` | 73 | 32 | **+41** |
| `SC_JOIN_AND_OR_OR_AND_OR_OR` | 41 | 0 | **+41** |
| `SC_JOIN_OR_BRIDGE` | 38 | 4 | **+34** |

### Gate types strongly associated with HIGHER delay (WORST+)

| Gate Type | Best Count | Worst Count | Delta |
|---|---|---|---|
| `SC_AND` | 41 | 208 | **-167** |
| `SC_JOIN` | 446 | 583 | **-137** |
| `SC_INV` | 676 | 776 | **-100** |
| `SC_BRIDGE` | 79 | 142 | **-63** |
| `SC_OR` | 83 | 126 | **-43** |

**Interpretation**:
- `SC_INV_WIRE` (single inverter-buffer) is the most consistent winner. It
  provides buffering with minimal transistor overhead.
- `SC_AND` is the most consistent loser. As a complex gate requiring multiple
  transistors per instance, it introduces more delay than equivalent
  decompositions (e.g., NAND2 + INV).
- `SC_INV` appears more frequently in slower variants, likely because standalone
  inverters are often inserted unnecessarily in suboptimal designs.
- Complex `SC_JOIN_*` chains with many stages appear more in slower variants,
  while simpler join structures (`SC_JOIN_OR_OR`, `SC_JOIN_v1`) appear more in
  faster ones.

**Greedy algorithm guidance**:
1. Prioritize transformations that replace `SC_AND` with its LIB expansion
   (NAND2 + INV) or with `SC_INV_WIRE` structures.
2. Prefer `SC_INV_WIRE` over standalone `SC_INV` where buffering is needed.
3. When using `SC_JOIN` chains, minimize the number of串联 stages. Simpler
   join structures (`SC_JOIN_OR_OR`) are systematically faster than complex
   chained variants.

---

## 3. NOR-Chain Structures (`SC_JOIN_OR_OR`) Outperform Direct OR Gates

**Observation**: `SC_JOIN_OR_OR` (two NOR2 gates chained, implementing a
buffered OR) appears significantly more often in fast variants (+52 delta).
Direct `SC_OR` gates appear more often in slow variants (−43 delta).

| Structure | Best Count | Worst Count | Delta |
|---|---|---|---|
| NOR2 chain (`SC_JOIN_OR_OR`) | 180 | 128 | +52 |
| Direct OR (`SC_OR`) | 83 | 126 | −43 |

**Interpretation**: In the ASAP7 standard cell library used by these circuits,
a NOR2 chain provides better drive strength and lower delay than a direct OR
gate. This is a technology-specific finding — it may not generalize to other
PDKs.

**Greedy algorithm guidance**: When the netlist contains `SC_OR`, attempt
replacement with `SC_JOIN_OR_OR` (two NOR2 gates in series with an inverter).

---

## 4. Transistor Reduction Without Gate Elimination

**Observation**: Fastest variants achieve lower transistor counts while
maintaining nearly identical gate counts (median gate difference = 0). This
indicates that the optimization works at the *transistor level within each gate*
rather than by removing logic stages.

**Interpretation**: The effective transformations replace multi-transistor
complex gates (e.g., `SC_AND` with 6+ transistors) with simpler gate
implementations (e.g., NAND2 with 4 transistors, INV with 2 transistors) that
implement the same logic function through a different gate-level decomposition.

**Greedy algorithm guidance**: For each gate instance in the critical path,
consider its standard-cell transistor count. Prefer transformation candidates
that reduce the *sum of transistor counts* along the critical path, even if
the number of gate instances stays the same or increases slightly.

---

## 5. Chain Length Minimization

**Observation**: Among `SC_JOIN_*` variants, shorter chain names (fewer串联
stages) appear more often in best variants. Complex chained structures with
many stages are systematically slower.

**Interpretation**: Each additional串联 stage adds its own gate delay. For
functionally equivalent transformations, fewer serial stages means lower
cumulative delay.

**Greedy algorithm guidance**: When generating candidate transformations,
favor those that reduce the maximum logic depth (number of gates on the
longest path from any input to the output). This aligns with standard digital
design intuition but is empirically confirmed in this dataset.

---

## Summary of Actionable Guidance

| Priority | Guidance | Confidence |
|---|---|---|
| 1 | Reduce total transistor count | **83%** of high-spread groups |
| 2 | Replace `SC_AND` → NAND2 + INV (or `SC_INV_WIRE`) | Strong gate-type signal (−167 delta) |
| 3 | Prefer `SC_INV_WIRE` for buffering | Strong gate-type signal (+100 delta) |
| 4 | Replace `SC_OR` → `SC_JOIN_OR_OR` (NOR2 chain) | Consistent gate-type signal |
| 5 | Minimize logic depth (longest input-to-output path) | Physics-aligned + data-supported |

These are intended as **transformation priorities** for a greedy optimization
loop — the algorithm should attempt transformations in this order, use a delay
model (or SPICE simulation) to verify the result, and keep the transformation
only if it actually reduces delay.

---

## New Data Update (delivery1+2, 2026-07-20)

> On the combined delivery1+2 dataset (1,437 circuits, ~543K rows, 300 exprs),
> the structural patterns are **substantially different** from the old data.
> The dataset contains far more diverse circuits with much larger variant spreads
> (median 17.7% vs 6.6% on old data). Key findings:

### Statistics
- Groups: 4,320 (was 2,559 on old data). High-spread (>10%): 2,681 groups from 106 exprs.
- Median spread: **17.7%** (was 6.6%). Median TC diff (worst − best): **0** (was 8).
- Fewer TC in fastest: **30%** only (was 83% on old data).

### Gate type preferences (high-spread groups, delivery1+2)

**BEST+ (appearing more in fast variants):**
| Gate Type | Best Count | Worst Count | Delta |
|---|---|---|---|
| SC_JOIN_OR_OR_AND_WIRE_AND_AND_WIRE_AND_AND_BRIDGE | 60 | 0 | +60 |
| SC_JOIN_OR_BRIDGE | 115 | 56 | +59 |
| SC_OR | 926 | 870 | +56 |
| SC_JOIN | 2,028 | 1,982 | +46 |
| SC_JOIN_OR_OR | 762 | 720 | +42 |
| SC_AND | 260 | 218 | +42 |

**WORST+ (appearing more in slow variants):**
| Gate Type | Best Count | Worst Count | Delta |
|---|---|---|---|
| SC_JOIN_OR_OR_AND_WIRE_AND_AND_WIRE_AND_AND_WIRE_B | 3 | 68 | −65 |
| SC_JOIN_BRIDGE | 12 | 59 | −47 |
| SC_BRIDGE | 784 | 817 | −33 |

### Cross-dataset comparison: what changed and what stayed

| Pattern | Old Data (1,005 circuits) | New Data (1,437 circuits) | Transfer? |
|---|---|---|---|
| Transistor count | **83%** fewer = faster | **30%** fewer = faster | **Broke** — not reliable |
| SC_INV_WIRE | Strong BEST+ (+100) | Not in top 8 | **Broke** |
| SC_AND | Strong WORST+ (−167) | NOW in BEST+ (+42) | **Reversed!** |
| SC_JOIN_OR_OR | BEST+ (+52) | Still BEST+ (+42) | **Transferred** |
| Complex chained JOIN | WORST+ | Still WORST+ | **Transferred** |
| SC_BRIDGE | WORST+ (−63) | Still WORST+ (−33, + −47 for JOIN_BRIDGE) | **Transferred** |

### Interpretation of cross-dataset differences
1. **The new data has fundamentally different circuit topologies.** Old data was
   dominated by small manual designs with clear transistor-count signals. New
   data includes much more complex e-graph-generated circuits where transistor
   count alone is not a reliable predictor.
2. **SC_AND reversed**: In old data, SC_AND was consistently slow. In new data,
   it appears more in fast variants — likely because in the new dataset context,
   SC_AND implementations are well-optimized by the synthesis tool, and the
   real bottleneck is elsewhere (bridging structures, chain complexity).
3. **What consistently matters across both datasets**: simpler JOIN structures
   are faster; complex chained JOINs and bridging structures are slower. These
   are the most robust, transferable patterns.
4. **The new data validates the transistor-wave finding** (delivery1+2 ablation):
   when structural patterns no longer provide a strong signal (TC diff=0, few
   gate-type signals), transistor-level physical data becomes the decisive
   differentiator.
