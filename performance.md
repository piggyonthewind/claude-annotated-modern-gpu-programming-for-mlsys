# Annotated Read: "Performance — What Makes a Kernel Fast"

A companion to <https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_performance/index.html>

**Audience:** same as the Background companion — someone who writes ordinary
software but has never studied GPUs. This chapter assumes you've absorbed the
factory picture from Background, so we reuse it throughout:

> **workers** = threads · **team** = CTA · **workstation** = SM · **team's table**
> = shared memory (SMEM) · **faraway warehouse** = global memory (GMEM/HBM) ·
> **delivery machine** = TMA · **special matrix machine** = Tensor Core.

If any of those are fuzzy, read [`background.md`](background.md) first. Every new
performance term below is still defined from zero.

The chapter answers **one question** and gives you a tool to answer it:

> **What is actually slowing my kernel down — moving the data, or doing the math?**

Everything else (roofline, arithmetic intensity, fusion, occupancy) exists to answer
that question and then act on the answer. The chapter's seven sections:

1. The Roofline Model
2. Arithmetic Intensity of Common Workloads
3. Optimizing Memory-Bound Kernels
4. The Optimization Ladder
5. Reducing Idle Time Through Overlap
6. Occupancy and Resource Pressure
7. Using Roofline to Guide Optimization

---

## 0. Why "is it math or movement?" is the whole game

From Background you know the two expensive things a GPU does: **move numbers**
between the warehouse and the workers, and **do arithmetic** on them. A kernel is
always ultimately limited by *one* of these two — it either spends most of its time
**waiting for data to arrive** (movement-limited) or **waiting for the math to
finish** (math-limited). It cannot be limited by both at once; one of them is the
bottleneck.

Why this matters so much: **the two problems have opposite fixes.** If you're
waiting on data, adding more math machines does nothing — they'd just sit idle
waiting too. If you're waiting on math, moving data faster does nothing — the data's
already there. So *before optimizing anything*, you must find out which case you're
in. Optimizing the wrong one is wasted effort. The whole chapter is a method for
finding out and then targeting the real limit.

---

## 1. The Roofline Model — your two speed limits on one chart

A GPU has exactly **two hard ceilings** on how fast a kernel can run. The roofline
model just draws both of them so you can see which one you're hitting.

### Ceiling A — the machine ceiling (peak compute)
The matrix machines and worker hands can only do so many arithmetic operations per
second, full stop. The unit is **FLOP/s** — floating-point operations per second.

> **FLOP** = one floating-point operation: one add *or* one multiply. (A fused
> multiply-add, `a*b+c`, counts as **2** FLOPs — a multiply and an add.) **FLOP/s**
> is how many of those the hardware can do per second.

For a B200 GPU the chapter uses a rounded **~2 PFLOP/s** (2×10¹⁵ FLOP/s) of dense
fp16/bf16 Tensor-Core throughput. No kernel can exceed that, ever. On the chart it's
a **flat horizontal line** — a hard ceiling of pure compute.

### Ceiling B — the road ceiling (memory bandwidth)
Numbers have to travel from the warehouse (HBM) to the workers, and the road has a
fixed width. The unit is **bytes/s** — the **memory bandwidth**.

> **Bandwidth** = how many bytes/second can stream between warehouse and chip. For a
> B200 the chapter uses **~8 TB/s** (8×10¹² bytes/s) of HBM3e bandwidth. (Don't
> confuse this with **latency** — the *delay* of one trip. Bandwidth is the road's
> *width*, latency is its *length*. Background's deep pipelines hide latency; roofline
> is about bandwidth.)

To actually *reach* that bandwidth roof you need **many memory moves in flight at
once** — one tile fetch is slow (high latency), so if you waited for each before
starting the next, the road would sit mostly empty. Concurrency comes from two places:
*within one team*, the deep pipeline (Background §4) keeps several tiles loading at once
(capped by SMEM space); *across the GPU*, the ~100+ workstations each run teams issuing
their own loads, so **hundreds of fetches are on the road simultaneously**. Roughly
*bytes-in-flight = bandwidth × latency* — you keep enough trips staggered on the road to
keep it full. More in-flight loads don't beat the bandwidth ceiling; they're how you
*reach* it.

Here's the key: how fast bandwidth *lets you compute* depends on **how much math you
squeeze out of each byte you haul in.** If every byte feeds a lot of math, the road
easily keeps the machines fed; if every byte feeds only a little math, the road
starves them. That "math per byte" is the star of this chapter (§2). On the chart the
memory ceiling is a **sloped line**: the more math-per-byte, the higher it rises.

### Putting both on the chart
The roofline is those two lines together. Your kernel's max possible speed is:

```
attainable FLOP/s  ≤  min( peak compute ,  bandwidth × (math per byte) )
                           └─ flat roof ─┘   └──── sloped roof ─────┘
```

You live under whichever line is lower. Where the sloped road-ceiling meets the flat
machine-ceiling is the **ridge point** — the exact "math per byte" at which the road
can *just* keep the machines fully fed:

> **ridge point = peak compute ÷ bandwidth.** For B200: `2×10¹⁵ ÷ 8×10¹² ≈ 250`
> **FLOP per byte.**

That single number splits the world in two:

- **Below 250 FLOP/byte → memory-bound.** You're on the sloped road-ceiling: not
  enough math per byte, so the road can't feed the machines and they idle. Your limit
  is *data movement*.
- **Above 250 FLOP/byte → compute-bound.** You're under the flat machine-ceiling: the
  road easily keeps up, and the machines are the bottleneck. Your limit is *math*.

---

## 2. Arithmetic Intensity — the one number that decides everything

"Math per byte" has a name:

> **Arithmetic Intensity (AI)** = total arithmetic ÷ total bytes moved to/from the
> warehouse = **FLOP / byte.** It measures how much useful math you extract from each
> byte you pay to haul in from HBM.

AI is the x-axis of the roofline. Compute your kernel's AI, compare it to the ridge
point (250 on B200), and you instantly know which regime you're in. That's the whole
diagnostic.

Analogy: AI is **how much cooking you do per trip to the warehouse.** Walk to the
warehouse, grab one onion, walk back, do one chop, walk back for the next onion —
terrible AI, you're just commuting (memory-bound). Grab a full crate and cook an
entire banquet from it before the next trip — great AI, you're actually cooking
(compute-bound). Raising AI = **more cooking per commute.**

Now the chapter's point: **different workloads have wildly different AI**, so they
sit in different places on the roofline.

### Elementwise and reductions — low AI, always memory-bound
An **elementwise** op does something to each element independently, e.g. `y = x + 1`
or a GELU. For each element you read a few bytes, do ~1 arithmetic op, write a few
bytes. That's roughly **1 FLOP per ~8 bytes** → AI far below 1 → *deep* in
memory-bound territory. A **reduction** (e.g. summing an array) is the same story:
one add per element read. These kernels are pure commuting — the machines are almost
idle; you're limited entirely by how fast bytes stream in. **For these, faster
Tensor Cores are useless; only moving fewer/faster bytes helps.**

### GEMM — AI *grows* with size, crosses into compute-bound
Matrix multiply is special because each byte you load gets **reused many times**.
Here's the actual arithmetic for a square N×N × N×N GEMM (this is worth seeing once):

```
FLOPs  = 2·N³         (an N³ grid of multiply-adds, 2 FLOP each)
bytes  = 3·N²·s       (read A and B, write C: 3 matrices of N² elements, s bytes each)

AI = 2N³ / (3N²·s) = 2N / (3s)
```

For fp16 (`s = 2` bytes) the 2 and the `s` cancel: **AI ≈ N/3 FLOP/byte.** So GEMM's
intensity *rises with the matrix size*. Small matrices are memory-bound; once N is
big enough that `N/3 > 250` (i.e. N ≳ 750), the *same operation* becomes
compute-bound. **This is why GEMM is the workload that actually uses those 2 PFLOP/s
— and why big matmuls are where GPUs shine.** Each loaded byte feeds O(N) worth of
math, so you cook a banquet per commute.

(There's a second, related lever the chapter shows with concrete tile numbers: how
big a *tile* you stage on the team's table. A 16×16 output tile with BK=64 in fp16
gives ~**8 FLOP/byte**; a 64×64 tile gives ~**32 FLOP/byte**. Bigger tiles reuse each
loaded input more → higher AI. More on this in §3.)

### Attention — in the middle, and Flash Attention moves it
Attention sits between the two extremes. The naive version has a hidden killer: it
computes a big **scores matrix** (every query against every key), **writes it all the
way out to the warehouse**, reads it back to softmax it, writes again, reads again to
multiply by values. All those giant round-trips are bytes moved with little math per
byte → drags AI down toward memory-bound.

> **Flash Attention** is the famous fix: it **never writes the big scores matrix to
> the warehouse.** It keeps each tile of scores on the team's table (SMEM), does the
> softmax-and-multiply right there, and only writes the small final result out. By
> eliminating the huge intermediate round-trips it slashes bytes-moved, which
> *raises AI* and pushes attention from memory-bound toward compute-bound. It's the
> single most important example of the next section's ideas.

---

## 3. Optimizing Memory-Bound Kernels — three ways to raise AI

If §2 told you you're memory-bound (below the ridge), you have one goal: **do more
math per byte** (raise AI) or **move fewer bytes**. Three tools:

### Fusion — stop making round-trips
> **Fusion** = combine several operations that run back-to-back into **one kernel**,
> so intermediate results stay on-chip instead of being written to the warehouse and
> immediately read back.

Example: `y = gelu(x + b)`. Done as three separate kernels, you'd write `x+b` to the
warehouse, read it back, write `gelu(...)`, read it back — the intermediate makes two
pointless round trips. Fused into one kernel, `x+b` lives in the workers' hands and
flows straight into the GELU; you read `x` once and write `y` once. Same math, a
fraction of the bytes → higher AI. (Flash Attention is fusion taken to the extreme.)

### Blocking / tiling — reuse each loaded byte more
> **Blocking** (a.k.a. **tiling**) = load a *block* of data onto the team's fast
> table once, then have the whole team reuse it many times before fetching the next.

This is the tile-size lever from §2: a bigger tile on the table means each byte you
hauled from the warehouse feeds more arithmetic before you evict it — directly raising
AI (recall 16×16 → 8 FLOP/byte vs 64×64 → 32 FLOP/byte). The limit is table space
(SMEM is small), which is why tile sizes are a central tuning knob.

Three things about tiling that trip people up:

**It moves the *same* numbers *fewer times*, not more numbers.** Each input element is
*needed* many times by the math no matter what — tiling only changes whether each reuse
costs a warehouse trip or a cheap table read. Trace one element `A[0][0]` in a 4×4
matmul: it's used by every output in row 0 (`C[0][0..3]`). *Without* a table you compute
outputs one at a time and re-fetch `A[0][0]` from the warehouse **4 times**; *with* the
row staged on the table you fetch it **once** and the other 3 uses read it locally. Over
an N×N matmul each element is loaded ~`N` times naively vs ~`N/B` times with a tile of
width `B` — so **total warehouse traffic drops** (~`2N³` → ~`2N³/B`). *This is why
tiling is the right move precisely when you're memory-bound: it cuts total traffic.*

**So bigger tiles help *even though* each load is bigger.** A larger fetch isn't waste:
outputs are *pairs* (rows × columns → math grows like tile **area**) while inputs are
*two lists* (rows + columns → bytes grow like tile **perimeter**). Area outruns
perimeter, so FLOP/byte rises:

| Tile (fp16, one K-slice) | Bytes loaded | Math done | AI |
|---|---|---|---|
| 16×16 | 16+16 = 64 B | 16×16 = 512 FLOP | **8** |
| 64×64 | 64+64 = 256 B | 64×64 = 8192 FLOP | **32** |

4× the tile side → 4× the bytes but **16×** the math → 4× the AI.

**Where the reused numbers live:** the shared tile goes in **shared memory (SMEM)** —
reuse *across the team* needs shared storage, and a register is private to one thread
(invisible to the others). Registers hold only the small per-thread working set a thread
is multiplying right now (copied in from SMEM) plus the accumulator. So the path is
**GMEM → SMEM (team tile, big reuse) → registers (tiny working set + running total)** —
the same "load once, reuse many" trick applied at each level.

### Low precision — make each number smaller
> Use **fewer bytes per number**: fp32 (4 bytes) → fp16/bf16 (2) → fp8 (1) → fp4
> (½). Half the bytes per number = half the traffic = double the AI for the same math.

This is why ML races toward fp8/fp4: for a memory-bound kernel, shrinking the dtype is
often the biggest single win, because the bottleneck *is* bytes. (The cost is
precision — see the numerics discussion in the Background companion.)

### One more layer: move the bytes you *do* move *efficiently*
Even at a fixed byte count, *how* you touch the warehouse matters:

> **Coalescing:** when the 32 threads of a warp read 32 *adjacent* addresses, the
> hardware fuses them into one wide transaction — full road utilization. If they read
> *scattered* addresses, it takes many transactions for the same bytes — you waste
> most of the road. Memory-bound kernels must access memory in contiguous, aligned
> patterns (**vectorized/coalesced** access).
>
> **Bank conflicts:** the team's table (SMEM) is split into parallel "banks"; if many
> threads hit the same bank at once they serialize. Laying data out to avoid conflicts
> keeps the table fast.

These don't change AI, but a memory-bound kernel that ignores them never reaches even
its road ceiling.

---

## 4. The Optimization Ladder — the staircase for compute-bound GEMM

Once a kernel is compute-bound (a big GEMM), the game changes: you're no longer
short on bytes, you're trying to keep the matrix machines **100% busy**. The chapter
shows this as a **ladder** — a sequence of techniques, each closing more of the gap
to the machine ceiling. Roughly, the GEMM "optimization journey" on B200 climbs:

1. **Synchronous tiled baseline** — the straightforward version: load a tile, compute
   it, store, repeat. Correct but the machines idle during every load.
2. **TMA** — hand the loading to the background delivery engine (Background §4) so
   workers stop babysitting copies. *The chapter notes this is the first big jump.*
3. **Warp specialization** — split the team so some warps *only* load (producers) and
   others *only* compute (consumers), handing off through the table. (Background's
   producer/consumer pattern.)
4. **CTA clusters** — let neighboring teams share loaded tiles via distributed shared
   memory (DSMEM), so a tile fetched once serves several teams.
5. **Multi-consumer execution** — several consumer warpgroups on one tile, splitting
   the output so more matrix machines run at once.

The point isn't to memorize the rungs — it's the *shape*: each step removes another
source of machine idleness, walking performance up toward the roofline ceiling.

---

## 5. Reducing Idle Time Through Overlap

This is the engine under the whole ladder, and it's the same idea as Background §4's
assembly line, now stated as a performance principle:

> A machine that is **waiting** produces nothing. The way to go fast once you're
> compute-bound is to **overlap independent stages** so that while one stage waits,
> another does useful work — the matrix machine computes tile N *while* the delivery
> engine fetches tile N+1 *while* the workers write tile N−1.

The contrast the chapter draws: **sequential** scheduling (do stage 1, then 2, then
3, repeat) leaves the expensive machine idle during stages it isn't involved in.
**Orchestrated concurrency** (pipelining) keeps every unit busy every cycle. Same
work, far less idle time — and idle time is the entire gap between your kernel and the
roofline ceiling. "Fast" mostly *means* "nothing valuable is ever waiting."

---

## 6. Occupancy and Resource Pressure — and a modern surprise

> **Occupancy** = how many warps (worker-squads) are resident on a workstation at
> once, relative to the maximum it can hold. Recall from Background §0 that the GPU
> hides a stall in one warp by instantly running a different ready warp — so
> *historically*, **high occupancy = more warps to switch among = better latency
> hiding.**

Occupancy is limited by **resource pressure**: each warp consumes registers (numbers
in workers' hands) and each team consumes table space (SMEM). The more each warp/team
hogs, the fewer fit on a workstation → lower occupancy. (This is exactly the register-
pressure story that motivated TMEM in Background §2.)

The surprise the chapter flags: **modern high-performance kernels often deliberately
run at *low* occupancy.** Why? Because they don't hide latency by swapping among many
warps anymore — they hide it with the **explicit deep pipeline** (§5). A few warps
each running a deep, carefully-overlapped pipeline can keep the matrix machines
saturated better than many shallow warps. So they spend the registers/table on
**pipeline depth** (more tiles in flight) instead of on **more resident warps**. The
old rule "maximize occupancy" is replaced by "**keep the critical machines busy**,"
and low occupancy is fine as long as they never idle.

---

## 7. Using Roofline to Guide Optimization — the three-step recipe

The chapter closes with the practical loop that ties it all together. Whenever a
kernel is slow, don't guess — do this:

1. **Estimate arithmetic intensity.** Roughly, how many FLOPs does the kernel do, and
   how many bytes does it move from the warehouse? Divide → AI (FLOP/byte).
2. **Compare to the ridge point** (250 on B200). Below it → **memory-bound**; above it
   → **compute-bound**. Now you know *which* problem you have.
3. **Measure the gap to the roof.** How close is your measured FLOP/s (or GB/s) to the
   relevant ceiling? The gap is your headroom.

Then aim your effort at the actual bottleneck:

- **Memory-bound** → §3 tools: fuse, tile bigger, drop precision, fix coalescing.
  (Buying a faster matrix machine would do *nothing*.)
- **Compute-bound** → §4–§6: climb the ladder, deepen the overlap, keep the machines
  saturated. (Moving data faster would do *nothing*.)

**The entire chapter in one sentence:** every kernel is capped by either data
movement or compute, whichever runs out first; *arithmetic intensity* (FLOP/byte)
versus the *ridge point* tells you which; then you either raise intensity (fusion,
tiling, low precision) for memory-bound kernels or eliminate machine idle-time
(pipelining, warp specialization, clusters) for compute-bound ones — and you use the
roofline to make sure you're fixing the limit you actually have.

---

## Glossary (quick lookup)

| Term | Plain meaning |
|------|---------------|
| **FLOP** | one floating-point op (add or multiply); an FMA counts as 2 |
| **FLOP/s** | how many FLOPs the hardware does per second (compute throughput) |
| **bandwidth** | bytes/second the road between warehouse and chip can carry (~8 TB/s on B200) |
| **latency** | the *delay* of a single memory trip (road length), vs bandwidth (road width) |
| **roofline model** | a chart of your two ceilings — compute (flat) and bandwidth (sloped) |
| **arithmetic intensity (AI)** | math per byte moved = FLOP/byte; the x-axis of the roofline |
| **ridge point** | AI where the two ceilings meet = peak compute ÷ bandwidth (~250 on B200) |
| **memory-bound** | AI below the ridge; limited by data movement; machines idle |
| **compute-bound** | AI above the ridge; limited by math; road keeps up |
| **elementwise / reduction** | low-AI ops (one op per element); always memory-bound |
| **GEMM AI ≈ N/3** | square fp16 matmul intensity grows with size → compute-bound when big |
| **Flash Attention** | keeps the scores matrix on-chip instead of writing it out; raises AI |
| **fusion** | merge back-to-back ops into one kernel to skip warehouse round-trips |
| **blocking / tiling** | stage a block on the table and reuse it many times; raises AI |
| **low precision** | fewer bytes per number (fp16/fp8/fp4) → less traffic → higher AI |
| **coalescing** | a warp reading adjacent addresses fuses into one wide transaction |
| **bank conflict** | multiple threads hitting the same SMEM bank; serializes, slows the table |
| **optimization ladder** | tiled → TMA → warp-specialization → clusters → multi-consumer |
| **overlap / pipelining** | run independent stages concurrently so nothing idles |
| **occupancy** | how many warps are resident per workstation vs the max |
| **resource pressure** | registers/SMEM each warp uses, which caps occupancy |
| **L2 cache** | a mid-level on-chip cache between the warehouse and the workstations |

---

## How to read the source chapter now

With this open beside it, the chapter should decode as:

1. **Roofline** → your two ceilings (compute flat, bandwidth sloped) and the ridge
   point that splits memory-bound from compute-bound.
2. **Arithmetic Intensity** → FLOP/byte for elementwise (low), GEMM (grows as N/3),
   and attention (Flash Attention raises it) — locating each on the roofline.
3. **Memory-bound optimization** → fuse, tile bigger, drop precision (+ coalesce).
4. **Optimization ladder / overlap / occupancy** → for compute-bound kernels, keep the
   matrix machines busy by pipelining, specializing warps, and clustering — even at
   low occupancy.
5. **Roofline to guide** → estimate AI → compare to ridge → measure gap → fix the real
   bottleneck.

And the one habit to keep from the whole chapter: **measure which wall you're hitting
before you start climbing** — the fixes for "waiting on data" and "waiting on math"
are opposite, and roofline is how you tell them apart.
