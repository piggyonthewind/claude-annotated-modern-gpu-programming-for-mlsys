# Annotated: *Modern GPU Programming for MLSys* — Background

A **from-zero annotated companion** to the *Background* chapter of
[**Modern GPU Programming for MLSys**](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html)
by the MLC community.

The original chapter is excellent but assumes a lot of background — it uses terms
like GEMM, warp, CTA, SMEM, TMA, and MMA without stopping to define them. This
companion re-explains every concept **from scratch, for a software engineer who has
never studied GPUs or hardware.** It assumes only that you can read a `for`-loop.
Every idea is anchored to either (a) a plain loop you could have written, or (b) a
running picture of *workers in a factory* (workers, teams, a shared table, a faraway
warehouse) — no CPU, cache, or assembly knowledge required.

## Read it

- 📖 **Web version (nicely formatted):** enable GitHub Pages (see below), then visit
  the published site — a single-page reader with a chapter switcher and per-page
  table of contents.
- 📝 **Plain markdown (the source of truth):**
  - [`background.md`](background.md) — Chapter 1: Background
  - [`performance.md`](performance.md) — Chapter 2: Performance
  - [`data_layout.md`](data_layout.md) — Chapter 3: Data Layout
  - [`layout_generations.md`](layout_generations.md) — Chapter 4: The Evolution of Tensor Core Data Layouts

## What it covers

### Chapter 1 — Background *(the execution model)*
Following the source chapter's four sections:

1. **The Execution Hierarchy** — thread → warp → warpgroup → CTA → cluster → grid, and the idea of *scope*
2. **Memory Spaces** — registers, shared memory, TMEM, global memory (HBM), DSMEM, as a "distance = time" hierarchy
3. **Compute: CUDA Cores and Tensor Cores** — what actually does the math, and the `wgmma` → `tcgen05` evolution
4. **The GEMM Data Pipeline** — load / compute / epilogue as an overlapped assembly line that keeps the Tensor Cores fed

Plus clarifications that came out of real questions: *lockstep exists only inside a
warp*, *warp vs. CTA vs. warpgroup*, *who does what* (TMA moves, Tensor Core
multiplies, threads conduct), and the numerics of Tensor-Core vs. CUDA-core matmul.

### Chapter 2 — Performance *(what makes a kernel fast)*
1. **The Roofline Model** — your two ceilings (compute vs. bandwidth) and the ridge point
2. **Arithmetic Intensity** — FLOP/byte for elementwise, GEMM (`≈ N/3`), and attention
3. **Optimizing Memory-Bound Kernels** — fusion, tiling, low precision, coalescing
4. **The Optimization Ladder** — tiled → TMA → warp specialization → clusters → multi-consumer
5. **Overlap, Occupancy & Resource Pressure** — keeping the matrix machines saturated
6. **Using Roofline to Guide Optimization** — a three-step "which wall am I hitting?" recipe

### Chapter 3 — Data Layout *(where each number physically lives)*
1. **The Shape-Stride Model** — layout as a function from `(i,j)` to an address; free transpose/reshape (like NumPy strides)
2. **Tile Layout** — nesting shape-stride to describe blocked/tiled matrices
3. **Named Axes** — TMEM `(@TLane,@TCol)` and register fragments `(@laneid,@reg)`, where a "location" is a tuple, not one address
4. **Replication & Offset** — broadcasting the same data into many copies, and multi-GPU mesh sharding
5. **Swizzle Layout** — XOR address permutation to kill shared-memory bank conflicts

### Chapter 4 — The Evolution of Tensor Core Data Layouts *(Ampere → Hopper → Blackwell)*
1. **Two memory requirements** — coalesced GMEM reads and bank-conflict-free SMEM, which every layout must respect
2. **Ampere** — operands *and* accumulator as register fragments; `ldmatrix` staging; hand-written XOR swizzle
3. **Hopper** — `wgmma` reads operands straight from SMEM via a 64-bit matrix descriptor (hardware-managed swizzle)
4. **Blackwell** — `tcgen05.mma` moves the accumulator into TMEM; block-scaled MMA and scale-factor staging
5. **The invariant** — each instruction's output layout must equal the next's input layout, or you get silent wrong answers

## How the site is built

There is no build step. [`index.html`](index.html) fetches [`background.md`](background.md)
and renders it in the browser (via [marked](https://marked.js.org/) +
[highlight.js](https://highlightjs.org/) from a CDN), auto-generating the sidebar
table of contents. To edit the content, just edit `background.md` — the page updates
itself. Because it fetches over HTTP, preview it through a server, not a `file://`
path:

```bash
# from the repo root
python -m http.server 8000    # then open http://localhost:8000
```

## Enable GitHub Pages

Repo **Settings → Pages → Build and deployment → Source: "Deploy from a branch" →
Branch: `main` / `/ (root)` → Save.** After a minute the site is live at
`https://piggyonthewind.github.io/claude-annotated-modern-gpu-programming-for-mlsys/`.

## Attribution & scope

This is an **independent, unofficial** study companion. It is **not** affiliated with
or endorsed by the authors of *Modern GPU Programming for MLSys*, and it does **not**
reproduce the book's text — it is original explanatory writing that points back to the
source chapter, which you should read directly. All credit for the underlying material
goes to the [original authors](https://mlc.ai/modern-gpu-programming-for-mlsys/).

Written collaboratively with Claude (Anthropic).
