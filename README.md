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
  the published site.
- 📝 **Plain markdown:** [`background.md`](background.md) — the single source of truth.

## What it covers

Following the source chapter's four sections:

1. **The Execution Hierarchy** — thread → warp → warpgroup → CTA → cluster → grid, and the idea of *scope*
2. **Memory Spaces** — registers, shared memory, TMEM, global memory (HBM), DSMEM, as a "distance = time" hierarchy
3. **Compute: CUDA Cores and Tensor Cores** — what actually does the math, and the `wgmma` → `tcgen05` evolution
4. **The GEMM Data Pipeline** — load / compute / epilogue as an overlapped assembly line that keeps the Tensor Cores fed

Plus extended clarifications that came out of real questions: *lockstep exists only
inside a warp*, *warp vs. CTA vs. warpgroup* (execution unit vs. cooperation unit vs.
fixed machine-interface), and *who does what* — TMA moves, Tensor Core multiplies,
threads conduct.

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
