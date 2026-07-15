# Annotated Read: "Background — Modern GPU Programming for MLSys"

A companion to <https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html>

**Audience:** someone who writes ordinary software (loops, functions, arrays) but
has **never** studied GPUs, hardware, or how computers work underneath. Nothing
here assumes you know what a CPU, a cache, or "assembly" is. Every new word is
defined the first time it appears, using either (a) a plain `for`-loop you could
have written, or (b) a picture of **workers in a big building**.

We follow the chapter's four sections in order:

1. The Execution Hierarchy — *who* does the work
2. Memory Spaces — *where* the numbers are kept
3. Compute: CUDA Cores and Tensor Cores — *what* actually does the math
4. The GEMM Data Pipeline — how 1–3 combine to multiply two matrices fast

---

## 0. The one idea everything rests on

Start with a loop you already know how to write. You have a big array of numbers
and you want to do the same thing to every element:

```python
# ordinary program: ONE worker walks the array, one element at a time
for i in range(1_000_000):
    out[i] = f(data[i])
```

This runs each `f(data[i])` one after another. A million elements means a million
steps taken in sequence.

**A GPU is a chip built to run that loop a completely different way: it does all
one million iterations **at the same time**.** Instead of one worker walking the
array, it hires a million tiny workers and says "each of you, take one `i`, compute
`out[i] = f(data[i])`, all at once."
 
Two facts make this work, and they define everything about GPUs:

- **Each tiny worker is weak.** On its own, one GPU worker is *slower* than you'd
  expect. The GPU makes no attempt to be fast at any single task.
- **There are thousands of them, and they're disposable.** You create tens of
  thousands of workers for one job and throw them all away when it's done.

So a GPU is not "a fast computer." It is **a very wide computer**: bad at doing one
thing quickly, spectacular at doing the same simple thing to millions of pieces of
data simultaneously. This only pays off if your problem has lots of independent,
identical work. Matrix math (the whole point of this chapter) has millions of
identical little multiplications — a perfect fit.

Now the single most important vocabulary word:

> **Thread** = one worker running one copy of your instructions on one piece of
> data. In the loop above, one thread is one execution of the loop body for one
> value of `i`. That's it. When the chapter says a GPU runs "thousands of threads,"
> read it as "thousands of copies of the loop body running side by side, each on
> its own element." Every other term in this document is about **how those threads
> are organized, where they keep their numbers, and what hardware they run on.**

(Note: "thread" here does **not** require any prior meaning. If you've heard the
word before in other contexts, just use the definition in the box — one worker
running one copy of the instructions.)

The specific job the chapter cares about is **GEMM** — multiplying two matrices.
We'll define it precisely in §4, but keep in mind: this whole chapter is really
"how do we organize a few hundred thousand threads to multiply two matrices as
fast as physically possible."

---

## 1. The Execution Hierarchy — *who* does the work

If you have hundreds of thousands of threads (workers), you can't treat them as one
undifferentiated mob. The GPU organizes them into **nested teams**, where each
bigger team can cooperate in more powerful (but more expensive) ways. The chapter's
ladder, smallest to largest:

> **thread → warp → warpgroup → CTA → cluster → grid**

Think of it as an org chart: individual → small squad → bigger squad → team →
neighborhood of teams → the entire hired workforce. Here is each rung.

### thread — one worker
Already defined above: one worker running one copy of the instructions on one
element. Has a little bit of private scratch space that only it can see (its
numbers-in-hand; more in §2). This is the atom everything is built from.

### warp = 32 threads locked together
**A warp is a squad of exactly 32 threads that are forced to perform the same step
at the same instant.** The number **32** is fixed by the hardware and shows up
everywhere, so it's worth memorizing.

Picture a **rowing crew of 32**: the coxswain calls one stroke, and all 32 rowers
pull their oar on that beat. They cannot each do their own thing at their own pace
— every worker in the warp executes instruction #1 together, then instruction #2
together, and so on.

- The official name for this style is **SIMT — "Single Instruction, Multiple
  Threads."** Decoded: *one* instruction is handed to *many* (32) threads at once.
  Each thread still has its own data and its own numbers, but they all march in
  step.
- **The cost of this, called "divergence":** suppose your instructions contain
  `if condition: A else: B`, and inside one warp 20 threads need branch `A` and 12
  need branch `B`. Because all 32 must move together, the crew does `A` first (the
  12 `B`-threads sit idle and do nothing), then does `B` (the 20 `A`-threads sit
  idle). You paid for 32 workers but only some did useful work at each step. **This
  is why GPU code avoids per-element `if/else` branching** — branchy logic makes
  workers idle. Straight-line "do the same math to everyone" code keeps all 32
  busy.

### warpgroup = 4 warps = 128 threads
Just a bigger squad: **4 warps standing next to each other = 128 threads.** It
exists because the most powerful matrix machines (§3) are operated by a whole
128-worker squad at once rather than by a single 32-worker crew. For now: warpgroup
= 128 workers acting as one unit.

### Important: lockstep exists ONLY inside a warp — not across warps

A natural question: threads inside one warp move in lockstep (same instruction, same
instant). Do the **4 warps inside a warpgroup** also move in lockstep with each
other? **No.** This is the single most common misunderstanding, so it's worth
stating flatly:

> **The 32 threads inside one warp are the *only* thing that runs in automatic
> lockstep. Every grouping above the warp — warpgroup, CTA, cluster, grid — is a set
> of units that run *independently* and only line up when *explicitly* told to.**

Concretely, the 4 warps of a warpgroup each run at their own pace. At any moment:

- warp 0 might be on instruction #100,
- warp 1 on instruction #40,
- warp 2 paused, waiting for a number to arrive from the warehouse (GMEM),
- warp 3 on instruction #75.

So a "warpgroup" is a *logical grouping*, not a lockstep group. The 4 warps drift
along independently and only have to **stop and synchronize at specific moments** —
namely when they jointly issue a warpgroup-scoped instruction like `wgmma` (the
128-worker matrix multiply, §3). At that point there is an explicit "are all 4 of you
here and ready?" checkpoint; between such points, they're four independent crews.

Back to the rowing analogy:

> A **warp** = one crew of 32, locked to the coxswain's beat.
> A **warpgroup** = **4 separate crews**, each rowing at its own pace — *except* every
> so often they must line up together to jointly lift one heavy object (the warpgroup
> matrix multiply). They coordinate **only** at that lift, via an explicit "everyone
> ready?" signal. The rest of the time, four independent crews.

The rule generalizes all the way up the hierarchy:

| Level | How its members run relative to each other |
|-------|--------------------------------------------|
| threads in a **warp** (32) | **lockstep** — same instruction, same instant (SIMT) |
| warps in a **warpgroup** (4) | **independent**; coordinate only at explicit sync points |
| warps/warpgroups in a **CTA** | **independent**; coordinate via barriers |
| CTAs in a **cluster / grid** | **independent**; coordinate via barriers / memory |

There's a reason it *must* be this way. Recall from §0 that the GPU hides slowness by
switching to a different ready warp whenever one stalls (like warp 2 above, waiting on
the warehouse). That trick only works *because* warps run independently — if the 4
warps in a warpgroup were forced into lockstep, a stall in any one of them would
freeze all four, and the GPU would lose its main way of staying busy.

### CTA — a team that shares a table and can wait for each other
**CTA stands for "Cooperative Thread Array."** It is a group of threads (often
128–1024 of them) that get to do two things ordinary strangers-threads can't:

1. **Share a small common table.** Every thread in the CTA can read and write one
   small shared workspace (called *shared memory*, §2). This is how the team passes
   numbers to each other cheaply.
2. **Wait for each other at a checkpoint.** The team can hit a line that says
   "everyone stop here until all of us arrive, then continue together." That
   checkpoint is called a **barrier**. It lets them coordinate: e.g. "all of you
   finish loading your piece before any of you start using the shared table."

Threads in *different* CTAs are basically strangers — they can't share a table or
wait for each other; they can only communicate through the slow faraway warehouse
(§2). So **the CTA is the boundary of cheap teamwork.**

> **Naming warning:** you will also see the words **"thread block"** (or just
> **"block"**) for this exact same thing. "CTA" and "block" are two names for one
> concept. The chapter uses **CTA**; other GPU material says "block." Same idea.

To place a CTA physically, one more word:

> **SM = "Streaming Multiprocessor."** Picture the GPU as a large factory floor
> with about **100+ identical workstations**. One SM is one workstation. A CTA (one
> team) is assigned to run at one workstation. The workstation is what supplies the
> team's shared table, their tools, and the matrix machines. So: **factory = GPU,
> workstation = SM, team = CTA, squad = warp, worker = thread.**

### Why three levels? warp vs. warpgroup vs. CTA

These are all "groups of threads," so why have three? Because each answers a different
need — and two of them are *fixed by hardware* while the third is *yours to choose*:

| Level | Size | Why it exists |
|-------|------|---------------|
| **warp** | fixed 32 | the hardware's lockstep execution bundle — you don't pick it |
| **warpgroup** | fixed 128 | exactly the hands needed to hold and drive one matrix-multiply (`wgmma`) |
| **CTA** | **flexible** (up to ~1024) | *your* team; you choose how many warps/warpgroups it holds |

Two distinctions do all the work:

- **warp vs. CTA — execution unit vs. cooperation unit.** A warp is *how threads
  execute* (32 in lockstep). A CTA is *how threads cooperate*: only a CTA owns a
  **shared table (SMEM)** and a **barrier**, and only threads in the same CTA can pass
  data to each other cheaply. A lone warp has no shared table and can't sync with other
  warps — so it can't stage a tile for reuse. That's why tiling needs a CTA, not just a
  warp.
- **warpgroup vs. CTA — a fixed machine-interface vs. a flexible team.** The matrix
  machine physically needs exactly 128 threads to feed it (32 is too few to hold a
  tile; a variable-size CTA is the wrong shape to build a hardware instruction around).
  So the warpgroup is the *fixed 128-thread unit that drives one MMA* — it owns no
  table; it's an *operation scope*, not a home for data. A CTA then **composes** several
  warpgroups: either doing the **same** job and splitting the data (2 consumer
  warpgroups each computing half the output tile — because 128 is a hard cap), or
  **different** jobs (1 producer warpgroup loading + 1 consumer warpgroup computing —
  the warp specialization from §4).

So: **warp = fixed execution bundle, warpgroup = fixed matrix-machine interface, CTA =
the flexible team you compose out of them.**

### cluster — a few neighboring teams who can reach each other's tables
Normally each team only sees its own table. A **cluster** is a small group of CTAs
placed at *neighboring* workstations, close enough that a worker on one team can
**reach over and use a nearby team's table directly** (this is *distributed shared
memory*, §2) instead of walking all the way to the warehouse. It's a recent feature
that lets a handful of teams cooperate as a slightly larger group.

### grid — the entire workforce for one job
When you start a GPU program, you launch a **kernel** (the word for "a function
that runs on the GPU"). You tell it how many teams of how many workers you want —
e.g. "4096 teams of 256 workers." That whole population is the **grid**. It's every
thread involved in this one launch.

### The real lesson of §1: *scope* — "who performs this step?"

The chapter's actual takeaway is a habit of mind. In GPU code, every operation is
performed *by some level of this ladder*, and that level is called its **scope**:

- Some steps are done by **one thread** on its own (e.g. kicking off a big memory
  transfer with **TMA**, §4).
- Some are done by a **warp** (32 workers) together.
- Some are done by a **warpgroup** (128 workers) together — the big matrix machine.
- Some are done by **two whole teams (2 CTAs)** cooperating at once.

So whenever you look at a line of a GPU kernel, you ask three questions together:
**(a) who does this — how many workers/what scope?  (b) where are the numbers it
touches (§2)?  (c) how is it triggered?** The chapter says learning to read those
three axes at once *is* learning to read GPU code. Hold that as you go into §2–§4.

---

## 2. Memory Spaces — *where* the numbers are kept

Here's the second big surprise for someone from ordinary programming. In normal
code, you write `x = data[i]` and never think about *where* `data` physically is —
it just works. On a GPU, **you must decide, by hand, which of several storage places
each number lives in**, and that decision is the main thing that makes code fast or
slow.

The mental model is **distance = time**. Imagine our factory again:

- Numbers in a worker's **hands** are instant to use.
- Numbers on the **team's table** are fast to grab.
- Numbers in the **warehouse across the factory** take a long walk to fetch.

Every storage tier below is just "how far from the worker, and therefore how slow."
Fastest/smallest first.

### Registers — the numbers in a worker's hands (per thread)
> The chapter's word is **RF, "Register File."**

Each worker holds a tiny handful of numbers that only it can see — the variables
it's actively computing with. Instant to use, but there are very few of them. If a
worker tries to juggle too many numbers at once, there's no room, and the
workstation can host fewer workers overall. (The chapter calls that squeeze
"register pressure" — too many numbers in hand per worker.)

### Shared memory (SMEM) — the team's small shared table (per CTA)
> **SMEM = "Shared Memory."**

A small workspace that **every worker in one team (CTA) can read and write.** It is
fast to reach and deliberately small — the chapter cites **up to 228 KB per
workstation (SM) on a B200** GPU. The whole point of it: the team walks to the
warehouse *once*, carries a batch of numbers back to this shared table, and then all
the workers reuse those numbers from the table many times without walking to the
warehouse again. Managing this table by hand is the core of fast GPU code.

### TMEM (Tensor Memory) — a special results-board (per CTA), new on Blackwell
> **TMEM = "Tensor Memory,"** introduced on the Blackwell generation of GPUs.

A separate board where the big matrix machine (§3) writes its **running totals** as
it multiplies. Its size, per the chapter: **128 rows × up to 512 columns of 32-bit
numbers**, exposed as four sections, one per warp in a warpgroup.

Why it exists: on older GPUs those running totals were held *in the workers' hands*
(registers), and a big matrix multiply needed so many that a worker's hands were
completely full, leaving no room and forcing the workstation to host very few
workers. Blackwell added this dedicated results-board so the totals live *off* the
workers' hands, freeing them up. If you remember one thing: **TMEM = the board where
matrix-multiply totals accumulate, kept separate from the workers' hands.**

**Why a *separate* memory, instead of the usual registers or the table?** The
accumulator is big (a whole tile of running totals), long-lived (updated on *every*
step of the K-loop), and read-modify-written constantly by the matrix machine.
Registers are tiny and *private per thread*, so holding a shared tile-sized accumulator
there fills every worker's hands and starves the workstation of resident workers
(wrecking the latency-hiding from §0). The shared table (SMEM) is already busy holding
the *input* tiles and isn't wired for the machine's every-step access pattern. So
Blackwell gives the accumulator its own board — big enough, wired straight to the
Tensor Core, and off both the workers' hands and the table.

### Global memory (GMEM) — the big warehouse (whole device)
> **GMEM = "Global Memory,"** physically made of **HBM = "High Bandwidth Memory."**

The huge storage — **tens of gigabytes** — that *every* worker in *every* team can
reach. It holds all your data (the full matrices, the model weights). It's the
"source of truth." But it's **far away**: a single fetch from here takes a long time
compared to the table or the hands. HBM ("high bandwidth") means you *can* move a
lot of data per second in bulk, but any individual trip still has a long delay.

**The single rule that drives the entire chapter:** trips to the warehouse are
slow, so *minimize them* — load from the warehouse once, carry the batch to your
team's table (SMEM), reuse it as much as possible on-table, and write the answer
back to the warehouse once at the end.

### DSMEM — reaching a neighbor team's table instead of the warehouse (per cluster)
> **DSMEM = "Distributed Shared Memory."** This is the chapter's named sub-topic.

Normally a team can only use its own table. DSMEM lets a team **reach over and read
a neighboring team's table directly** — but only for teams in the same *cluster*
(§1), i.e. at neighboring workstations wired together. Why it helps: if two teams
both need the same batch of numbers, without DSMEM they'd each make the slow trek to
the warehouse. With DSMEM, one team fetches it to its table, and the neighbor just
reaches over — **no second warehouse trip.**

### The whole memory picture in one table

| Storage | Who can use it | Speed | Size (rough) | What's kept here |
|---------|----------------|-------|--------------|------------------|
| Registers (RF) | one worker (thread) | instant | a few numbers | the numbers being computed right now |
| Shared memory (SMEM) | one team (CTA) | fast | ≤228 KB / workstation | a batch carried from the warehouse for reuse |
| TMEM | one team (CTA) | fast | 128×512 numbers | matrix-multiply running totals (Blackwell) |
| DSMEM | one cluster of teams | fast | = the neighbors' tables | shared batch, borrowed from a nearby team |
| Global memory (GMEM/HBM) | everyone | slow, far | tens of GB | all the real data; the source of truth |

The design goal behind everything: **keep each number as close to the worker as you
can, and touch the faraway warehouse as few times as possible.**

---

## 3. Compute: CUDA Cores and Tensor Cores — *what* does the math

Each workstation (SM) has two very different kinds of math equipment. The chapter's
point is that one is enormously more powerful than the other, and that shapes how
you write everything.

### CUDA Cores — the workers' ordinary hands
> The chapter mentions **ALU = "Arithmetic Logic Unit."**

A **CUDA core** is the basic ability to do *one* arithmetic step: one addition or
one multiplication, for one worker. It's the everyday tool the workers use for
normal math — computing indexes, running activation functions, adding things up.
Nothing special; just "a worker doing one `+` or `×` at a time." Speed of math is
measured in **FLOP/s** = "floating-point operations per second" = how many little
`+`/`×` steps get done per second (one such step is a "FLOP").

### Tensor Cores — the special matrix machine
A **Tensor Core** is a dedicated machine at the workstation that multiplies **two
whole small grids of numbers in a single action**, instead of one number at a time.

> **MMA = "Matrix Multiply-Accumulate."** This is the action the machine performs:
> take a small grid `A`, a small grid `B`, multiply them, and **add** the result
> onto a running total `C` — all as *one* operation. Where a worker's hands do one
> `a×b` and then one `+c`, this machine does an entire tile's worth of
> multiply-and-add in a single shot. That's where the chapter's **"10×+ more
> throughput"** comes from.

Because the matrix machine is so much faster than the workers' hands, **the whole
kernel is organized to keep that machine busy.** The workers (CUDA cores) mostly
exist to fetch and arrange numbers so the machine never sits idle.

The complication the chapter warns about: **the matrix machine's controls change
with every new GPU generation**, and to use the newest ones you sometimes have to
write very low-level instructions by hand (see the PTX note below). The history it
references:

- **Older GPUs:** the machine was operated per-**warp** (32 workers), and both its
  inputs and its running totals sat in the workers' hands (registers).
- **Hopper generation → `wgmma`** (read: "warpgroup MMA"): now a whole **warpgroup**
  (128 workers) operates the machine together, on bigger grids, and it can read its
  inputs straight from the team's **table (SMEM)**. It's also **asynchronous** — you
  start the multiply and immediately go do other work while it runs. (That "start it
  and walk away" property is the key that makes the §4 assembly line possible.)
- **Blackwell generation → `tcgen05`** (the 5th-generation machine): the running
  totals now live on the separate **results-board (TMEM)** instead of in the
  workers' hands, and **two teams (2 CTAs)** can jointly operate one bigger multiply,
  sharing inputs.

> **How you actually give these instructions — two words the chapter drops:**
> **CUDA C++** is the ordinary language you write GPU programs in (it looks like
> regular C++ code with some GPU extras). Your CUDA C++ gets translated into
> **PTX**, a much lower-level list of exact machine instructions. Normally you never
> see the PTX. But when a brand-new machine control like `wgmma` or `tcgen05` is so
> new that the ordinary language can't express it yet, expert kernel authors write
> the low-level PTX instruction by hand. This chapter is written at that expert
> level, which is why it keeps naming these raw instructions.

### CUDA core vs. Tensor core, head to head — and who does what

| | **CUDA core** | **Tensor core** |
|---|---|---|
| Does | one scalar op (`+` or `×`) | a whole tile **matrix multiply-accumulate** |
| Granularity | one number | a grid of numbers at once |
| Driven by | one thread | a warp / warpgroup, collectively |
| Flexibility | **general** — any math, control flow, activations | **only** dense matrix multiply |
| Throughput | baseline | **10×+** on matrix work |
| Inputs | general (fp32, int, …) | low precision in (fp16/bf16/fp8), accumulates in fp32 |

The way to hold it: a real kernel has **three kinds of worker**, and the threads are
the *conductor*, not the calculator —

> **TMA moves, the Tensor Core multiplies, and the threads conduct.** The delivery
> engine (TMA) hauls the data; the matrix machine (Tensor Core) does the heavy
> multiply; the threads (CUDA cores) mostly *issue* those two, coordinate the pipeline
> with barriers, and do the lighter epilogue math (scale, bias, activation). For a
> **non**-matrix kernel (softmax, normalization, elementwise) the Tensor Cores sit idle
> and the CUDA cores do 100% of the math.

---

## 4. The GEMM Data Pipeline — putting it all together

First, define the target. **GEMM = "GEneral Matrix-Matrix Multiply."** In loop form
(which is all it is):

```python
# Multiply matrix A (M×K) by matrix B (K×N) to get C (M×N):
for i in range(M):
    for j in range(N):
        total = 0
        for k in range(K):
            total += A[i][k] * B[k][j]   # <-- multiply-and-add, over and over
        C[i][j] = alpha * total + beta * C[i][j]
```

That inner "multiply two numbers and add to a running total" is *exactly* what the
Tensor Core's MMA does — but on whole grids at once instead of single numbers.
"General" just means it also does the `alpha * … + beta * C[i][j]` scaling at the
end. **This triple loop is the operation underneath every layer of a neural network**
(each linear layer, each attention score, each MLP is a GEMM), which is why making it
fast is the entire ballgame.

Because the matrices are far too big to fit on a team's table, you **chop them into
small square pieces called "tiles"** and process one pair of tiles at a time. A
high-performance GEMM is then a **three-stage assembly line** that moves tiles up
through the storage tiers of §2, through the matrix machine of §3, using the right
scopes of §1:

```
   WAREHOUSE ──①Load──▶ TEAM TABLE ──②Compute──▶ RESULTS BOARD ──③Finish──▶ WAREHOUSE
     (GMEM)   via TMA      (SMEM)     via MMA         (TMEM)                    (GMEM)
```

### Stage ① — Load: warehouse → team table
Fetch the next tiles of `A` and `B` from the faraway warehouse (GMEM) onto the
team's fast table (SMEM).

> **TMA = "Tensor Memory Accelerator."** A dedicated **delivery machine** that hauls
> a whole tile from warehouse to table when **a single worker** just says "go." It
> works out all the addressing itself, so the workers don't waste steps fetching by
> hand, and it runs **in the background** — the worker fires it off and immediately
> does other work.
>
> **TMA multicast:** one delivery can drop copies onto *several teams' tables* at
> once — used when two teams share the same tile, so you fetch it once instead of
> twice.
>
> (Careful: **TMA** = the *delivery machine*; **TMEM** from §2 = the *results
> board*. Two different things that unluckily both start with "Tensor Memory." The
> chapter even calls TMA "Tensor Memory *Architecture*" loosely — it's really the
> *Accelerator*, the copy engine.)

### Stage ② — Compute: team table → results board
The matrix machine (Tensor Core) reads the two tiles off the table (SMEM), multiplies
them, and adds the result onto the running total on the results board (TMEM). This
repeats across all the tiles along the `K` direction of the loop above — that
repeating middle part is called the **mainloop** ("the main loop that accumulates
tile after tile").

### Stage ③ — Finish (the "epilogue"): results board → warehouse
> **Epilogue** = everything you do to the finished total *after* the mainloop, before
> storing it. That's the `alpha * total + beta * C` scaling, plus optional extras
> done while the numbers are still conveniently on-chip: add a bias, apply an
> activation function (like ReLU or GELU), convert the number format, and finally
> write the answer tile back to the warehouse.
>
> Doing those extras here — "epilogue fusion" — means a linear-layer-plus-activation
> is *one* GPU program instead of two, because you finish the leftover work while the
> data is already close by, instead of shipping it to the warehouse and fetching it
> back.

### The actual lesson of §4: overlap the three stages

A naive version does ① then ② then ③ for one tile, then starts the next tile — so
the expensive matrix machine sits **idle** during every warehouse fetch. The
chapter's core point is to run the three stages **overlapping, like a real assembly
line:**

- While the matrix machine multiplies tile **N** (stage ②)…
- …the delivery machine is already fetching tile **N+1** (stage ①)…
- …and the workers are writing out the finished tile **N−1** (stage ③).

Nobody waits; the matrix machine never goes idle. This overlap is only possible
because the delivery (TMA) and the multiply (wgmma/tcgen05) both run **in the
background** — you start them and keep going (§3).

The price of overlapping is coordination. If stage ② grabs a tile off the table
before stage ① finished putting it there, it reads garbage. So the stages hand off
using the **barriers** from §1:

> A stage that produces data raises a flag saying "tile ready"; the stage that
> consumes it waits for that flag before touching the table. That's the checkpoint
> mechanism (barrier) making the handoff safe.

**The entire chapter in one sentence:** chop the matrices into tiles; use background
delivery machines and background matrix machines to stream those tiles
warehouse → table → results-board → warehouse; overlap the load / compute / finish
stages into an assembly line; and use checkpoints (barriers) so tiles are handed off
safely — all so the expensive matrix machines never sit idle.

---

## Glossary (quick lookup)

| Term | Plain meaning |
|------|---------------|
| **thread** | one worker running one copy of your instructions on one element |
| **warp** | a squad of 32 threads forced to do the same step at the same time |
| **warpgroup** | 4 warps = 128 threads acting as one unit |
| **CTA** (= "block") | a team of threads that share a table and can wait for each other |
| **cluster** | a few neighboring teams that can reach each other's tables |
| **grid** | the entire workforce launched for one job (one kernel) |
| **kernel** | a function that runs on the GPU |
| **SM** | one workstation on the factory floor; runs one team (CTA); ~100+ per GPU |
| **SIMT** | "one instruction, many threads" — the 32-in-lockstep style |
| **divergence** | wasted work when threads in a warp need different `if/else` branches |
| **scope** | which level (1 thread / warp / warpgroup / 2 teams) performs a step |
| **registers (RF)** | the few numbers a single worker holds in hand; instant |
| **shared memory (SMEM)** | the small fast table one team shares (≤228 KB/SM) |
| **TMEM** | the results-board where matrix totals accumulate (Blackwell) |
| **DSMEM** | reaching a neighbor team's table instead of the warehouse |
| **global memory (GMEM)** | the huge, slow, faraway warehouse; all real data |
| **HBM** | the physical stuff GMEM is made of: big, high bulk speed, long delay |
| **CUDA core / ALU** | a worker's ability to do one `+` or `×` at a time |
| **Tensor Core** | the special machine that multiplies whole tiles in one action |
| **MMA** | that machine's action: multiply two tiles, add to a running total |
| **FLOP/s** | how many little `+`/`×` steps happen per second (the speed metric) |
| **wgmma** | Hopper's matrix machine, run by a 128-worker squad, reads the table |
| **tcgen05** | Blackwell's matrix machine; totals on the board (TMEM); 2-team capable |
| **TMA** | background delivery machine: hauls a tile warehouse→table on one worker's "go" |
| **TMA multicast** | one delivery dropped onto several teams' tables at once |
| **GEMM** | "general matrix-matrix multiply" — the triple-loop `C = αA·B + βC` |
| **tile** | a small square piece of a big matrix, processed one at a time |
| **mainloop** | the repeating multiply-and-accumulate over tiles |
| **epilogue** | the finishing work after the mainloop (scale, bias, activation, store) |
| **barrier** | a checkpoint where a team waits so handoffs are safe |
| **CUDA C++** | the ordinary language you write GPU programs in |
| **PTX** | the low-level instruction list your program becomes; hand-written for new features |

---

## How to read the source chapter now

With this open beside it, the chapter's four sections should decode as:

1. **Execution Hierarchy** → the thread→warp→warpgroup→CTA→cluster→grid teams, and
   the habit of asking "who (what scope) performs this step?"
2. **Memory Spaces** → the "distance = time" table (hands → team table → results
   board → warehouse), and the rule of touching the warehouse as rarely as possible.
3. **Compute** → ordinary worker hands (CUDA cores) vs. the special matrix machine
   (Tensor Cores / MMA), and why its controls keep changing each GPU generation.
4. **GEMM Data Pipeline** → load / compute / finish as an overlapping assembly line,
   coordinated by checkpoints, that keeps the matrix machine constantly fed.

And behind all of it, the §0 idea: a GPU wins not by being fast at one thing, but by
running the same simple work across a huge number of workers — so the whole craft is
organizing those workers and moving numbers close enough that the matrix machines
never wait.

---

## Appendix: Numerics & determinism — why Tensor-Core and CUDA-core matmuls differ

A practical payoff of the CUDA-core vs. Tensor-core distinction (§3), and directly
relevant if you're chasing bit-reproducible results (e.g. logprobs in RL).

**Root cause: floating-point arithmetic isn't exact or associative.** In finite
precision `(a+b)+c ≠ a+(b+c)`. Summing `[1e8, 1, −1e8]` in ~fp32: left-to-right gives
`0` (the `1` rounds off the bottom of `1e8`), reordered gives `1`. Same numbers,
different order, different answer. A dot product is a big sum, so the **order** you add
the partial products changes the last bits.

Three ways a Tensor-Core matmul and a CUDA-core matmul diverge on the *same* inputs:

1. **Summation order.** A Tensor Core sums its partial products in a fixed
   hardware-defined tree; a CUDA-core loop sums in whatever order the code picks →
   different rounding.
2. **Input precision.** Tensor Cores get speed by truncating inputs — even the "fp32"
   path is usually **TF32**, which chops the mantissa from 23 to ~10 bits *before*
   multiplying. A CUDA-core fp32 multiply uses the full operands, so the products differ
   before any summation.
3. **Fusion / accumulator width.** A fused multiply-add rounds once; a separate
   multiply-then-add rounds twice. The two paths fuse and accumulate differently.

**Does a CUDA core reduce precision?** Not *secretly* — it computes in exactly the
datatype you give it (feed it fp32, you get a genuine fp32 multiply; it can even do
fp64). Every op still rounds to its type (normal floating-point), but it won't quietly
drop *below* what you asked for. The Tensor Core's fast modes *do* silently downgrade
(TF32/bf16) unless you turn that off (e.g. a framework's `allow_tf32 = false`).

**For determinism**, none of this is random by itself — a fixed path gives identical
bits every run. The trap is the path *changing* between two runs you expect to match:
Tensor Core vs. CUDA core; a different tile shape / reduction order (often triggered
just by a different batch size or sequence length); or **split-K** with atomic
accumulation, which is genuinely nondeterministic run-to-run. To get bit-reproducible
results, pin the whole path: same core type, same precision, same tile config, and a
fixed reduction order with no atomics.
