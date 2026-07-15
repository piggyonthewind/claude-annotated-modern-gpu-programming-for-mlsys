# Annotated Read: "The Evolution of Tensor Core Data Layouts"

A companion to <https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_layout_generations/index.html>

**Audience:** same as the other companions — an ordinary software engineer, no GPU
background assumed. This chapter is the most hardware-specific one so far, and it leans
hard on **Chapter 3 (Data Layout)** — shape-stride, register fragments, named axes,
swizzle. If those are fuzzy, skim `data_layout.md` first. We keep the factory picture:

> **warehouse** = global memory (GMEM) · **team's table** = shared memory (SMEM) ·
> **worker** = thread · **32 workers** = a warp · **128 workers** = a warpgroup ·
> **results-board** = TMEM · **matrix machine** = Tensor Core.

The chapter's sections:

1. Two Memory-Access Requirements to Keep in Mind
2. Ampere — operands *and* accumulators in registers
3. Hopper — reading directly from shared memory
4. Blackwell — moving accumulators into TMEM
5. Per-Thread Register Fragments Across Generations
6. Comparing the Three Data Paths

---

## 0. The problem this chapter is about: the machine is *picky* about its tray

The Tensor Core (matrix machine) doesn't just accept a pile of numbers. It demands its
inputs **arranged in one exact way** — a specific "which thread holds which element" or
"which SMEM slot holds which element" arrangement. Call it the machine's **tray**: the
operand layout it insists on.

So every GEMM has a chore hiding inside it: **get the tile out of the warehouse and
onto the machine's tray in exactly the arrangement it wants** — while *also* obeying the
two hard memory rules (§1). That chore is pure overhead — it's not the multiply, it's
the *arranging* — and it costs registers, instructions, and latency.

This chapter is the **history of who does that arranging and how much it costs**, across
three GPU generations:

- **Ampere:** the workers hand-arrange everything into their own hands (registers), and
  the machine's output comes back into their hands too. Lots of staging.
- **Hopper:** you hand the machine a *map* to the ingredients on the table, and it
  fetches them itself in the right arrangement. Workers stop pre-arranging inputs.
- **Blackwell:** same map for inputs, and now the machine writes its running totals onto
  its own dedicated board (TMEM) instead of into the workers' hands. Registers freed.

> **The one design principle:** each generation **removes a layer of intermediate
> staging** between memory and the Tensor Core — cutting latency and freeing the
> register file for other work.

---

## 1. Two Memory-Access Requirements — the constraints in the background

Before the generations, the chapter reminds you of two rules any layout must respect
(both from Chapters 2–3). The Tensor Core's tray has to be reachable *without breaking
these*:

- **Global-memory reads must coalesce.** When a warp reads from the warehouse, the 32
  threads should touch **contiguous, aligned** addresses so the hardware fuses them into
  a few wide transactions (Performance §3). Scattered reads waste the road.
- **Shared-memory reads must spread across banks.** The table (SMEM) is split into banks;
  if many threads hit one bank at once they **serialize** (a bank conflict, Data Layout
  §5). Layouts must scatter simultaneous accesses across banks.

The tension: the Tensor Core wants its operands in a particular tray arrangement, but
that arrangement still has to load from the warehouse coalesced *and* sit in the table
conflict-free. **Satisfying all three at once is exactly what makes these layouts
intricate** — and what the swizzles and descriptors below exist to handle.

---

## 2. Ampere — everything lives in registers

On the Ampere generation, both the **inputs (A, B)** and the **accumulator (C/D)** of a
matrix multiply live in the **workers' hands (registers)**, scattered across the 32
threads of a warp as **register fragments** (Data Layout §3). The whole warp issues one
`mma.sync` instruction *collectively* — all 32 threads participate in one matrix
multiply — using the example shape **m16n8k16** (a 16×16-ish tile per instruction).

The data path is: **SMEM → (load into the fragment layout) → registers → (`mma.sync`) →
registers**.

### A concrete fragment mapping (what "scattered across lanes" really means)
The machine's tray is a precise rule assigning each element to a specific lane and
register slot. The chapter works an example for the accumulator. Using each thread's
**lane id** `l` (0–31), define a group and a position:

```
g = l // 4      (which group of 4 lanes)
t = l  % 4      (position within the group)
```

Then that lane holds **four** fp32 accumulator values, at logical coordinates
`(g, 2t)`, `(g, 2t+1)`, `(g+8, 2t)`, `(g+8, 2t+1)`. Concretely, **lane 5**: `g=1, t=1`
→ it owns C at `(1,2)`, `(1,3)`, `(9,2)`, `(9,3)`. The A operand (fp16/bf16) is 8
elements per lane in 4 registers; B is 4 elements per lane in 2 registers (two 16-bit
values packed per 32-bit register). This exact scatter *is* the machine's tray — feed it
anything else and you get garbage.

You can write that scatter in Chapter 3's notation, e.g. `S[(8,4,2):(4@laneid,
1@laneid, 1@reg)]` — the point isn't the symbols, it's that **the operand layout is a
precise function from logical `(i,j)` to `(which lane, which register)`.**

### `ldmatrix` — the instruction that builds the fragment
Getting each lane to hold *its* scattered elements by hand would be miserable — every
thread would gather four elements from four different SMEM spots. So Ampere added a
dedicated instruction:

> **`ldmatrix`** = a special load that takes an **8×8 tile in SMEM** and **distributes
> it across the 32 threads in exactly the fragment layout `mma.sync` expects**, in one
> shot. Variants `.x1/.x2/.x4` load 1/2/4 tiles at once; `.trans` loads it transposed
> (column-major). Lanes 0–7 supply the row base-addresses; every lane receives its two
> packed 16-bit elements based on `(l//4, l%4)`.

`ldmatrix` is the "hand the workers their tray positions automatically" tool. Its
reverse, **`stmatrix`** (added on Hopper), stores a register fragment back to SMEM the
same way.

### Writing back and swizzling
When you write results back to SMEM (for the epilogue) you hit the bank-conflict problem
head-on: a plain **row-major** layout writes rows fine (spread across banks) but a
**column** read lands all 8 elements in **one bank** → an 8-way conflict. Ampere's
answer is **hand-written XOR swizzle** (Data Layout §5): the kernel author computes
XOR-permuted addresses so row writes stay coalesced *and* column reads scatter across
banks. On Ampere, **you** manage that swizzle math explicitly. Remember that — it's the
thing Hopper takes off your plate.

**Ampere in one line:** the warp explicitly stages A/B from SMEM into register fragments
(via `ldmatrix`), the multiply reads and writes registers, and you hand-manage swizzle.
Maximum control, maximum staging overhead, maximum register pressure.

---

## 3. Hopper — the machine reads the table itself

Hopper's big move: **stop staging inputs into registers at all.** Instead of loading A
and B into the fragment layout first, you just tell the Tensor Core *where the tile is
on the table* and let it read directly.

- The operation is **`wgmma.mma_async`**, issued by a whole **warpgroup** (4 warps = 128
  threads) cooperating on one MMA.
- Its inputs come **straight from SMEM**: B always from SMEM; A from SMEM *or* registers
  (the two modes are **SS** = both from SMEM, and **RS** = A from registers, B from
  SMEM).
- It's **asynchronous** — you launch it and move on (this is what enables the pipeline
  from Background §4).

### The matrix descriptor — a map instead of a staging pass
How does the machine "know where the tile is"? You hand it a **descriptor**:

> **Matrix descriptor** = a **64-bit value that is a recipe for finding and reading a
> tile in SMEM**: the start address, the stride offsets that step between rows/row-groups
> (`ldo` = leading-dimension offset, `sdo` = stride-dimension offset), a base offset, and
> the **swizzle mode**. All the address fields are encoded in **16-byte units**. The
> Tensor Core follows this recipe to pull operands in the right arrangement itself.

Example the chapter gives: a **K-major** tile with **128-byte swizzle** has `ldo` fixed
at 1 while `sdo` steps between eight-row groups; the swizzle mode in the descriptor tells
the hardware the atom shape and XOR permutation to apply. **The crucial win: the swizzle
you hand-computed on Ampere is now just a *field in the descriptor* — the hardware
applies it for you.** No more `ldmatrix` staging, no more hand-written XOR addresses for
inputs.

### Accumulators are still in registers
One thing Hopper *didn't* change: the accumulator **C/D still lives in per-thread
registers** across the warpgroup. The epilogue consumes those register fragments (the
instruction shape decides how many elements each thread holds). So a Hopper kernel
juggles **two** layout representations at once: **descriptors** for the SMEM-resident
A/B, and **register fragments** for the accumulator (and for A when using RS mode).

**Hopper in one line:** inputs are read straight from the table via a 64-bit descriptor
(swizzle handled in hardware), eliminating the register-staging step — but the running
total still sits in the workers' hands.

---

## 4. Blackwell — the accumulator moves onto its own board (TMEM)

Blackwell finishes the job by taking the *last* big thing out of registers: the
accumulator. The operation is **`tcgen05.mma`**.

- **A and B** are still read from SMEM via descriptors (as on Hopper).
- **The accumulator C/D now lives in TMEM** — the dedicated results-board (Background §2,
  Data Layout §3), not in registers.
- Execution is **asynchronous**; the epilogue pulls the result out of TMEM with
  **`tcgen05.ld`** and then waits for it with **`tcgen05.wait::ld`**.

So the full path is: **A/B in SMEM → (`tcgen05.mma`) → C/D in TMEM → (`tcgen05.ld`) →
register fragment → store to GMEM**. Registers are no longer tied up holding a big
accumulator during the mainloop, which frees the register file for prologue/epilogue
work. (This is the concrete reason TMEM exists — see the "why a separate memory" note in
Background §2.)

> **One rule that gets sharper here:** the **write** layout (how `tcgen05.mma` lays the
> result into TMEM) and the **read** layout (how `tcgen05.ld` pulls it out) **must
> match**. Mismatch = wrong data. More on this invariant in §6.

### Block-scaled MMA and scale factors (the fiddly new part)
Blackwell pushes very low precision (fp8/fp4), and those need **per-block scaling** to
keep numerics sane. So the MMA takes two extra little matrices of **scale factors**:

- **SFA** of shape **M × SFK** and **SFB** of shape **N × SFK** (SFK = number of
  scale-blocks along K). The MMA multiplies each K-block by the matching scale.

These scale factors take a special route that **bypasses the normal SMEM operand path**:
**GMEM → (TMA) → SMEM → (`tcgen05.cp`) → TMEM**, and the MMA selects the right scale from
TMEM by block index.

> **`tcgen05.cp`** = the copy instruction that writes scale factors into TMEM in the
> layout the MMA expects. The chapter's layout example
> `S[(4,32,4):(4@TCol,1@TLane,1@TCol)] + R[4:32@TLane]` maps `(Mgroup, lane, sfk)` to
> TMEM byte positions, and the `.32x128b.warpx4` form writes one 32-lane window then
> **broadcasts** it to the other three windows of the warpgroup (this is the replication
> from Data Layout §4).

### Word-level replication (`scale_vec`)
Finally, a knob for how a scale value is repeated *inside* a single 32-bit TMEM word:

- **`scale_vec::1X`** → `[SF0, SF0, SF0, SF0]` (one scale, repeated)
- **`scale_vec::2X`** → `[SF0, SF1, SF0, SF1]` (a pair, repeated)
- **`scale_vec::4X`** → `[SF0, SF1, SF2, SF3]` (four distinct K-block scales)

and `SFA_ID`/`SFB_ID` pick which copy the MMA reads. You don't need to memorize this —
just recognize it's the fine-grained end of the same **replication/offset** notation from
Data Layout §4, applied down at the level of bytes within a word.

**Blackwell in one line:** inputs via descriptors (like Hopper), but the accumulator
moves off registers into purpose-built TMEM, and block-scaling adds a separate
scale-factor path into TMEM — freeing registers and enabling fp8/fp4.

---

## 5. Per-Thread Register Fragments Across Generations

The **register fragment** — the idea that a matrix is scattered across the lanes, each
holding a piece — is the one concept that survives all three generations, but its **job
shifts**:

| Generation | What the register fragment is *for* |
|------------|-------------------------------------|
| **Ampere** | holds **A, B, and the accumulator** throughout the compute — everything is a fragment |
| **Hopper** | A/B moved to SMEM descriptors; fragment now mainly holds the **accumulator** during compute |
| **Blackwell** | accumulator moved to TMEM; fragment is now just the **bridge between TMEM and the epilogue** (via `tcgen05.ld/.st`, with several shape options) |

So the fragment goes from "where *everything* lives" (Ampere) to "a temporary staging
buffer for the epilogue" (Blackwell). Same mechanism, steadily smaller role — which is
just another view of the "remove a staging layer each generation" theme.

---

## 6. Comparing the Three Data Paths

Line them up and the trend is one clean progression — **each generation moves the operand
source one step closer to the machine and off the registers**:

| | **Ampere** | **Hopper** | **Blackwell** |
|---|---|---|---|
| Instruction | `mma.sync` | `wgmma.mma_async` | `tcgen05.mma` |
| Scope | warp (32) | warpgroup (128) | warpgroup (128) |
| A / B source | **registers** (staged via `ldmatrix`) | **SMEM** via descriptor | **SMEM** via descriptor |
| Accumulator | registers | registers | **TMEM** |
| Swizzle | **hand-written** XOR | **descriptor field** (hardware) | descriptor field (hardware) |
| Async? | no (`.sync`) | yes | yes |

Each step down the table **reduces per-thread latency and register pressure**: Ampere
stages everything explicitly; Hopper hides the swizzle and input staging behind
descriptors; Blackwell additionally evicts the accumulator into TMEM.

### The invariant rule that ties it all together
Across every generation, one rule never bends:

> **The output layout of one instruction must be exactly the input layout the next one
> expects.** The tray a stage produces has to be the tray the next stage consumes — the
> `ldmatrix` output must match `mma.sync`'s input; the `tcgen05.mma` write layout must
> match the `tcgen05.ld` read layout; the epilogue's fragment must match what the store
> wants. A mismatch doesn't error loudly — it silently produces **wrong numbers** or
> tanks performance. This is *why* a precise layout notation (Chapter 3) matters so much:
> it's how you *prove* two stages' trays line up.

**The whole chapter in one sentence:** getting a tile into the Tensor Core's exact
required arrangement is real work, and GPU generations have progressively removed that
work — Ampere stages operands and accumulator in registers with hand-managed swizzle;
Hopper reads operands straight from SMEM via hardware descriptors; Blackwell also moves
the accumulator into TMEM — with the unbreakable rule that each stage's output layout
must match the next stage's input layout.

---

## Glossary (quick lookup)

| Term | Plain meaning |
|------|---------------|
| **operand layout** | the exact arrangement (which lane/slot holds which element) the Tensor Core demands |
| **register fragment** | a matrix scattered across the warp's lanes, each thread holding a piece |
| **lane id** | a thread's index within its warp (0–31) |
| **`mma.sync`** | Ampere matrix-multiply-accumulate; whole warp, operands+accumulator in registers |
| **`ldmatrix`** | Ampere load that distributes an 8×8 SMEM tile into the fragment layout the MMA wants |
| **`stmatrix`** | the reverse: store a register fragment back to SMEM (Hopper) |
| **swizzle** | XOR address permutation to avoid SMEM bank conflicts (hand-written on Ampere) |
| **`wgmma.mma_async`** | Hopper warpgroup MMA; reads operands directly from SMEM, asynchronous |
| **warpgroup** | 4 warps = 128 threads acting as one unit |
| **matrix descriptor** | a 64-bit recipe (address, strides `ldo`/`sdo`, swizzle mode) for reading an SMEM tile |
| **SS / RS mode** | wgmma input modes: both operands from SMEM (SS), or A from registers + B from SMEM (RS) |
| **`tcgen05.mma`** | Blackwell MMA; A/B from SMEM descriptors, accumulator into TMEM, async |
| **`tcgen05.ld` / `.st`** | Blackwell load/store moving data between TMEM and register fragments |
| **`tcgen05.cp`** | Blackwell copy that writes scale factors into TMEM in the required layout |
| **TMEM** | Blackwell's dedicated on-chip accumulator board (Background §2) |
| **block-scaled MMA** | low-precision (fp8/fp4) MMA that scales each K-block by a scale factor |
| **scale factors (SFA, SFB)** | the `M×SFK` and `N×SFK` matrices of per-block scales |
| **`scale_vec::1X/2X/4X`** | how many distinct scales are packed/repeated within a 32-bit TMEM word |
| **invariant rule** | output layout of one instruction must equal the input layout of the next |

---

## How to read the source chapter now

With this open beside it, the chapter should decode as:

1. **Two memory requirements** → any Tensor-Core layout must still coalesce GMEM reads
   and avoid SMEM bank conflicts.
2. **Ampere** → operands + accumulator are register fragments; `ldmatrix` builds the
   fragment; you hand-manage XOR swizzle.
3. **Hopper** → `wgmma` reads operands straight from SMEM via a 64-bit descriptor
   (swizzle handled in hardware); accumulator still in registers.
4. **Blackwell** → `tcgen05.mma` keeps SMEM descriptors for inputs but moves the
   accumulator into TMEM; block-scaling adds a scale-factor path into TMEM.
5. **Fragments across generations / three paths** → the fragment's role shrinks from
   "holds everything" to "epilogue bridge," and every generation removes a staging layer
   — under the invariant that adjacent instructions' layouts must match.

The one idea under all of it: **the Tensor Core is picky about how its data is arranged,
arranging it costs registers and latency, and the whole history is about doing that
arranging closer to the metal each generation — while never letting two stages' layouts
fall out of sync.**
