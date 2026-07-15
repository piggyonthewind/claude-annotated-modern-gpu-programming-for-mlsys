# Annotated Read: "Data Layout and Its Notation"

A companion to <https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_data_layout/index.html>

**Audience:** same as the other companions — an ordinary software engineer, no GPU
background assumed. This chapter is more abstract than Background and Performance: it's
about **notation** — a compact algebra for writing down *where each number physically
lives*. We'll build it from one idea you may already know (how NumPy/PyTorch store an
array) and keep the factory picture from the earlier chapters:

> **warehouse** = global memory (GMEM) · **team's table** = shared memory (SMEM) ·
> **worker** = thread · **32 workers** = a warp · **results-board** = TMEM.

The chapter's sections:

1. The Shape-Stride Model
2. Tile Layout
3. Named Axes (TMEM and register fragments)
4. Replication and Offset
5. Swizzle Layout

---

## 0. Why a whole chapter about "where numbers live"?

In Background and Performance we moved "tiles" around without ever saying *exactly*
which physical slot each number sits in. But to actually write these kernels you must
be precise about that, because **the physical arrangement of the data decides whether
your memory accesses are fast or slow** (coalescing, bank conflicts — Performance §3),
whether a transpose costs a copy or is free, and how data maps onto weird hardware
memories like TMEM and registers.

So this chapter builds a **notation** — a small, consistent way to write down the
mapping from a number's *logical* position (row `i`, column `j`) to its *physical*
location (which memory slot / which lane / which GPU). Think of it as the algebra that
lets you *reason* about layout instead of guessing. Once you have it, coalescing,
transposes-without-copies, MMA operand layouts, broadcasts, multi-GPU sharding, and
swizzling all become the *same* notation with different numbers plugged in.

The single foundational idea:

> **Data layout = a function that turns a logical index into a physical location.**
> Everything in this chapter is a richer and richer version of that one function.

---

## 1. The Shape-Stride Model — the core trick

### Memory is 1-D; matrices are 2-D
Start with the fundamental mismatch. A matrix is a 2-D grid — you think of it as
`A[i][j]`. But the warehouse (memory) is really **one long shelf of numbered slots**:
address 0, 1, 2, 3, … It has no rows or columns. So somebody has to decide **how to
flatten** the 2-D grid onto the 1-D shelf — which slot holds `A[i][j]`.

That decision is the layout, and the rule is astonishingly simple: **each logical
dimension is assigned a "stride" — how many slots you jump when that index goes up by
one.** Then:

```
physical address = i·stride_i + j·stride_j        (a dot product of index and strides)
```

### Row-major vs column-major = just different strides
Take a 3×4 matrix (3 rows, 4 columns):

- **Row-major** (C/NumPy default): lay each row down contiguously. Moving one column
  (`j → j+1`) steps **1** slot; moving one row (`i → i+1`) steps **4** slots (skip a
  whole row). Strides = `(4, 1)`. Element `(1,2)` → `1·4 + 2·1 = slot 6`.
- **Column-major** (Fortran/MATLAB/cuBLAS default): lay each column down contiguously.
  Strides = `(1, 3)`. Element `(1,2)` → `1·1 + 2·3 = slot 7`.

Same logical matrix, two different physical arrangements — captured entirely by the
stride numbers. The chapter writes this as a compact pair:

> **`S[(shape):(strides)]`** — e.g. `S[(3,4):(4,1)]` is a 3×4 row-major matrix.
> The shape says how big each dimension is; the strides say how far apart consecutive
> elements of that dimension sit in memory. To find any element's address, dot the
> index vector with the stride vector. That's the whole model.

If you've used NumPy/PyTorch, this **is** how they work — `.shape` and `.strides` are
exactly these two tuples. The chapter's notation is the same idea made first-class.

### The payoff: transpose and reshape are *free*
Here's why the model is powerful. To **transpose**, you don't move any data — you just
**swap the shape/stride pairs**. A row-major `S[(3,4):(4,1)]` transposed becomes
`S[(4,3):(1,4)]`: same physical slots, relabeled so `i` and `j` trade roles. The
numbers never move; only the *addressing rule* changes. This is a **view** — a
different logical window onto the same physical bytes. (It's exactly why `x.T` in
NumPy is instant and shares memory with `x`.) Reshapes that are compatible with the
strides are free the same way.

So the shape-stride model already buys you the most important trick: **many "data
movements" are actually just changes of the addressing function, costing nothing.**

---

## 2. Tile Layout — nesting the model for blocks

Background/Performance kept chopping matrices into **tiles**. How do you write the
layout of a *tiled* matrix? You **nest** the shape-stride model.

Take the chapter's example: an **8×8** matrix physically arranged as **2×4 tiles**
(each tile is 2 rows × 4 columns). Now a single element isn't located by one `(i,j)` —
it's located by a **hierarchy**:

```
(which tile row, which tile col,  row within tile, col within tile)
```

i.e. a *nested* coordinate. And the layout function generalizes in the obvious way:
**each level of the hierarchy gets its own shape and its own stride**, and the address
is still the dot product — just over the nested coordinates. Moving to the next tile
over jumps by one stride; moving one element inside a tile jumps by a smaller stride.

Two operations the chapter names:

- **Unflatten:** take a flat index `0…63` and *decompose* it into the nested
  coordinates `(tile-row, tile-col, in-tile-row, in-tile-col)` according to the shape
  hierarchy. It's integer div/mod, the same way you'd turn a flat array index back into
  `(row, col)` — just with more levels.
- **The general layout function:** the single rule `address = Σ (coordinate_k ·
  stride_k)` applied over *arbitrarily nested* shapes/strides. One uniform formula
  describes flat matrices, tiled matrices, tiles-of-tiles, whatever — you just add
  levels.

Why this matters: tiling (from Performance) and the swizzles/fragments later in this
chapter are all *just* particular choices of nested shapes and strides. The tile layout
is the bridge from "flat array" to "the structured, blocked layouts real kernels use."

---

## 3. Named Axes — when "location" isn't a single number

So far a physical location was **one integer** (a shelf-slot address). That works for
the warehouse (GMEM) and the table (SMEM), which really are linear arrays. But two GPU
memories are **not** linear, and the model has to grow to describe them.

### The problem
- **TMEM** (the Blackwell results-board, Background §2) is physically a **2-D grid**:
  **128 rows (one per lane) × up to 512 columns** of 32-bit words. A location there is
  a *pair* `(row, column)`, not one address.
- **A register fragment** is even stranger. When a Tensor Core MMA uses an 8×8 operand,
  that operand is **not stored in one place** — its 64 elements are **scattered across
  the 32 lanes of the warp**, each lane (thread) holding a couple of them in its own
  registers. To point at logical element `(i,j)` you must say *"lane number L, register
  slot r"* — again a pair, in a completely different coordinate system.

A single integer address can't express either of these. So the chapter tags coordinates
with **named axes**:

> **Named axes** = labels that say *which physical coordinate system* a number lives in.
> Tags like `@TLane` / `@TCol` (TMEM's row/column), or `@laneid` / `@reg` (which
> thread, which register slot). The layout function's *output* becomes a **tuple of
> tagged coordinates** instead of a single integer.

### The two-dimensional TMEM address space
A TMEM tile (e.g. a 128×256 accumulator) is addressed by `(@TLane, @TCol)`: `@TLane`
picks one of the 128 lane-rows, `@TCol` picks a 32-bit column. The layout function maps
your logical `(i,j)` to this pair. (The chapter also notes a 32-bit `@TCol` can be
split into 4 byte-wide sub-columns for packing smaller types.)

### Register fragment
The **m8n8 fragment**: an 8×8 logical tile = 64 elements spread over 32 lanes = **2
elements per lane**. The layout function maps logical `(i,j)` → `(@laneid, @reg)`:
"element `(i,j)` lives in lane 5's register slot 1," etc. This mapping is *fixed by the
hardware* — it's the exact arrangement the Tensor Core expects its operands in — which
is why you need a precise notation for it: to feed the MMA correctly, or to move data
between SMEM and this fragment layout, you must know exactly which lane holds which
logical element.

The takeaway: **"where does this number live" can be a coordinate in lane-space or
register-space, not just a shelf address — and named axes are how you write that down
unambiguously.**

---

## 4. Replication and Offset — the same data in many places

Two more things real hardware does to data, and the notation for each.

### Replication (broadcast) — copies of the same logical data
GPUs can **broadcast**: physically duplicate the same logical value into several
locations so many consumers can read it in parallel without contention. The chapter's
example is **block-scaling scale factors** shared across warps: a small set of scale
values gets copied into each of the four 32-lane windows of a warpgroup (a `.warpx4`
broadcast) so every warp has a local copy.

> **Replication notation `R[shape:strides@axis]`** records a *replica dimension*: how
> many copies exist and where each physical copy sits. Crucially, the copies are the
> *same logical data* — indexing "replica 0" and "replica 3" gives identical values;
> they just live at different physical spots.

Example the chapter uses: four `Mgroup` replicas along `@TLane` at offsets 0, 32, 64,
96 — i.e. the same data mirrored into four lane-windows.

### Offset — shift the base, don't duplicate
Distinct from replication:

> **Offset notation `O[strides@axis]`** just **translates the base location** — it
> shifts *where the whole layout starts* — without making any copies. Same single copy
> of the data, addressed from a different starting point.

The distinction matters: **replication = one logical value, many physical copies (a
broadcast); offset = one physical copy, relocated base.** They look similar in code but
mean opposite things about how many bytes actually exist.

### Same notation scales to multi-GPU meshes
The elegant part: this exact machinery describes **data spread across multiple GPUs**.
Model a **2×2 mesh** of GPUs with device axes `@gpuid_x`, `@gpuid_y`. Then a tensor can
be:

- **fully sharded** — each GPU holds a distinct slice (no replicas),
- **sharded + replicated** — some data copied across GPUs (`R[...]` on a device axis),
- **sharded + offset** — slices placed with a shifted base (`O[...]` on a device axis).

So the *same* replication/offset notation you used for lanes inside one chip also
expresses how a tensor is partitioned across a whole cluster of GPUs. One algebra, from
register slots up to a multi-GPU mesh.

---

## 5. Swizzle Layout — permuting addresses to dodge bank conflicts

The last section fixes a concrete performance bug using a clever layout.

### The bug: bank conflicts (recap + detail)
The team's table (SMEM) is split into **banks** — think 32 parallel tellers, each
owning some of the slots (typically each 4-byte word maps to `bank = (word index) mod
32`). A warp of 32 lanes can read **one word from each bank simultaneously** — full
speed — *but only if the 32 lanes hit 32 different banks.* If several lanes hit the
**same** bank at once, those accesses **serialize** (the teller handles them one by
one). That's a **bank conflict** (Performance §3).

Here's when it bites: store a tile **row-major** with a row width that's a multiple of
32. Reading a **row** is fine — consecutive elements fall in consecutive banks. But
reading a **column**, every element is one full row apart, so they all land in the
**same bank** → a 32-way conflict → 32× slower. And GEMM needs to read the same SMEM
tile **both** by row and by column (one operand each way), so *some* access is going to
be a column read. A plain row-major layout can't make both directions conflict-free.

### The fix: XOR swizzle
> **Swizzle** = permute the physical addresses within the tile — using an **XOR** of
> some index bits — so that the elements you tend to access *together* (a column) get
> **scattered across different banks**, while the *logical* shape is unchanged (you
> still index `[i][j]` normally). It's a remapping of *where* each element physically
> sits, not of what the element *is*.

Concretely, the column index used for storage gets XOR-ed with some bits of the row
index. That nudges each successive row's element into a *different* bank, so a column
read now touches 32 distinct banks instead of one. Row reads stay conflict-free too. The
XOR is cheap (hardware bit-twiddle) and reversible, so addressing stays exact.

### Swizzle modes
The permutation comes in fixed granularities, defined by an **atom** of **8 rows × a
byte width**:

> **`SWIZZLE_128B`, `SWIZZLE_64B`, `SWIZZLE_32B`** — atoms 8 rows tall and 128 / 64 /
> 32 bytes wide (plus finer interleaved 16-byte modes). **Selection rule:** pick the
> **largest** atom width that fits the tile's **contiguous** dimension. Example: an fp16
> tile whose contiguous row is 64 elements = 64×2 = **128 bytes** → use `SWIZZLE_128B`.
> Every access to that tile must use the *same* mode, or the addressing won't line up.

A couple of size facts the chapter leans on: the repeating swizzle unit is a **16-byte
sector**, and bank mapping is commonly at **4-byte** granularity.

Formally, swizzle is a **nonaffine** (XOR-based) transform *composed on top of* the
plain **affine** (linear dot-product) layout from §1–§2. So the full picture is: base
shape-stride layout, then a swizzle permutation layered over it — still one well-defined
function from logical index to physical slot, just with a bit-twiddle in the middle.

---

## Glossary (quick lookup)

| Term | Plain meaning |
|------|---------------|
| **data layout** | the function mapping a logical index `(i,j,…)` to a physical location |
| **shape** | the size of each logical dimension |
| **stride** | how many memory slots you jump when one index increases by 1 |
| **shape-stride `S[(shape):(strides)]`** | the compact way to write a layout; address = index·strides |
| **row-major** | rows stored contiguously (NumPy default); strides like `(ncols, 1)` |
| **column-major** | columns stored contiguously (cuBLAS default); strides like `(1, nrows)` |
| **view** | a re-strided window on the same bytes (transpose/reshape with no copy) |
| **tile layout** | a nested shape-stride describing a blocked/tiled matrix |
| **unflatten** | split a flat index into nested `(tile, in-tile)` coordinates |
| **general layout function** | one dot-product rule over arbitrarily nested shapes/strides |
| **named axes** | tags for non-linear coordinate systems (`@TLane`, `@TCol`, `@laneid`, `@reg`) |
| **TMEM address space** | the 2-D `(@TLane, @TCol)` grid: 128 rows × ≤512 32-bit cols |
| **register fragment** | an MMA operand scattered across warp lanes (m8n8 = 2 elems/lane) |
| **replication `R[…@axis]`** | same logical data broadcast into multiple physical copies |
| **offset `O[…@axis]`** | shift the layout's base location without copying |
| **GPU mesh** | a grid of GPUs (`@gpuid_x/@gpuid_y`) over which a tensor is sharded/replicated |
| **bank** | one of ~32 parallel SMEM sub-memories; same-bank simultaneous access serializes |
| **bank conflict** | multiple lanes hitting one bank at once → serialized, slow |
| **swizzle** | XOR-based address permutation that spreads accesses across banks |
| **swizzle modes** | `SWIZZLE_128B/64B/32B` — atoms of 8 rows × that byte width |
| **sector** | a 16-byte contiguous unit; the repeating swizzle granularity |
| **affine vs nonaffine** | the base layout is linear; swizzle adds a nonlinear XOR on top |

---

## How to read the source chapter now

With this open beside it, the chapter should decode as:

1. **Shape-Stride Model** → memory is 1-D, so a layout is just strides turning `(i,j)`
   into an address; transpose/reshape are free re-stridings (like NumPy).
2. **Tile Layout** → nest the shape-stride model to describe blocked matrices; one
   dot-product rule over nested coordinates.
3. **Named Axes** → for TMEM and register fragments, a "location" is a *tuple of tagged
   coordinates* (`@TLane/@TCol`, `@laneid/@reg`), not a single address.
4. **Replication & Offset** → `R[…]` broadcasts the same data into many physical copies;
   `O[…]` just shifts the base — and the same notation describes multi-GPU sharding.
5. **Swizzle** → XOR-permute physical addresses so both row and column reads hit
   distinct SMEM banks, killing bank conflicts, while the logical shape is untouched.

The one idea under all of it: **a layout is a function from logical index to physical
place** — and this chapter just keeps enriching that function (nesting for tiles, tuples
for exotic memories, replicas/offsets for broadcasts and meshes, an XOR for swizzling)
until it can describe every arrangement a real GPU kernel actually uses.
