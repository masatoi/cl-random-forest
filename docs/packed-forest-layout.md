# The packed forest layout

A description of how `src/experimental/compiled-trees.lisp` flattens a trained random
forest into arrays for inference, written so the scheme can be reviewed and better ones
proposed. It states the constraints an alternative has to satisfy, the decisions actually
taken and what each was measured to be worth, and the candidates that were noticed but not
tried.

Everything here is experimental. `src/experimental/` is outside every ASDF system and is
loaded by hand.

## 1. What it replaces

cl-random-forest represents a trained tree as linked `node` structs. Prediction walks them:

```lisp
(defun predict-dtree (dtree datamatrix datum-index)
  (let ((dist (node-class-distribution (find-leaf (dtree-root dtree) datamatrix datum-index))))
    (argmax dist)))
```

`find-leaf` chases a pointer per level, reading `node-test-attribute` and
`node-test-threshold` out of a struct at each one. Then `node-class-distribution` **recounts
the leaf's class histogram from its `sample-indices`, on every single prediction** — leaf
values are deliberately not cached (see CLAUDE.md's "Why every predict/test call takes
(model datamatrix datum-index)").

A forest sums those per-tree distributions and takes the argmax of the total:

```lisp
(defun class-distribution-forest (forest datamatrix datum-index)
  ;; zero the accumulator
  (dolist (dtree (forest-dtree-list forest))
    (let ((dist (node-class-distribution (find-leaf (dtree-root dtree) datamatrix datum-index))))
      (loop for i below n-class do (incf (aref class-count-array i) (aref dist i)))))
  ;; divide by n-tree
  class-count-array)
```

Two costs are worth separating, because they behave differently. The histogram recount
scales with how many training samples reached the leaf, and dominates for shallow trees:
on MNIST a depth-5 tree walks at 184k predictions/s while a depth-15 tree walks at 889k,
because the shallow one has 60000 samples spread over 32 leaves. The traversal itself is
comparatively cheap.

## 2. Constraints any layout must satisfy

These are not preferences. Each has bitten during this work.

**C1. Exact agreement with `predict-forest`.** Not "similar accuracy" — the same answer on
every datum. Folding each leaf to its own argmax and taking a majority vote is a
*different classifier*: an earlier sketch in the repository did that and differed on 48 of
10000 MNIST predictions without anyone noticing. The forest sums normalised distributions.
Ties in `argmax` go to the lowest class index. `count-packed-forest-disagreements` exists
to enforce this and must read zero.

**C2. Splits are numeric only.** `(>= (aref datamatrix datum-index attribute) threshold)`
takes the **left** branch. There are no categorical splits and no missing-value handling,
so there is nothing corresponding to LightGBM's `decision_type_`. Getting the comparison
direction backwards is an easy and silent error.

**C3. Thresholds are `single-float`, features are `fixnum` attribute indices.** The whole
library is single-float; a double-float declaration under `(safety 0)` reads today's arrays
as the wrong type without complaining.

**C4. The layout is a derived read-only view.** Training, pruning, feature importance and
reconstruction all need the `node` structs and are not replaced. Building must therefore be
cheap: the iterated pruning loop in `src/experimental/fused-sibling-refinement.lisp`
rebuilds the forest 16 to 19 times, so a representation that costs 48 seconds to build (the
compiled one) is unusable there and one that costs 0.05 seconds is free.

**C5. Prediction must write only a caller-supplied accumulator.** `predict-forest`
accumulates into `forest-class-count-array` and every leaf read overwrites
`dtree-class-count-array`, both slots on the shared model. Predicting from several threads
through it returns up to 80% wrong answers, silently. Reentrancy is the property that makes
parallel inference possible at all.

**C6. Leaf numbering should coincide with the refine index space.** This one was discovered
rather than designed, and is worth keeping. Global Refinement maps a datum to the vector of
leaf indices it reaches, offset per tree by `forest-index-offset`. `set-leaf-index!` numbers
leaves with `do-leaf`, which recurses left then right; the packed builder emits leaves in
the same order and counts globally across trees in `forest-dtree-list` order. The two
numberings turn out to be identical — verified on 20000 (datum, tree) pairs with zero
mismatches — so the packed traversal's result *is* the refine index, and refinement can run
on the packed form with no payload array and no remapping.

## 3. The layout as implemented

Structure of arrays, one array per node attribute, flattened across the whole forest rather
than per tree.

```lisp
(defstruct (packed-forest (:constructor %make-packed-forest))
  (n-class 0 :type fixnum)
  (n-tree 0 :type fixnum)
  (n-internal 0 :type fixnum)
  (n-leaf 0 :type fixnum)
  (feature    ... :type (simple-array (unsigned-byte 32) (*)))   ; n-internal
  (threshold  ... :type (simple-array single-float (*)))         ; n-internal
  (left       ... :type (simple-array (signed-byte 32) (*)))     ; n-internal
  (right      ... :type (simple-array (signed-byte 32) (*)))     ; n-internal
  (roots      ... :type (simple-array (signed-byte 32) (*)))     ; n-tree
  (distributions ... :type (simple-array single-float (* *)))    ; n-leaf x n-class
  (leaf-classes  ... :type (simple-array (unsigned-byte 32) (*)))) ; n-leaf, :class payload
```

**Only internal nodes get slots.** A leaf is encoded as a negative child index, `~leaf`, so
leaf 0 is -1. `roots` may itself be negative, for a tree that never split.

**Node ordering is preorder**, per tree, trees in `forest-dtree-list` order: a node is
assigned its index, then its left subtree is emitted, then its right.

Traversal, one loop for the whole forest:

```lisp
(dotimes (tree n-tree)
  (let ((node (aref roots tree)))
    (loop while (>= node 0)
          do (setf node (if (>= (aref datamatrix datum-index (aref feature node))
                                (aref threshold node))
                            (aref left node)
                            (aref right node))))
    (let ((row (lognot node)))
      (dotimes (k n-class) (incf (aref acc k) (aref table row k))))))
```

The loop's continuation test is the leaf test, so a leaf costs no extra read.

`build-packed-forest` also takes `:leaf-payload :class`, which stores each leaf's argmax in
`leaf-classes` instead, for the single-tree case where no summing is needed.

## 4. What each decision was measured to be worth

The layout was arrived at in three steps, each measured against the previous one on the
same forest in the same process. Rates are predictions per second on the test set.

**Step 1 — flatten at all.** Against the library's walking prediction, on 500-tree forests:
letter depth 10, 3274 → 8696; MNIST depth 10, 2278 → 8217. Most of that is the leaf
histogram disappearing, not the layout.

**Step 2 — type the struct slots.** The slots were originally untyped, so the accessors
returned `T` and every caller had to re-declare what it read; one place did not, and SBCL
fell back to a runtime dispatch on the array element type there. Compiling with notes
visible found exactly two, both on that expression. Typing the slots took notes to zero.
Single-tree rates on letter: 92.4M → 102.8M (+11.5%) at depth 5, 51.0M → 54.7M (+7.1%) at
depth 10, 38.4M → 40.3M (+7.5%) at depth 15. The forest path, which had no notes, did not
improve.

**Step 3 — leaves as negative indices, and 32-bit instead of fixnum indices.** Node arrays
fell to between a quarter and a third: MNIST 500 trees at depth 10, 14.0 MB → 4.0 MB.
Speed followed size, and the gain grows with the model rather than shrinking:

| | array | packed | ratio |
|---|---|---|---|
| letter tree d=5 | 103.2M | 123.5M | 1.19x |
| letter tree d=15 | 41.2M | 46.2M | 1.14x |
| letter 500x d=5 | 38910 | 55350 | 1.43x |
| letter 500x d=10 | 8361 | 15267 | 1.84x |
| MNIST tree d=5 | 49.4M | 84.2M | 1.79x |
| MNIST 500x d=5 | 56074 | 68610 | 1.23x |
| MNIST 500x d=10 | 8077 | 19047 | 2.35x |

## 5. Against compiling the trees into code

The obvious alternative is to emit each tree as a nest of `if`s and call `compile`, turning
thresholds into immediates and the traversal into straight-line branches. That is also
implemented, in the same file. All four representations on one forest each:

| | internal nodes | compile | code MB | packed MB | walk | compiled | array | packed |
|---|---|---|---|---|---|---|---|---|
| letter 500x d=5 | 14551 | 2.3 s | 0.8 | 1.7 | 7396 | 52173 | 40257 | 53860 |
| letter 500x d=10 | 113547 | 16.1 s | 5.9 | 13.0 | 3274 | 13495 | 8696 | 16051 |
| MNIST 500x d=5 | 15485 | 1.9 s | 0.8 | 0.8 | 2834 | 61255 | 56074 | 68259 |
| MNIST 500x d=10 | 259401 | 47.9 s | 13.5 | 13.9 | 2278 | 10438 | 8217 | 19120 |

Packed is fastest in all four, by 1.03x to 1.83x over compiled, and the margin grows with
the model. Compiling loses at scale for a reason worth stating, because it constrains what
a better layout should avoid: 500 separate compiled functions are far slower than one
function called 500 times. Measured, the traversals in a compiled forest cost 13.4 µs where
500 times a single tree's own measured time is 3.1 µs at MNIST depth 5, and 52.3 against
4.0 at depth 10. A tree timed alone runs with its code hot in the instruction cache; a
forest touches 500 code blobs once each per prediction. One small loop over data does not
have that problem.

On memory the two are close — 13.5 MB of machine code against 13.9 MB of arrays on MNIST's
largest forest — but two thirds of the packed total is the leaf class distributions
(9.9 MB), not the tree structure (4.0 MB). letter, with 26 classes, spends 11.4 of its 13.0
MB the same way. A regression forest, whose leaf value is a scalar, would be far smaller.

## 6. Structural facts a better scheme could exploit

Measured on a 200-tree letter forest at depth 12, 58171 internal nodes and 58371 leaves:

- **`left[i]` is always exactly `i + 1` when it is not a leaf.** All 28528 of the internal
  left children satisfied it. This follows from preorder emission and leaves not taking
  slots, and it means the `left` array carries no information beyond "is the left child a
  leaf, and if so which one". A scheme that encoded that in one bit plus a leaf number
  could drop a whole 4-byte-per-node array.
- **`right[i]` is always ahead of `i` or a leaf.** All 58171 satisfied it, so right children
  are forward references only.
- Internal nodes and leaves are within one of each other per tree (58171 against 58371 for
  200 trees), as a binary tree requires.
- Feature indices are tiny here — letter has 16 features, MNIST 784 — so 32 bits per
  feature is already generous and 16 would do for both.

## 7. Candidates not tried

Listed because a reviewer should know they were considered and left alone, not overlooked.

**Drop the `left` array.** Section 6 shows it is redundant given preorder. The information
that remains is one bit per node plus the leaf number, which could go in the unused high
bits of `feature`, or in `right`'s sign, or a separate bit vector — though a bit vector is
exactly what removing the previous `leaf-p` array was worth 1.1-2.4x, so re-introducing one
needs care about where the bit is read from.

**Narrow the feature index to 16 bits.** Sufficient for both datasets here and for most
tabular problems; halves that array. Needs a fallback for wide data.

**Bin the thresholds, as LightGBM actually does.** LightGBM does not compare floats at
prediction time: it bins each feature during training and stores `threshold_in_bin_` as an
integer, comparing integers. That would replace a 4-byte float with a 1- or 2-byte bin
index and turn a float comparison into an integer one, but it changes training, not just
the layout, and it is an approximation of the split — so it collides with constraint C1
unless the binning is exact.

**Array of structures rather than structure of arrays.** This is the sharpest open
question. SoA is right when a pass touches one field across many nodes. Tree descent
touches *all four fields of one node* and then jumps somewhere unpredictable — which is the
case AoS is for: one cache line per node instead of three or four separate streams. A node
packed as (feature:u16, threshold:f32, right:s32, flags) fits in 16 bytes, four to a cache
line. This was not tried, and nothing measured here distinguishes it. LightGBM uses SoA,
but LightGBM's arrays are per-tree and small.

**Cache-conscious node ordering.** Preorder is one choice; breadth-first, or a
van Emde Boas layout, would change which nodes share a cache line. Depth-5 trees fit in
cache under any ordering, so this only matters for the deep forests, which is exactly where
the current scheme is weakest.

**Perfect-tree indexing for shallow trees.** A tree of bounded depth can be stored as a
complete binary tree indexed by the path bits, with no child pointers at all. Wasteful for
unbalanced trees, free of pointer arrays for balanced ones.

**Batch prediction.** Everything here predicts one datum at a time. Pushing many data
through one tree before moving to the next would touch each node's data once per batch
instead of once per datum, and is the shape SIMD would need.

## 8. What is measured and what is not

Measured: exact agreement with the library on every datum of both test sets, at every
configuration; throughput for all four representations on one forest; bytes for the array
layouts and machine code size for the compiled one; build time; the effect of each of the
three design steps in isolation.

Not measured: parallel scaling of the packed layout specifically — the array layout reached
3.7-4.9x on eight threads, and packed's working set is 3.5 times smaller, so it should do
better, but that has not been run. Regression forests are not implemented at all; their
leaf value is a scalar and the distribution table would collapse to a vector. Nothing has
been serialized to disk; the library has no persistence of any kind today.

## 9. Reproducing

```lisp
(ql:quickload :cl-random-forest-test/fixture)
(load "src/experimental/compiled-trees.lisp")
(in-package :cl-random-forest/src/experimental/compiled-trees)

(run-four-ways :dataset :letter)          ; the four-way table
(run-layout-comparison :dataset :mnist)   ; array against packed
(run-decomposition :dataset :mnist)       ; where a forest's time goes
(run-parallel-scaling :dataset :letter)   ; scaling, and what walking does in threads
```

MNIST needs `dataset/mnist.scale` and `mnist.scale.t`; letter is fetched by the test
fixture. Measurements were taken on x86-64 SBCL, one run each unless stated.
