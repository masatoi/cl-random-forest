# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Guidelines

@prompts/repl-driven-development.md
@prompts/common-lisp-expert.md

These two prompts are copied verbatim from the `cl-mcp` repository, so a few of their
assumptions do not hold here:

- `repl-driven-development.md` describes the **cl-mcp MCP tools** (`repl-eval`,
  `lisp-edit-form`, `clgrep-search`, …). They only apply when the `cl-mcp` MCP server is
  connected to the session; otherwise fall back to the built-in Read/Edit/Grep/Bash tools.
- Both prompts assume **Rove** as the test framework and `mallet` as the linter. This project
  uses Rove too (`asdf:test-system`, `rove`) but has **no lint step** — see *Commands* below.

## Overview

Common Lisp implementation of Random Forest for multiclass classification and univariate
regression, plus **Global Refinement** and **Global Pruning** of a trained forest
(Ren, Cao, Wei, Sun, "Global Refinement of Random Forest", CVPR2015). Multivariate regression
is not implemented.

## Commands

Load (requires `cl-online-learning` and `cl-libsvm-format` from the same author, checked out
into quicklisp/roswell `local-projects`):

```lisp
(ql:quickload :cl-random-forest)   ; package nickname: CLRF
```

Run the test suite (rove-based):

```lisp
(asdf:test-system :cl-random-forest)
```

```sh
./t/run-test.ros                   # what CI runs; roswell with dynamic-space-size=2048
rove cl-random-forest-test.asd     # rove's own runner -- prints nothing under CCL, see below
```

`rove <system>.asd` mutes `*standard-output*` and routes rove's report through a
synonym-stream lookup that resolves differently on CCL, so under CCL it runs the suite but
prints nothing and the exit code is the only signal. CI uses `./t/run-test.ros` for that
reason; prefer it when you need to see results.

Tests are split into feature systems, each of which can be run on its own:

| System | Tests |
|---|---|
| `cl-random-forest-test/dataset` | dataset download and conversion (3) |
| `cl-random-forest-test/decision-tree` | decision tree accuracy (2) |
| `cl-random-forest-test/forest` | random forest accuracy (2) |
| `cl-random-forest-test/refinement` | global refinement accuracy, learner-type plumbing and convergence detection (9) |
| `cl-random-forest-test/parallel` | parallelized training accuracy (4, SBCL only) |
| `cl-random-forest-test/regression` | univariate regression behaviour (5) |
| `cl-random-forest-test/pruning` | global pruning behaviour (5) |

`cl-random-forest-test` is the aggregate that runs all seven.
`cl-random-forest-test/fixture` holds the shared dataset loaders and helpers and has no tests.

Load the feature system first (`ql:quickload` or `asdf:load-system`), then:

```lisp
(rove:run :cl-random-forest-test/forest)                              ; one feature
(rove:run-test 'cl-random-forest-test/forest::a9a-forest-accuracy)    ; one test
```

There is no lint step. CI (`.github/workflows/ci.yml`) runs the matrix
{sbcl-bin, ccl-bin} × {ubuntu-latest, macOS-latest}.

Test/example caveats:
- `cl-random-forest-test/regression`, `.../pruning`, and the seven synthetic tests in
  `.../refinement` (`refine-learner-default-path-unchanged`, three `refine-learner-of-type-*`
  and three `refine-learner-process-*`) need no network: they use
  `cl-random-forest-test/fixture`'s deterministic synthetic data, or build their own.
  Everything else downloads datasets.
- Of the regression and pruning suites' ten tests, seven are **property assertions**, not
  pinned accuracy numbers, and three deliberately pin bugs that are still open: `regression-refine-learner-default-gamma-diverges`
  (issue #16), `pruning-strands-leaves-without-sample-indices` (issue #14) and
  `pruning-does-not-update-forest-n-leaf` (issue #15). Each says so in a comment. When one of
  them starts **failing**, the underlying bug has been fixed and the test should be replaced by
  a positive assertion rather than "repaired".
- Datasets are loaded lazily and memoized in `cl-random-forest-test/fixture`, so any single
  test can be run on its own and will fetch only what it needs.
- The three `refine-learner-process-*` tests synthesise a refine dataset directly rather than
  building a forest, so they run in under a second. They also carry deliberate label noise:
  on separable data the best and last epoch coincide and the contract they check is not
  observable.
- `train-refine-learner-process` needs a `cl-online-learning` with `clol:copy-learner`.
  After updating that library, recompile this one (`asdf:load-system :cl-random-forest
  :force t`): SBCL open-codes `defstruct` copiers, so a stale fasl keeps calling the old
  shallow one and the bug reappears silently.
- The tests **download datasets over the network** (`wget` on the `PATH`) into `dataset/` under
  the system directory (gitignored). Same for most `example/` files.
- The five dataset-driven suites are accuracy assertions with a `±1.0` tolerance over averaged
  runs — they are slow (500-tree forests, repeated 5–10×) and inherently stochastic.
- Some examples need `dynamic-space-size >= 2500` (noted in their headers); start SBCL via
  `ros -Q dynamic-space-size=4096`.

## Architecture

### System layout

`package-inferred-system`: **one package per file, package name = path**
(`cl-random-forest/src/random-forest` ⇔ `src/random-forest.lisp`). Adding a file needs no
`.asd` change — dependencies come from `defpackage`.

`cl-random-forest.lisp` is the façade: it `use-reexport`s `src/random-forest`,
`src/reconstruction`, `src/feature-importance` into `:cl-random-forest` (nickname `:clrf`).
Note `src/utils` is *not* reexported; import from it explicitly.

- `src/random-forest.lisp` — everything core: `dtree`/`node`/`forest` structs, classification
  and regression trees/forests, global refinement, global pruning.
- `src/utils.lisp` — sampling, the parallelization macros, libsvm-format and clol dataset readers.
- `src/feature-importance.lisp` — Mean Decrease Accuracy (OOB permutation) and Mean Decrease Impurity.
- `src/reconstruction.lisp` — invert a leaf assignment back into an approximate input vector.
- `src/experimental/` — **not part of the system**; nothing depends on these packages, so they are
  never compiled by `quickload`. `workspace.lisp` is a REPL scratch file.
- `example/` — also not part of the system; load individual files manually.

### Data representation (strict)

- `datamatrix`: `(simple-array single-float (n-datum n-dim))`, one **row** per datum.
- classification `target`: `(simple-array fixnum (n-datum))`, class ids starting at **0**.
- regression `target`: `(simple-array single-float (n-datum))`.

All internal arithmetic is `single-float` (converted from `double-float` repo-wide; older
example code may still be stale). `make-dtree`/`make-rtree` `check-type` these, so a
`double-float` matrix fails loudly.

### Why every predict/test call takes `(model datamatrix datum-index)`

Training mutates preallocated scratch arrays hung off the `dtree` (`tmp-arr1/2`, `best-arr1/2`),
then `clean-dtree!` nils out both the scratch arrays **and** `dtree-datamatrix`. A trained tree
retains only `target` and the node structure. Hence prediction never takes a datum vector — the
caller passes the datamatrix back in plus a row index. Leaf values are also *not* cached: each
prediction recomputes the class distribution (`node-class-distribution`) or mean
(`node-regression-mean`) from the leaf's `sample-indices` on the fly.

### Regression reuses the classification structs

There is no separate rtree/regression-forest type. `make-rtree` builds a `dtree` with `n-class`
left `nil`, and `rtree?` tests exactly that; `%print-forest` branches the same way. When touching
`split-node!` / `set-best-children!`, remember the gain computation differs: classification
weights child gains by sample-count ratio, regression does not.

### The two optional flags exist for the downstream modules

`make-dtree`/`make-forest` take `remove-sample-indices?` (default `t`) and `save-parent-node?`
(default `nil`):

- `remove-sample-indices? t` nils `node-sample-indices` on every node that gets split, including
  the **root**. Feature importance needs the root's indices to derive the OOB set
  (`dtree-oob-sample-indices`), so it requires forests built with `:remove-sample-indices? nil`.
- `save-parent-node? t` links children back to parents; `src/reconstruction.lisp` walks upward
  from a leaf, so it requires this.

### Global refinement pipeline

`make-refine-dataset` maps each datum to the vector of leaf indices it reaches, one per tree,
offset into a single global index space via `forest-index-offset` (cumulative
`dtree-max-leaf-index`). That vector is fed to `cl-online-learning` as a sparse vector:
`make-refine-learner`'s default is sparse AROW for binary, `one-vs-rest` of sparse AROWs for
multiclass, sparse RLS for regression. `make-refine-learner-of-type` swaps in any other
`cl-online-learning` multiclass sparse learner (e.g. `sparse-lr+ftrl`, for L1-sparse weights) in
place of AROW; it is multiclass-only, since `one-vs-rest` needs more than 2 classes.
`train-refine-learner` is one epoch; `train-refine-learner-process` loops until dev accuracy stops
improving (and is a **macro** — it `setf`s its first argument).

`pruning!` sorts leaf-parent nodes by the L2 norm of the refine learner's weights for their two
leaves, deletes the lowest `pruning-rate` fraction, and re-runs `set-leaf-index-forest!`. The leaf
index space therefore changes: after pruning you **must** rebuild both the refine dataset and the
refine learner before training again.

### Parallelization

`lparallel:*kernel*` is the only switch. `dotimes/pdotimes`, `mapcar/pmapcar`, `mapc/pmapc`,
`push-ntimes` in `src/utils.lisp` expand to a runtime `(if lparallel:*kernel* ...)`, so setting
the kernel to `nil` restores serial execution with no recompilation. Parallelized:
`make-forest`, `make-regression-forest`, `make-refine-dataset`, `train-refine-learner`.

## Known broken code

`src/feature-importance.lisp` and `src/reconstruction.lisp` `:use` only the *exported* symbols of
`src/random-forest`, but their bodies call unexported internals (`find-leaf`, `do-leaf`,
`traverse`, `square`, `rtree?`, `dtree-root`, `dtree-target`, `node-*` accessors,
`make-refine-vector`, `calc-accuracy`) and `mapcar/pmapcar` from `src/utils`. Those names intern
as fresh, unbound symbols, so the system compiles and loads but the entry points fail at runtime
(e.g. `(forest-feature-importance ...)` → "The function ... MAPCAR/PMAPCAR is undefined").

The working convention is what `example/regression/simple-regression.lisp` does:
`(:import-from #:cl-random-forest/src/random-forest #:traverse #:do-leaf #:dtree-root ...)`.
Fixing either module means adding the needed `:import-from` clauses (or exporting the internals).

Also in that file: `src/feature-importance.lisp` exports `#:forest-feature-importance-by-impurity`
but defines `forest-feature-importance-impurity`.
