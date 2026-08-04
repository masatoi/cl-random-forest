# Does FTRL-Proximal's L1 sparsity make Global Pruning easier?

FTRL-Proximal was added to `cl-online-learning` and wired in as a pluggable Global
Refinement learner type (`make-refine-learner-of-type`, see
`docs/superpowers/specs/2026-08-04-ftrl-refinement-pruning-sparsity-design.md` for the
change and its rationale). This report measures whether the L1 sparsity FTRL induces in
the refine learner's weights actually helps Global Pruning (`pruning!` in
`src/random-forest.lisp`), and records the judgment the design doc's criteria produce.

All numbers below are copied verbatim from the trailing comment blocks in
`src/experimental/ftrl-pruning-sparsity.lisp` (one block per measurement run, Task 2 and
Task 3 of the implementing plan). Nothing here was re-measured or extrapolated beyond the
group-sparsity comparison in the section below with that name, which is a direct
computation from those same numbers.

## Conclusion

Threshold-based pruning is viable by the design doc's own criterion: FTRL's
leaf-parent-zero-rate clears the 0.1 bar by lambda1=3 (18.3%) and reaches 40.9% by
lambda1=10, at only a 0.34-point accuracy cost relative to AROW. But sparsity by itself
does not make `pruning!` remove more nodes -- it always deletes a fixed quantile of
leaf-parents regardless of which criterion ranked them -- and pruning AROW and FTRL at
matched rates (0.1, 0.5) produced no accuracy difference after the standard
rebuild-and-retrain step. FTRL's payoff is a data-driven stopping signal, not a better
pruning outcome, and getting it costs 0.3-0.9 accuracy points across the lambda1 range
where it is useful.

## Premises

Three different things get called "sparsity" here, and only the third one bounds what a
pruning criterion could actually remove:

| Metric | Definition | What it measures |
|---|---|---|
| element-zero-rate | Zero fraction over all (class, leaf) weight entries | What the upstream `cl-online-learning` MNIST example reports; not directly tied to pruning |
| leaf-zero-rate | Fraction of leaves whose weight is zero in *every* class | Group sparsity across classes -- the thing the design doc's biggest worry was about |
| leaf-parent-zero-rate | Fraction of leaf-parents whose two children are both all-class-zero (`children-l2-norm` = 0) | What a threshold-based pruning criterion could actually delete |

`pruning!` sorts leaf-parents by `children-l2-norm` and deletes a fixed quantile:

```lisp
(pruning-size (floor (* (length leaf-parents) pruning-rate)))
```

The number of nodes removed is set entirely by `pruning-rate` and the forest's own
leaf-parent count. A leaf-parent scoring exactly `0.0` and one scoring `1e-6` are treated
identically by this code today -- there is no path by which "the weights got sparser"
changes how many nodes a given `pruning-rate` removes. Sparsity can only matter through
(a) changing which leaf-parents a fixed-rate cut selects, or (b) enabling a genuinely
different criterion -- "delete every exact zero" -- that does not use `pruning-rate` at
all. Both are measured below.

## Measured results on letter

Forest config (matches `example/classification/letter.lisp`, plus the flag pruning
needs): `:n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 15
:remove-sample-indices? nil`. `:remove-sample-indices? nil` is required because the
default (`t`) leaves pruned parents unable to answer a query afterwards (issue #14).
Dataset: `letter` (26 classes, 16 dimensions, 15000 train / 5000 test rows). Measured
forest accuracy 91.90%, matching the example's reference baseline (91.4%).

Refine learners were trained for a fixed 20 epochs, not with
`train-refine-learner-process` -- see Limitations for why that mechanism could not be
used. 20 was chosen over the original plan of 10 because lambda1=30 and lambda1=100 were
still visibly climbing at epoch 10 (+0.22 and +0.34 points on the last epoch
respectively). At 20 epochs, lambda1<=10 is flat, lambda1=30 is nearly flat (+0.06 on
the last epoch), and lambda1=100 is still creeping up (+0.20 on the last epoch) -- its
accuracy below should be read as a slightly pessimistic lower bound, not a converged
value.

| learner | lambda1 | accuracy | element-zero | leaf-zero | leaf-parent-zero |
|---|---|---|---|---|---|
| AROW |  | 97.14 | 6.8% | 0.4% | 0.0% |
| FTRL | 0.0 | 97.20 | 0.0% | 0.0% | 0.0% |
| FTRL | 1.0 | 96.84 | 71.6% | 19.9% | 4.9% |
| FTRL | 3.0 | 96.80 | 88.7% | 41.6% | 18.3% |
| FTRL | 10.0 | 96.80 | 96.9% | 63.7% | 40.9% |
| FTRL | 30.0 | 96.28 | 99.0% | 79.9% | 63.4% |
| FTRL | 100.0 | 95.18 | 99.6% | 90.7% | 81.6% |

n-leaf 160389, n-leaf-parent 53986, epochs 20.

`leaf-parent-zero-rate` rises monotonically with lambda1 and clears the 0.1-0.5
operational `pruning-rate` range used in the project's examples by lambda1=10 already:
the element-wise sparsity the upstream MNIST example reports concentrates into whole-leaf
and whole-parent zeros here, rather than staying scattered thinly across (class, leaf)
pairs. Accuracy stays within about a point of the AROW baseline (97.14%) through
lambda1=30 (96.28%, -0.86pt); lambda1=100 costs close to two points (95.18%, -1.96pt,
and per the convergence caveat above that is itself a slight underestimate of where it
would land).

`make-forest`'s bagging has no fixed RNG seed, so exact numbers vary run to run by a few
tenths of a point: across four independent runs, forest accuracy ranged 91.36-91.90% and
AROW refine accuracy 97.14-97.28%. The qualitative picture -- monotonic
leaf-parent-zero-rate, accuracy holding through lambda1=30, lambda1=100 short of
converged -- was stable across all of them.

## Ranking comparison

The interesting question beyond raw sparsity is whether FTRL orders leaf-parents
differently from AROW at all. Measured at lambda1=10 (the largest lambda1 whose accuracy
cost is close to run-to-run noise):

```
N-LEAF-PARENT 54145, SPEARMAN 0.7393, BOTTOM-10%-OVERLAP 0.234,
BOTTOM-50%-OVERLAP 0.784, FTRL-ZERO-COUNT 22046
```

`FTRL-ZERO-COUNT / N-LEAF-PARENT` = 40.7%, matching the sweep's 40.9% up to run-to-run
forest variance.

**Read the overlap numbers with care**, because FTRL's zero-tied block (22046
leaf-parents) is larger than both K windows measured:

- Bottom-10% (K=5414) is *entirely* inside FTRL's zero block. The 23.4% figure is not
  measuring rank agreement -- it measures how much of AROW's true bottom-10% happens to
  fall inside FTRL's much larger, unordered zero set. It should not be read as evidence
  about ranking agreement.
- Bottom-50% (K=27072) is *mostly* inside the zero block (22046 of 27072 = 81.4% of that
  window is the tied zero block; only the remaining 18.6% is drawn from FTRL's actual
  ordering beyond zero). The 78.4% figure is therefore also ties-dominated, though less
  arbitrarily than the 10% figure.
- Only **Spearman (0.739)** is safe to quote as "the two criteria broadly agree," and
  even it carries a caveat: AROW's leaf-parent-zero-rate measured 0.0% throughout this
  project, so AROW's ranking is strictly tie-free, while FTRL's is ~41% tied at zero.
  0.739 compares a tie-free ranking against a heavily-tied one -- it says the two
  criteria broadly agree on which leaf-parents matter least, not that they agree
  leaf-parent by leaf-parent.

## Pruning applied end to end

Four `pruning!` + rebuild + retrain rows, each on its own freshly built forest, at
lambda1=10:

| learner | rate | accuracy before | accuracy after | leaves before | leaves after |
|---|---|---|---|---|---|
| AROW | 0.1 | 97.26 | 97.26 | 159621 | 154236 (-3.4%) |
| FTRL | 0.1 | 97.00 | 97.08 | 159031 | 153662 (-3.4%) |
| AROW | 0.5 | 97.16 | 97.18 | 159987 | 133003 (-16.9%) |
| FTRL | 0.5 | 96.58 | 96.68 | 157923 | 131137 (-17.0%) |

Accuracy does not drop for either learner at either rate: it is exactly flat for AROW at
rate 0.1 and ticks up 0.02-0.10pt for the other three rows after rebuild-and-retrain,
including at rate 0.5, which removes about a sixth of all leaves. This matches the
CVPR2015 global-pruning paper's headline result that a trained, refine-learned forest is
redundant enough to prune substantially and recover full accuracy on retrain.

**This does not show FTRL beating AROW on accuracy at a matched pruning rate.** FTRL's
absolute accuracy sits below AROW's throughout (0.18-0.26pt at rate 0.1, 0.50-0.58pt at
rate 0.5, before and after pruning respectively) -- consistent with the 0.34pt cost
lambda1=10 already carried in the sweep above. Pruning at either rate neither closes that
gap nor widens it.

The leaf-count columns being nearly identical between AROW and FTRL at a given rate
(154236 vs 153662; 133003 vs 131137) is not itself a finding: `pruning!` deletes exactly
`floor(n-leaf-parent * rate)` leaf-parents regardless of which learner ranked them, so
the *count* removed tracks the rate and the forest's own (build-to-build variable)
leaf-parent count, not the learner. What differs between AROW and FTRL is *which*
leaf-parents get removed -- that is what the Spearman/overlap numbers above speak to, not
the counts here.

## Group sparsity

`letter` has 26 classes. A leaf can only be pruned as a whole if all 26 per-class
weights die together, so the question the design doc flagged as the make-or-break one is
how correlated those 26 per-class zero events are. If they were independent,
leaf-zero-rate would equal `element-zero-rate^26`; measured leaf-zero-rate well above
that predicts real positive correlation ("when one class's weight at a leaf has died,
the others are more likely to have died too"), and at or near that value predicts no
group effect beyond chance.

| lambda1 | element-zero-rate | leaf-zero-rate (measured) | independence prediction (element-zero-rate^26) | measured / predicted |
|---|---|---|---|---|
| 1 | 71.6% | 19.9% | 0.017% | ~1170x |
| 3 | 88.7% | 41.6% | 4.43% | ~9.4x |
| 10 | 96.9% | 63.7% | 44.1% | ~1.4x |
| 30 | 99.0% | 79.9% | 77.0% | ~1.04x |
| 100 | 99.6% | 90.7% | 90.1% | ~1.01x |

At every lambda1 measured, actual leaf-zero-rate exceeds the independence prediction --
group sparsity across the 26 classes is real, not an artifact, confirming the mechanism
the design doc's MNIST "670 dead border pixels" anecdote proposed but did not quantify
for this project. The excess is largest in relative terms at moderate sparsity (three
orders of magnitude at lambda1=1) and narrows toward 1x as element-zero-rate itself
approaches 100% (lambda1=30/100): once almost every element is already zero, independence
and correlation predict nearly the same number, so the ratio necessarily compresses even
though the correlation has not gone away -- a ceiling effect, not a sign that correlation
weakens. At lambda1=10, the operating point used for the ranking and pruning
comparisons above, the measured rate (63.7%) is 1.4x the independence prediction
(44.1%): roughly 20 of its 63.7 percentage points come from cross-class correlation
rather than from 26 independent coin flips landing zero together.

(One further, unrequested but cheap comparison: leaf-parent-zero-rate at lambda1=10
(40.9%) is close to leaf-zero-rate squared (63.7%^2 = 40.6%) -- unlike the
strong cross-class correlation above, whether a leaf's *sibling* is also all-zero looks
close to independent of whether the leaf itself is.)

## Judgment

Applying the design doc's criterion directly: *"leaf-parent-zero-rate reaches the
operational pruning-rate range (0.1-0.5 in the examples) -> threshold-based pruning
succeeds; below that -> it does not."* Measured leaf-parent-zero-rate clears 0.1 already
at lambda1=3 (18.3%) and is 40.9% at lambda1=10, comfortably inside the 0.1-0.5 range, at
an accuracy cost of only 0.34 points relative to AROW -- well under the second criterion
("accuracy at the same pruning amount should not fall far from the AROW baseline") and
well under the 1.0-point line the implementing plan calls out as the threshold for
flagging a hard tradeoff. **By the letter of the design doc, this is a pass: 26-class
group sparsity did materialize enough to be usable, so the "it failed, consider
`sparse-softmax+ftrl` instead" branch does not apply.**

That said, "viable" here means a legitimate, data-determined all-zero leaf-parent set
exists to prune -- not that pruning with it beats current practice. The end-to-end
section above found no accuracy advantage over AROW's existing L2-norm-plus-manual-rate
approach at matched rates; both recover (or slightly exceed) their pre-pruning accuracy
after retrain. The concrete thing FTRL adds is a stopping condition that AROW cannot
offer at all: AROW's leaf-parent-zero-rate measured 0.0% throughout this project, so an
AROW-based threshold criterion would prune nothing, ever; `pruning-rate` has to be
guessed by hand. FTRL's exact zeros are a "prune me" signal that exists independent of
any chosen rate.

The natural next step is a threshold-based entry point into `pruning!` alongside the
existing rate-based one -- for example `(pruning! forest learner &key rate threshold)`,
where `:threshold t` (or a numeric cutoff) replaces "delete the lowest `rate` fraction by
`children-l2-norm`" with "delete every leaf-parent whose `children-l2-norm` is exactly
(or below) the threshold." That is naturally self-terminating: run train -> prune ->
rebuild -> retrain in a loop until a pass produces zero deletions, with no
`pruning-rate` to hand-tune. This is scoped as future work; nothing here implements it.

Per the plan's own tradeoff check: reaching leaf-parent-zero-rate >= 0.1 does not require
paying the 1.0-point line -- that line is only crossed at lambda1=100 (-1.96pt, and by
the convergence caveat above, even that is a pessimistic estimate). At lambda1=10, the
value used throughout the ranking and pruning experiments, the accuracy cost is a modest
0.2-0.6pt depending on which table it is read from -- real, but not disqualifying.

## Limitations

- **Single dataset.** Only `letter` was measured; the design doc's own measurement
  procedure made an MNIST cross-check conditional on `letter` looking promising ("if
  promising, cross-check against the existing MNIST baseline"), and that step was not
  carried out. The conclusions above are `letter`-specific (26 classes, 16 dimensions);
  they are not verified to hold at MNIST's class/dimension count or feature density.
- **Unseeded, stochastic forest construction.** `make-forest`'s bagging has no fixed RNG
  seed. Every row above comes from its own forest build; across the runs performed,
  forest accuracy ranged 91.36-91.90% and AROW refine accuracy 97.14-97.28%. The
  qualitative shape (monotonic leaf-parent-zero-rate, accuracy holding through
  lambda1=30, group-sparsity excess shrinking toward 1x as sparsity saturates) was stable
  across reruns, but no formal variance analysis was done.
- **Single run per (lambda1, rate) cell.** No repeated trials, so there is no confidence
  interval on any accuracy number above, including the "pruning does not cost accuracy"
  observation in the end-to-end section, which rests on four single forest builds.
- **lambda1=100 is not fully converged even at 20 epochs** (+0.20pt accuracy on the last
  epoch). Its accuracy (95.18%) is a slightly pessimistic lower bound. Leaf-parent-zero-
  rate at that lambda1 is expected to be close to its final value already, since which
  coordinates are zero is set by whether their |z| accumulator ever crosses lambda1, not
  by how far past it they drift once they do -- but this was not directly verified past
  epoch 20.
- **`train-refine-learner-process`'s rollback-to-best-epoch is broken and was not usable
  for these measurements.** `train-refine-learner-process-inner`
  (`src/random-forest.lisp`) keeps a pre-epoch snapshot of the learner via
  `clol::copy-sparse-arow` / `copy-one-vs-rest` and returns that snapshot when test
  accuracy stops improving. Both are `defstruct`'s default (shallow) copiers --
  `cl-online-learning` defines neither a `:copier` option nor a hand-written copy
  function for either struct -- so the snapshot's weight arrays are the *same* arrays the
  learner keeps training on. What the function returns when it "rolls back" is actually
  whatever the last epoch trained to, not the best epoch. This is why every measurement
  in this project, including the ones in this report, used a fixed epoch count instead of
  that mechanism. FTRL makes the underlying bug worse, not better: it carries
  per-coordinate `z` and `n` accumulators on top of the weight vector, all of which would
  need deep copying to fix this. This is a real bug worth filing as a separate issue; it
  is documented here, not filed, per this task's scope.
- **No search over FTRL's other hyperparameters.** `alpha`/`beta`/`lambda2` were held at
  `0.1`/`1.0`/`1.0` throughout; only `lambda1` was swept. A different `alpha`/`beta`
  could shift where the sparsity/accuracy tradeoff curve sits.
