# Does FTRL-Proximal's L1 sparsity make Global Pruning easier?

FTRL-Proximal was added to `cl-online-learning` and wired in as a pluggable Global
Refinement learner type (`make-refine-learner-of-type`, see
`docs/superpowers/specs/2026-08-04-ftrl-refinement-pruning-sparsity-design.md` for the
change and its rationale). This report measures whether the L1 sparsity FTRL induces in
the refine learner's weights actually helps Global Pruning (`pruning!` in
`src/random-forest.lisp`), and records the judgment the design doc's criteria produce.

All numbers below are copied verbatim from the trailing comment blocks in
`src/experimental/ftrl-pruning-sparsity.lisp` (one block per measurement run, Tasks 2, 3
and 5 of the implementing plan). Nothing here was re-measured or extrapolated beyond the
group-sparsity comparison and the MNIST independence check, both direct computations from
those same numbers.

## Conclusion

Threshold-based pruning is viable by the design doc's own criterion: FTRL's
leaf-parent-zero-rate clears the 0.1 bar by lambda1=3 (18.3%) and reaches 40.9% by
lambda1=10, at only a 0.34-point accuracy cost relative to AROW. But sparsity by itself
does not make `pruning!` remove more nodes -- it always deletes a fixed quantile of
leaf-parents regardless of which criterion ranked them -- and pruning AROW and FTRL at
matched rates (0.1, 0.5) produced no accuracy difference after the standard
rebuild-and-retrain step. FTRL's payoff is a data-driven stopping signal, not a better
pruning outcome, and getting it costs 0.3-0.9 accuracy points across the lambda1 range
where it is useful. A second dataset with fewer classes (MNIST, 10 vs `letter`'s 26)
reproduces the sparsity finding -- leaf-parent-zero-rate clears the operational range at
an equal or lower accuracy cost -- though MNIST was swept for sparsity only; `pruning!`
was not run end to end on it there, so the end-to-end pruning result above is `letter`-only
-- see "MNIST cross-check" below.

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

Up to lambda1=30, actual leaf-zero-rate exceeds the independence prediction even after
accounting for the one-decimal rounding on element-zero-rate: at lambda1=30,
element-zero-rate's true value lies in [98.95%, 99.05%], whose upper end predicts at most
`0.9905^26` = 78.0%, still below the measured 79.9%. Group sparsity across the 26 classes
is real over that range, not an artifact, confirming the mechanism the design doc's MNIST
"670 dead border pixels" anecdote proposed but did not quantify for this project.

The lambda1=100 row cannot be signed the same way. Element-zero-rate there is recorded as
99.6%, one decimal place, so its true value could be anywhere in [99.55%, 99.65%]; at the
top of that interval, `0.9965^26` = 91.3%, which is *above* the measured leaf-zero-rate of
90.7%. Whether the measured rate exceeds or falls short of the independence prediction at
lambda1=100 is therefore undetermined by the precision recorded here -- this is not a
small excess, it is a sign that cannot be read off these numbers, and this report does not
re-measure at higher precision to settle it.

The excess, where it is established (lambda1<=30), is largest in relative terms at
moderate sparsity (three orders of magnitude at lambda1=1) and narrows toward 1x as
element-zero-rate itself approaches 100% (lambda1=10/30): once almost every element is
already zero, independence and correlation predict nearly the same number, so the ratio
necessarily compresses even though the correlation has not gone away -- a ceiling effect,
not a sign that correlation weakens. At lambda1=10, the operating point used for the
ranking and pruning comparisons above, the measured rate (63.7%) is 1.4x the independence
prediction (44.1%): roughly 20 of its 63.7 percentage points come from cross-class
correlation rather than from 26 independent coin flips landing zero together.

(One further, unrequested but cheap comparison: leaf-parent-zero-rate at lambda1=10
(40.9%) is close to leaf-zero-rate squared (63.7%^2 = 40.6%) -- unlike the
strong cross-class correlation above, whether a leaf's *sibling* is also all-zero looks
close to independent of whether the leaf itself is.)

## MNIST cross-check

The measurements above are all `letter`-specific (26 classes). Since the design doc's
own criterion for viability rests entirely on group sparsity -- a leaf is only prunable
once its weight is zero in *every* class -- class count is the natural axis to vary to
test whether the finding generalizes. This section repeats the sweep on MNIST (10
classes, 784 dimensions, 60000 train / 10000 test rows), at the same three lambda1
values (3.0, 10.0, 30.0) that bracketed the operational pruning-rate range on `letter`.

Forest config matches `example/classification/mnist.lisp`'s `mnist-forest` (`:n-tree 500
:bagging-ratio 0.1 :max-depth 10 :n-trial 10 :min-region-samples 5`), plus
`:remove-sample-indices? nil` for the same reason as `letter`. MNIST's labels needed one
extra step beyond `letter`'s: `read-data` subtracts 1 from every LIBSVM label, correct
for 1-based label files, but MNIST's labels already start at 0 -- a `shift-labels-up!`
helper adds 1 back after reading, matching what `example/classification/mnist.lisp` does
inline.

Before sweeping, the forest was sanity-checked against that example's recorded baseline:
measured forest accuracy 93.46% versus the example's 93.38%, and 98760 leaf-parents
versus its 98008 before pruning -- both close enough to rule out a labeling or dimension
bug (a doubled or missing label shift would have driven accuracy far below 93%, not
within a tenth of a point of it) and consistent with the build-to-build variance
`make-forest`'s unseeded bagging already produces on `letter` (91.36-91.90% across four
runs).

| learner | lambda1 | accuracy | element-zero | leaf-zero | leaf-parent-zero |
|---|---|---|---|---|---|
| AROW |  | 98.27 | 7.2% | 0.8% | 0.0% |
| FTRL | 3.0 | 98.08 | 87.3% | 52.1% | 25.5% |
| FTRL | 10.0 | 97.93 | 94.5% | 73.5% | 52.2% |
| FTRL | 30.0 | 97.76 | 97.4% | 85.3% | 71.0% |

n-leaf 249251, n-leaf-parent 98283, 20 epochs (this is the sweep's own forest build,
separate from the sanity-check build above).

**10 classes versus 26, matched lambda1** (`letter` numbers repeated from the table
above):

| lambda1 | dataset (classes) | accuracy | delta vs AROW | leaf-zero | leaf-parent-zero |
|---|---|---|---|---|---|
| 3 | letter (26) | 96.80 | -0.34 | 41.6% | 18.3% |
| 3 | MNIST (10) | 98.08 | -0.19 | 52.1% | 25.5% |
| 10 | letter (26) | 96.80 | -0.34 | 63.7% | 40.9% |
| 10 | MNIST (10) | 97.93 | -0.34 | 73.5% | 52.2% |
| 30 | letter (26) | 96.28 | -0.86 | 79.9% | 63.4% |
| 30 | MNIST (10) | 97.76 | -0.51 | 85.3% | 71.0% |

At every matched lambda1, MNIST's leaf-zero-rate and leaf-parent-zero-rate are both
higher than `letter`'s, and MNIST's accuracy cost is smaller or equal, never larger.
**Fewer classes made whole-leaf zeros easier to reach, not harder, and did so at a lower
accuracy cost** -- the opposite of the direction that would have undermined the design
doc's group-sparsity premise. This holds despite MNIST's element-zero-rate being
comparable to, or even slightly *below*, `letter`'s at the same lambda1 (94.5%/97.4% vs
96.9%/99.0% at lambda1=10/30): the extra leaf- and leaf-parent-level sparsity on MNIST is
not coming from more sparsity per weight, it is coming from needing fewer classes'
weights to die together before a whole leaf (or leaf-parent) zeros out.

The same independence check as the Group Sparsity section above, run on MNIST's 10
classes:

| lambda1 | element-zero-rate | leaf-zero-rate (measured) | independence prediction (element-zero-rate^10) | measured / predicted |
|---|---|---|---|---|
| 3 | 87.3% | 52.1% | 25.7% | ~2.03x |
| 10 | 94.5% | 73.5% | 56.8% | ~1.29x |
| 30 | 97.4% | 85.3% | 76.9% | ~1.11x |

Measured leaf-zero-rate exceeds the independence prediction at every lambda1 here too --
cross-class correlation is real at 10 classes, the same qualitative finding as `letter`'s
26-class table (ratios there: ~9.4x/1.4x/1.04x at the same three lambda1). The *ratio* to
the independence baseline is smaller on MNIST purely because the baseline itself is
larger with fewer classes -- raising a fraction below 1 to the 10th power shrinks it less
than raising it to the 26th -- not because the correlation is weaker: MNIST's absolute
leaf-zero-rate is higher than `letter`'s at every matched lambda1 despite the smaller
ratio.

Against the existing pruning figures already in `example/classification/mnist.lisp`
(98008 leaf-parents before `pruning!` at rate 0.1, 93228 after, refine accuracy 98.259%,
all under the AROW learner): AROW's leaf-parent-zero-rate here measured 0.0%, same as
`letter`, so AROW offers no data-driven stopping signal on MNIST either -- `pruning-rate`
still has to be chosen by hand, independent of class count. FTRL's leaf-parent-zero-rate
clears that file's low-end operating rate (0.1) already at lambda1=3 (25.5%, versus
`letter` needing lambda1=3 to reach only 18.3%) and clears the whole 0.1-0.5 range by
lambda1=10 (52.2%) -- MNIST reaches the design doc's viability bar at least as easily as
`letter` did, at a smaller accuracy cost (0.19-0.51pt across lambda1=3-30, versus
`letter`'s 0.34-0.86pt).

**This confirms, and if anything strengthens, the `letter` finding: reducing class count
from 26 to 10 did not make group sparsity harder to obtain -- it made it easier, on both
axes (rate and accuracy cost) that matter for pruning.** One piece of evidence does not
carry over from the `letter` measurement: per-epoch convergence. This run's driver script
printed only `print-sweep`'s final-epoch summary, not each row's per-epoch accuracy
curve, so unlike `letter` there is no direct evidence here that lambda1=30 had flattened
out by epoch 20 on MNIST. MNIST's smaller accuracy costs at every lambda1 (never larger
than `letter`'s already-converged-at-20-epochs costs at the same lambda1) and its 4x
larger training set (60000 vs 15000 rows, so 4x more gradient updates per epoch) both
suggest convergence should be at least as fast as `letter`'s here, not slower -- but this
was not directly measured and is recorded as a limitation below, not a verified fact.

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

- **Two datasets, not a general class-count sweep.** The design doc's own measurement
  procedure made an MNIST cross-check conditional on `letter` looking promising; that
  check was carried out (see "MNIST cross-check" above) and reproduced -- and
  strengthened -- the qualitative `letter` finding at 10 classes, 784 dimensions. But two
  data points (10 and 26 classes) do not establish a trend, and both datasets share the
  project's other example defaults (`bagging-ratio 0.1`, comparable tree counts); the
  finding is not verified to hold at other class counts, dimensionalities, or feature
  densities beyond these two.
- **MNIST epoch-convergence not directly verified.** The MNIST sweep's driver printed
  only the final-epoch summary, not each row's per-epoch accuracy curve (unlike `letter`,
  where the curves confirmed lambda1<=10 had converged by epoch 20 and lambda1=30/100
  were still creeping up). MNIST's smaller accuracy costs at every lambda1 and its 4x
  larger training set both suggest its convergence is at least as fast as `letter`'s, but
  this is an inference, not a measurement.
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
  need deep copying to fix this. Filed as issue #20,
  https://github.com/masatoi/cl-random-forest/issues/20 ("train-refine-learner-process
  never rolls back to the best epoch"); the mechanism above is the detail needed to fix
  it.
- **No search over FTRL's other hyperparameters.** `alpha`/`beta`/`lambda2` were held at
  `0.1`/`1.0`/`1.0` throughout; only `lambda1` was swept. A different `alpha`/`beta`
  could shift where the sparsity/accuracy tradeoff curve sits.
