;;; -*- coding:utf-8; mode:lisp -*-
;;;
;;; Does FTRL-Proximal's L1 sparsity make global pruning easier?
;;;
;;; Load with the test fixture system on the path, then LOAD this file:
;;;
;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;   (load "src/experimental/ftrl-pruning-sparsity.lisp")
;;;   (in-package :cl-random-forest/src/experimental/ftrl-pruning-sparsity)
;;;   (run-letter-sweep)
;;;
;;; Not part of any system. See docs/superpowers/specs/2026-08-04-*.md for the design.

;;; This file needs MAKE-L2-NORM, COLLECT-LEAF-PARENT, CHILDREN-L2-NORM and
;;; DTREE-MAX-LEAF-INDEX -- everything the pruning criterion is made of -- which are
;;; internal to SRC/RANDOM-FOREST and not re-exported by the :CL-RANDOM-FOREST facade
;;; (unlike src/experimental/multi-grained-scanning.lisp, which only needs exported
;;; symbols and so lives directly in `(in-package :cl-random-forest)`). That need does
;;; not license defining this file's own 17 helpers -- SPARSITY-REPORT, LEAF-COUNT,
;;; SPEARMAN, *EPOCHS* and the rest -- inside SRC/RANDOM-FOREST itself: doing so would
;;; intern them straight into the library's own package, with no collision today but a
;;; real one waiting (e.g. issue #15's eventual real LEAF-COUNT). So this file gets its
;;; own package instead, following the convention CLAUDE.md's "Known broken code"
;;; section documents and example/regression/simple-regression.lisp demonstrates: :USE
;;; CL and the CL-RANDOM-FOREST facade for everything exported, plus explicit
;;; :IMPORT-FROM clauses naming each internal symbol this file needs from
;;; SRC/RANDOM-FOREST and SRC/UTILS (the facade does not re-export SRC/UTILS at all).
(defpackage :cl-random-forest/src/experimental/ftrl-pruning-sparsity
  (:use #:cl
        #:cl-random-forest)
  (:import-from #:cl-random-forest/src/random-forest
                #:make-l2-norm
                #:collect-leaf-parent
                #:children-l2-norm
                #:dtree-max-leaf-index)
  (:import-from #:cl-random-forest/src/utils
                #:read-data))

(in-package :cl-random-forest/src/experimental/ftrl-pruning-sparsity)

;;;; Sparsity metrics
;;;;
;;;; Three different things get called "sparsity" here and only the third one bounds
;;;; how much PRUNING! can remove:
;;;;
;;;;   element-zero-rate    zeros among all (class, leaf) weights
;;;;   leaf-zero-rate       leaves whose weight is zero in EVERY class
;;;;   leaf-parent-zero-rate  leaf-parents whose two children are both all-class-zero
;;;;
;;;; PRUNING! sorts leaf-parents by CHILDREN-L2-NORM and deletes a fixed quantile, so
;;;; the third number is the fraction a threshold-based criterion could delete instead.

(defun sparsity-report (forest learner)
  "Return the three sparsity rates for LEARNER's weights over FOREST as a plist."
  (let* ((l2 (make-l2-norm learner))
         (parents (collect-leaf-parent forest))
         (n-class (clol::one-vs-rest-n-class learner))
         (learners (clol::one-vs-rest-learners-vector learner))
         (weight-of (clol::one-vs-rest-learner-weight learner))
         (n-element 0)
         (n-element-zero 0))
    (dotimes (k n-class)
      (let ((w (funcall weight-of (svref learners k))))
        (dotimes (j (length w))
          (incf n-element)
          (when (zerop (aref w j)) (incf n-element-zero)))))
    (list :element-zero-rate (/ (float n-element-zero) n-element)
          :leaf-zero-rate (/ (float (count 0.0 l2)) (length l2))
          :leaf-parent-zero-rate
          (/ (float (count-if (lambda (node) (zerop (children-l2-norm node l2 forest)))
                              parents))
             (length parents))
          :n-leaf (length l2)
          :n-leaf-parent (length parents))))

(defun leaf-parent-scores (forest learner)
  "CHILDREN-L2-NORM of every leaf-parent, in COLLECT-LEAF-PARENT order.
The order is what makes two learners' scores comparable element by element: the node
list comes from the same forest, so index I is the same node in both vectors."
  (let* ((l2 (make-l2-norm learner))
         (parents (collect-leaf-parent forest))
         (scores (make-array (length parents) :element-type 'single-float)))
    (loop for node in parents
          for i from 0
          do (setf (aref scores i) (children-l2-norm node l2 forest)))
    scores))

;;;; letter fixture
;;;;
;;;; Settings copied from example/classification/letter.lisp (forest 91.4%, refine
;;;; 97.34% there), plus :remove-sample-indices? nil, which PRUNING! needs -- with the
;;;; default t every pruned parent becomes a leaf that cannot answer a query (issue #14).

(defun letter-forest ()
  "Return (values forest refine-train refine-test train-target test-target) for letter."
  (multiple-value-bind (datamatrix target) (cl-random-forest-test/fixture:letter-train)
    (multiple-value-bind (datamatrix-test target-test)
        (cl-random-forest-test/fixture:letter-test)
      (let ((forest (make-forest cl-random-forest-test/fixture:+letter-n-class+
                                 datamatrix target
                                 :n-tree 500 :bagging-ratio 0.1
                                 :min-region-samples 5 :n-trial 10 :max-depth 15
                                 :remove-sample-indices? nil)))
        (values forest
                (make-refine-dataset forest datamatrix)
                (make-refine-dataset forest datamatrix-test)
                target
                target-test)))))

;;;; The sweep

(defparameter *epochs* 20
  "Fixed epoch count. TRAIN-REFINE-LEARNER-PROCESS is unusable here: its rollback keeps
a shallow struct copy that shares the weight arrays, so it always returns the last
epoch, not the best one.
Raised from 10 to 20 on the letter sweep: at 10 epochs AROW and low-lambda1 FTRL had
plateaued, but lambda1 >= 30 was still climbing (lambda1=100 by +0.34 accuracy on the
last epoch alone). At 20 epochs lambda1 <= 10 is flat; lambda1=30/100 are close but
still creeping up by epoch 20 -- see the measurements at the end of this file.")

(defun train-epochs (learner refine-train train-target refine-test test-target)
  "Train LEARNER for *EPOCHS* epochs, returning the per-epoch test accuracy list.
The caller checks the list is flat at the end -- if accuracy is still climbing, the
epoch count is too low for a fair comparison."
  (loop repeat *epochs*
        do (train-refine-learner learner refine-train train-target)
        collect (test-refine-learner learner refine-test test-target :quiet-p t)))

(defun sweep-lambda1 (forest refine-train refine-test train-target test-target lambda1-list)
  "Train one AROW baseline and one FTRL learner per lambda1, returning a list of plists."
  (cons
   (let* ((learner (make-refine-learner forest))
          (curve (train-epochs learner refine-train train-target refine-test test-target)))
     (list* :learner :arow :lambda1 nil :accuracy-curve curve
            :accuracy (car (last curve))
            (sparsity-report forest learner)))
   (loop for lambda1 in lambda1-list
         collect
         (let* ((learner (make-refine-learner-of-type forest 'clol::sparse-lr+ftrl
                                                      0.1 1.0 lambda1 1.0))
                (curve (train-epochs learner refine-train train-target
                                     refine-test test-target)))
           (list* :learner :ftrl :lambda1 lambda1 :accuracy-curve curve
                  :accuracy (car (last curve))
                  (sparsity-report forest learner))))))

(defun print-sweep (rows)
  "Print ROWS as a markdown table ready to paste into the report."
  (format t "~&| learner | lambda1 | accuracy | element-zero | leaf-zero | leaf-parent-zero |~%")
  (format t "|---|---|---|---|---|---|~%")
  (dolist (row rows)
    (format t "| ~A | ~@[~,1F~] | ~,2F | ~,1F% | ~,1F% | ~,1F% |~%"
            (getf row :learner)
            (getf row :lambda1)
            (getf row :accuracy)
            (* 100 (getf row :element-zero-rate))
            (* 100 (getf row :leaf-zero-rate))
            (* 100 (getf row :leaf-parent-zero-rate))))
  (format t "~&n-leaf ~D, n-leaf-parent ~D, epochs ~D~%"
          (getf (car rows) :n-leaf) (getf (car rows) :n-leaf-parent) *epochs*)
  rows)

(defun run-letter-sweep (&optional (lambda1-list '(0.0 1.0 3.0 10.0 30.0 100.0)))
  "Build the letter forest, sweep lambda1, print the table and return the rows."
  (multiple-value-bind (forest refine-train refine-test train-target test-target)
      (letter-forest)
    (format t "~&forest accuracy: ~,4F~%"
            (test-forest forest
                         (nth-value 0 (cl-random-forest-test/fixture:letter-test))
                         test-target :quiet-p t))
    (values (print-sweep (sweep-lambda1 forest refine-train refine-test
                                        train-target test-target lambda1-list))
            forest refine-train refine-test train-target test-target)))

;;;; Measured on letter, 500 trees, max-depth 15, 20 epochs, 4-worker lparallel kernel:
;;;;
;;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;;   (setf lparallel:*kernel* (lparallel:make-kernel 4))
;;;;   (load "src/experimental/ftrl-pruning-sparsity.lisp")
;;;;   (run-letter-sweep)
;;;;
;;;; forest accuracy: 91.90%
;;;;
;;;; | learner | lambda1 | accuracy | element-zero | leaf-zero | leaf-parent-zero |
;;;; |---|---|---|---|---|---|
;;;; | AROW |       | 97.14 |  6.8% |  0.4% |  0.0% |
;;;; | FTRL | 0.0   | 97.20 |  0.0% |  0.0% |  0.0% |
;;;; | FTRL | 1.0   | 96.84 | 71.6% | 19.9% |  4.9% |
;;;; | FTRL | 3.0   | 96.80 | 88.7% | 41.6% | 18.3% |
;;;; | FTRL | 10.0  | 96.80 | 96.9% | 63.7% | 40.9% |
;;;; | FTRL | 30.0  | 96.28 | 99.0% | 79.9% | 63.4% |
;;;; | FTRL | 100.0 | 95.18 | 99.6% | 90.7% | 81.6% |
;;;; n-leaf 160389, n-leaf-parent 53986, epochs 20
;;;;
;;;; leaf-parent-zero-rate rises monotonically with lambda1 and reaches well past any
;;;; PRUNING-RATE used in the examples (0.1-0.5) by lambda1=10 already -- the element-
;;;; wise sparsity MNIST reports does concentrate into whole-leaf and whole-parent zeros
;;;; here, it is not scattered thinly across (class, leaf) pairs the way the design
;;;; doc's "group sparsity" concern worried it might be. Accuracy stays within about a
;;;; point of the AROW baseline (97.14%) through lambda1=30 (96.28%, -0.86pt, 63.4%
;;;; leaf-parent-zero); lambda1=100 costs close to two points (95.18%, -1.96pt) and is
;;;; the first row where the drop is bigger than run-to-run noise.
;;;;
;;;; Epoch count: the brief's original *EPOCHS* was 10. At 10 epochs AROW and FTRL
;;;; lambda1<=3 had plateaued, but lambda1=30 was still gaining +0.06 to +0.22 accuracy
;;;; on the last epoch and lambda1=100 +0.14 to +0.34 -- per the brief's own Step 4
;;;; instruction ("if still climbing, redo with *EPOCHS* 20") this file now defaults to
;;;; 20. At 20 epochs lambda1<=10 is flat (<0.02pt change epoch 19->20); lambda1=30 is
;;;; nearly flat (+0.06); lambda1=100 is still creeping up (+0.20 epoch 19->20, curve
;;;; below). Higher L1 delays learning because a coordinate's |z| has to cross lambda1
;;;; before the weight moves off zero at all, so very sparse configurations need more
;;;; epochs to reach their converged accuracy. The lambda1=100 accuracy above is
;;;; therefore a slightly pessimistic lower bound; leaf-parent-zero-rate should not move
;;;; much further once the surviving coordinates have crossed the threshold, since that
;;;; is governed by which leaves ever accumulate enough |z|, not by how far past it they
;;;; get.
;;;;
;;;; Accuracy curves (test accuracy %, epoch 1 through 20):
;;;;   AROW    lambda1=nil : 96.94 97.14 97.22 97.24 97.18 97.16 97.14 97.12 97.14 97.14
;;;;                         97.14 97.14 97.14 97.14 97.14 97.14 97.14 97.14 97.14 97.14
;;;;   FTRL    lambda1=0   : 96.24 97.16 97.22 97.20 97.22 97.28 97.26 97.26 97.20 97.18
;;;;                         97.20 97.20 97.20 97.20 97.20 97.20 97.20 97.20 97.18 97.20
;;;;   FTRL    lambda1=1   : 96.62 96.82 96.82 96.84 96.84 96.82 96.78 96.78 96.80 96.82
;;;;                         96.82 96.82 96.84 96.82 96.82 96.84 96.84 96.84 96.84 96.84
;;;;   FTRL    lambda1=3   : 95.50 96.26 96.76 96.76 96.78 96.76 96.76 96.78 96.78 96.78
;;;;                         96.76 96.78 96.76 96.80 96.82 96.82 96.82 96.80 96.80 96.80
;;;;   FTRL    lambda1=10  : 91.66 94.78 95.58 95.94 96.10 96.26 96.38 96.54 96.56 96.60
;;;;                         96.70 96.72 96.78 96.78 96.78 96.80 96.78 96.78 96.80 96.80
;;;;   FTRL    lambda1=30  : 81.40 89.64 92.70 93.90 94.68 94.94 95.22 95.50 95.72 95.82
;;;;                         95.90 95.94 96.04 96.02 96.06 96.12 96.14 96.20 96.22 96.28
;;;;   FTRL    lambda1=100 : 70.84 76.44 81.24 84.46 87.54 89.32 90.94 92.10 92.84 93.30
;;;;                         93.56 93.96 94.28 94.48 94.70 94.74 94.80 94.96 94.98 95.18
;;;;
;;;; MAKE-FOREST's bagging has no fixed RNG seed, so the exact numbers above vary run to
;;;; run by a few tenths of a point: three independent 20-epoch-ish runs gave forest
;;;; accuracy 91.36/91.68/91.42/91.90 and AROW refine accuracy 97.14-97.28. The
;;;; qualitative picture -- monotonic leaf-parent-zero-rate, accuracy holding through
;;;; lambda1=30, lambda1=100 still short of converged -- was stable across all of them.

;;;; Ranking comparison
;;;;
;;;; The interesting question is not only "how many weights are zero" but "does FTRL
;;;; rank leaf-parents differently from AROW at all". AROW is confidence-weighted, so a
;;;; rarely-visited leaf gets a large weight jump from a single update and may be
;;;; protected by the current criterion; FTRL's L1 threshold should drop it instead.

(defun average-ranks (scores)
  "Tie-averaged 1-based ranks of SCORES, as a double-float vector.
Ties must be averaged, not broken arbitrarily: with lambda1 large most leaf-parents
score exactly 0.0, and any tie-breaking would invent an ordering the model never had."
  (let* ((n (length scores))
         (order (let ((v (make-array n)))
                  (dotimes (i n) (setf (aref v i) i))
                  (sort v #'< :key (lambda (i) (aref scores i)))))
         (ranks (make-array n :element-type 'double-float)))
    (let ((i 0))
      (loop while (< i n) do
        (let ((j i))
          (loop while (and (< (1+ j) n)
                           (= (aref scores (aref order (1+ j)))
                              (aref scores (aref order i))))
                do (incf j))
          (let ((r (+ 1.0d0 (/ (+ i j) 2.0d0))))
            (loop for k from i to j
                  do (setf (aref ranks (aref order k)) r)))
          (setf i (1+ j)))))
    ranks))

(defun spearman (scores-a scores-b)
  "Spearman rank correlation of two equal-length score vectors, ties averaged."
  (let* ((ra (average-ranks scores-a))
         (rb (average-ranks scores-b))
         (n (length ra))
         (mean (/ (1+ n) 2.0d0))
         (num 0d0) (var-a 0d0) (var-b 0d0))
    (dotimes (i n)
      (let ((x (- (aref ra i) mean))
            (y (- (aref rb i) mean)))
        (incf num (* x y))
        (incf var-a (* x x))
        (incf var-b (* y y))))
    (if (or (zerop var-a) (zerop var-b))
        0d0
        (/ num (sqrt (* var-a var-b))))))

(defun bottom-k-overlap (scores-a scores-b k)
  "Fraction of the K lowest-scoring indices that SCORES-A and SCORES-B agree on.

Read this with care once FTRL's zero set is larger than K: every zero-scoring
leaf-parent is equally low, so the bottom K under FTRL is an arbitrary subset of that
set and the overlap understates how much the two criteria really agree. Compare K
against LEAF-PARENT-ZERO-RATE * N before drawing a conclusion."
  (flet ((bottom (scores)
           (let ((order (make-array (length scores))))
             (dotimes (i (length scores)) (setf (aref order i) i))
             (subseq (sort order #'< :key (lambda (i) (aref scores i))) 0 k))))
    (let ((in-a (make-hash-table)))
      (map nil (lambda (i) (setf (gethash i in-a) t)) (bottom scores-a))
      (/ (float (count-if (lambda (i) (gethash i in-a)) (bottom scores-b)))
         k))))

(defun compare-rankings (forest arow-learner ftrl-learner)
  "Rank agreement between the two learners' pruning criteria over FOREST."
  (let* ((a (leaf-parent-scores forest arow-learner))
         (b (leaf-parent-scores forest ftrl-learner))
         (n (length a)))
    (list :n-leaf-parent n
          :spearman (spearman a b)
          :bottom-10%-overlap (bottom-k-overlap a b (floor (* n 0.1)))
          :bottom-50%-overlap (bottom-k-overlap a b (floor (* n 0.5)))
          :ftrl-zero-count (count 0.0 b))))

;;;; Pruning end to end
;;;;
;;;; Pruning renumbers the leaf index space (SET-LEAF-INDEX-FOREST!), so both the refine
;;;; dataset and the learner have to be rebuilt before re-training. This is the README
;;;; procedure and T/PRUNING.LISP's PRUNING-PRESERVES-REFINE-ACCURACY does the same.

(defun leaf-count (forest)
  "Total leaves across FOREST. Not FOREST-N-LEAF: that slot goes stale on pruning
\(issue #15)."
  (reduce #'+ (mapcar #'dtree-max-leaf-index (forest-dtree-list forest))))

(defun prune-and-relearn (forest datamatrix datamatrix-test train-target test-target
                          learner rate &optional learner-args)
  "Prune FOREST at RATE using LEARNER's weights, then rebuild and re-train.

LEARNER-ARGS is NIL to rebuild with MAKE-REFINE-LEARNER, or (learner-type . params) to
rebuild with MAKE-REFINE-LEARNER-OF-TYPE -- the rebuilt learner must be the same kind
as the one that chose the pruning, or the before/after accuracies are not comparable.
FOREST is mutated."
  (let ((accuracy-before (test-refine-learner
                          learner (make-refine-dataset forest datamatrix-test)
                          test-target :quiet-p t))
        (leaves-before (leaf-count forest)))
    (pruning! forest learner rate)
    (let* ((refine-train (make-refine-dataset forest datamatrix))
           (refine-test (make-refine-dataset forest datamatrix-test))
           (relearner (if learner-args
                          (apply #'make-refine-learner-of-type forest
                                 (car learner-args) (cdr learner-args))
                          (make-refine-learner forest)))
           (curve (train-epochs relearner refine-train train-target
                                refine-test test-target)))
      (list :rate rate
            :accuracy-before accuracy-before
            :accuracy-after (car (last curve))
            :accuracy-curve curve
            :leaf-count-before leaves-before
            :leaf-count-after (leaf-count forest)))))

(defun run-letter-pruning (lambda1)
  "Compare AROW and FTRL as pruning criteria on letter at LAMBDA1.

Prints the ranking agreement once, then one PRUNE-AND-RELEARN row per (learner, rate).
Every row rebuilds the forest from scratch because PRUNE-AND-RELEARN mutates it: reusing
one forest across rates would prune the 0.5 row on top of the already-pruned 0.1 row."
  (let ((ftrl-args (list 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0))
        (datamatrix (nth-value 0 (cl-random-forest-test/fixture:letter-train)))
        (datamatrix-test (nth-value 0 (cl-random-forest-test/fixture:letter-test))))
    (multiple-value-bind (forest refine-train refine-test train-target test-target)
        (letter-forest)
      (let ((arow (make-refine-learner forest))
            (ftrl (apply #'make-refine-learner-of-type forest ftrl-args)))
        (train-epochs arow refine-train train-target refine-test test-target)
        (train-epochs ftrl refine-train train-target refine-test test-target)
        (format t "~&ranking agreement at lambda1 ~,1F:~%  ~S~%"
                lambda1 (compare-rankings forest arow ftrl))))
    (dolist (rate '(0.1 0.5))
      (dolist (args (list nil ftrl-args))
        (multiple-value-bind (forest refine-train refine-test train-target test-target)
            (letter-forest)
          (let ((learner (if args
                             (apply #'make-refine-learner-of-type forest args)
                             (make-refine-learner forest))))
            (train-epochs learner refine-train train-target refine-test test-target)
            (format t "~&~A rate ~,2F:~%  ~S~%"
                    (if args :ftrl :arow) rate
                    (prune-and-relearn forest datamatrix datamatrix-test
                                       train-target test-target learner rate args))))))))

;;;; Measured on letter, 500 trees, max-depth 15, 20 epochs, 4-worker lparallel kernel,
;;;; lambda1 10.0 (Task 2's "largest lambda1 whose accuracy cost is close to run-to-run
;;;; noise"):
;;;;
;;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;;   (setf lparallel:*kernel* (lparallel:make-kernel 4))
;;;;   (load "src/experimental/ftrl-pruning-sparsity.lisp")
;;;;   (run-letter-pruning 10.0)
;;;;
;;;; ranking agreement at lambda1 10.0:
;;;;   (:N-LEAF-PARENT 54145 :SPEARMAN 0.7392955287108665d0 :BOTTOM-10%-OVERLAP 0.23420762
;;;;    :BOTTOM-50%-OVERLAP 0.78446364 :FTRL-ZERO-COUNT 22046)
;;;;
;;;; FTRL-ZERO-COUNT / N-LEAF-PARENT = 40.7%, matching Task 2's sweep (40.9%) up to
;;;; run-to-run forest variance. That zero count is bigger than both K windows below
;;;; (5414 at 10%, 27072 at 50%), so read BOTTOM-K-OVERLAP with its own caveat in mind:
;;;; the bottom 10% (5414 leaf-parents) is entirely inside FTRL's 22046-wide zero-tied
;;;; block, so FTRL's "bottom 10%" there is an arbitrary subset of ties and the 23.4%
;;;; overlap number is not really measuring rank agreement, just how much of AROW's true
;;;; bottom-10% happens to fall inside FTRL's much larger zero set. The bottom 50%
;;;; (27072) is mostly but not entirely inside the zero set (22046 of 27072, 81.4%), so
;;;; that figure is less arbitrary but still ties-dominated. Only SPEARMAN (0.74) and
;;;; FTRL-ZERO-COUNT are safe to quote without this caveat, and SPEARMAN itself compares
;;;; AROW's tie-free ranking (AROW's leaf-parent-zero-rate measured at 0.0% throughout
;;;; this project) against FTRL's heavily-tied one -- 0.74 says the two criteria broadly
;;;; agree on which leaf-parents are least useful, not that they agree leaf-parent by
;;;; leaf-parent.
;;;;
;;;; Four PRUNE-AND-RELEARN rows, each on its own freshly built forest:
;;;;
;;;; | learner | rate | accuracy before | accuracy after | leaves before | leaves after |
;;;; |---|---|---|---|---|---|
;;;; | AROW | 0.1 | 97.26 | 97.26 | 159621 | 154236 (-3.4%) |
;;;; | FTRL | 0.1 | 97.00 | 97.08 | 159031 | 153662 (-3.4%) |
;;;; | AROW | 0.5 | 97.16 | 97.18 | 159987 | 133003 (-16.9%) |
;;;; | FTRL | 0.5 | 96.58 | 96.68 | 157923 | 131137 (-17.0%) |
;;;;
;;;; Reading these: accuracy does not drop for either learner at either rate -- it is
;;;; flat (AROW 0.1) or ticks up by 0.02-0.10pt after retraining, including at rate 0.5,
;;;; which deletes about a sixth of all leaves. This matches the CVPR2015 global-pruning
;;;; result that a trained forest is redundant enough for both criteria to prune
;;;; substantially and recover full accuracy on retrain; it does not show FTRL winning
;;;; over AROW on accuracy at a matched rate here (FTRL's absolute accuracy is ~0.5-1pt
;;;; below AROW's before and after pruning at both rates, consistent with the accuracy
;;;; cost lambda1=10 already had in Task 2's sweep -- pruning does not add to that gap or
;;;; close it).
;;;;
;;;; The leaves-before/leaves-after counts are near-identical between AROW and FTRL at a
;;;; given rate (154236 vs 153662; 133003 vs 131137) -- that is expected and not a
;;;; finding: PRUNING! deletes FLOOR(N-LEAF-PARENT * RATE) leaf-parents regardless of
;;;; which learner chose them, so the *count* removed is set by RATE and the forest's own
;;;; leaf-parent count (which itself varies build to build), not by which learner ranked
;;;; them. What differs between AROW and FTRL is *which* leaf-parents get removed, which
;;;; SPEARMAN and the overlap numbers above speak to, not the leaf-count columns here.
;;;;
;;;; Net read for the report: FTRL's ranking is correlated with AROW's (spearman 0.74)
;;;; but far from identical, and its L1 zero set is large enough that a big chunk of any
;;;; small-to-medium pruning rate is chosen from ties rather than a strict order. Despite
;;;; that, pruning guided by either criterion is harmless to accuracy on letter at rate
;;;; 0.1 and 0.5 after the standard rebuild-and-retrain step -- this experiment does not
;;;; find a case where FTRL's sparsity makes pruning behave differently in outcome from
;;;; AROW's L2-norm criterion, only that it makes the criterion cheaper to read off
;;;; (FTRL's zeros are already an explicit "prune me" signal; AROW's are not).
;;;;
;;;; Run time: well under a minute end to end (five 500-tree forests plus six 20-epoch
;;;; refine trainings), matching Task 2's observation that letter's bagging-ratio 0.1
;;;; keeps each tree's training set small.

;;;; MNIST fixture
;;;;
;;;; Confirms the letter finding on a second dataset with a different class count (10 vs
;;;; letter's 26) -- the design doc flagged group sparsity (a leaf is only prunable when
;;;; its weight is zero in *every* class) as the thing the whole proposal hinges on, so
;;;; class count is the natural axis to vary.

(defun shift-labels-up! (target)
  "Add 1 to every label in TARGET, in place.
CL-RANDOM-FOREST/SRC/UTILS:READ-DATA subtracts 1 from every LIBSVM label, which is right
for 1-based label files. MNIST's labels already start at 0, so they come back as -1..8
and have to be shifted back. example/classification/mnist.lisp does the same thing."
  (dotimes (i (length target) target)
    (incf (aref target i))))

(defun mnist-forest ()
  "Return (values forest refine-train refine-test train-target test-target) for MNIST.
Forest settings are MNIST-FOREST's from example/classification/mnist.lisp -- the forest
that file's PRUNING! calls operate on (refine 98.259%, 98008 leaf-parents) -- plus
:remove-sample-indices? nil. READ-DATA is imported from CL-RANDOM-FOREST/SRC/UTILS via
this package's DEFPACKAGE, which the :CL-RANDOM-FOREST facade does not re-export."
  (let ((dir cl-random-forest-test/fixture:*dataset-dir*))
    (multiple-value-bind (datamatrix target)
        (read-data (merge-pathnames "mnist.scale" dir) 784)
      (multiple-value-bind (datamatrix-test target-test)
          (read-data (merge-pathnames "mnist.scale.t" dir) 784)
        (shift-labels-up! target)
        (shift-labels-up! target-test)
        (let ((forest (make-forest 10 datamatrix target
                                   :n-tree 500 :bagging-ratio 0.1
                                   :min-region-samples 5 :n-trial 10 :max-depth 10
                                   :remove-sample-indices? nil)))
          (values forest
                  (make-refine-dataset forest datamatrix)
                  (make-refine-dataset forest datamatrix-test)
                  target
                  target-test))))))

;;;; Measured on MNIST, 500 trees, max-depth 10, 20 epochs, 4-worker lparallel kernel.
;;;; Sanity check (its own forest build, separate from the sweep build below --
;;;; MAKE-FOREST has no fixed RNG seed, same build-to-build variance as letter):
;;;;
;;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;;   (setf lparallel:*kernel* (lparallel:make-kernel 4))
;;;;   (load "src/experimental/ftrl-pruning-sparsity.lisp")
;;;;   (in-package :cl-random-forest/src/experimental/ftrl-pruning-sparsity)
;;;;   (multiple-value-bind (forest rd rt tg tgt) (mnist-forest) ...)
;;;;
;;;; forest accuracy: 93.46% (example/classification/mnist.lisp's reference: 93.38%)
;;;; n-leaf-parent: 98760 (reference before PRUNING! there: 98008)
;;;;
;;;; Both numbers are close enough to the example's reference values to trust the labels
;;;; and dimension are right (a doubled or missing SHIFT-LABELS-UP! would have driven
;;;; accuracy far below 93%, not within a tenth of a point of it), and are consistent with
;;;; the few-tenths/few-hundred build-to-build variance already documented for letter.
;;;;
;;;; The sweep itself, sweeping only lambda1 3.0/10.0/30.0 -- the three points that
;;;; bracketed the project's 0.1-0.5 operational PRUNING-RATE range on letter, per this
;;;; task's brief -- launched with the plan's nohup line corrected: QUICKLOAD before
;;;; setting LPARALLEL:*KERNEL*, not after (the plan's original order fails because the
;;;; LPARALLEL package does not exist yet in a fresh image, so the SETF drops into the
;;;; debugger and the run hangs). The 4-worker kernel parallelizes MAKE-FOREST over trees
;;;; and MAKE-REFINE-DATASET/TRAIN-REFINE-LEARNER over independent one-vs-rest classes --
;;;; exact parallelism, not an approximation, so it changes wall-clock time only, not the
;;;; numbers below:
;;;;
;;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;;   (setf lparallel:*kernel* (lparallel:make-kernel 4))
;;;;   (load "src/experimental/ftrl-pruning-sparsity.lisp")
;;;;   (in-package :cl-random-forest/src/experimental/ftrl-pruning-sparsity)
;;;;   (multiple-value-bind (forest rd rt tg tgt) (mnist-forest)
;;;;     (print-sweep (sweep-lambda1 forest rd rt tg tgt (list 3.0 10.0 30.0))))
;;;;
;;;; | learner | lambda1 | accuracy | element-zero | leaf-zero | leaf-parent-zero |
;;;; |---|---|---|---|---|---|
;;;; | AROW |      | 98.27 |  7.2% |  0.8% |  0.0% |
;;;; | FTRL | 3.0  | 98.08 | 87.3% | 52.1% | 25.5% |
;;;; | FTRL | 10.0 | 97.93 | 94.5% | 73.5% | 52.2% |
;;;; | FTRL | 30.0 | 97.76 | 97.4% | 85.3% | 71.0% |
;;;; n-leaf 249251, n-leaf-parent 98283, epochs 20
;;;;
;;;; 10 classes (MNIST) versus 26 (letter), matched lambda1 -- letter numbers copied from
;;;; the block above:
;;;;
;;;; | lambda1 | dataset (classes) | accuracy | delta vs AROW | leaf-zero | leaf-parent-zero |
;;;; |---|---|---|---|---|---|
;;;; | 3  | letter (26) | 96.80 | -0.34 | 41.6% | 18.3% |
;;;; | 3  | MNIST (10)  | 98.08 | -0.19 | 52.1% | 25.5% |
;;;; | 10 | letter (26) | 96.80 | -0.34 | 63.7% | 40.9% |
;;;; | 10 | MNIST (10)  | 97.93 | -0.34 | 73.5% | 52.2% |
;;;; | 30 | letter (26) | 96.28 | -0.86 | 79.9% | 63.4% |
;;;; | 30 | MNIST (10)  | 97.76 | -0.51 | 85.3% | 71.0% |
;;;;
;;;; At every matched lambda1, MNIST's leaf-zero-rate and leaf-parent-zero-rate are both
;;;; higher than letter's, and MNIST's accuracy cost is smaller or equal, never larger.
;;;; Fewer classes made whole-leaf zeros easier to reach, not harder, and cheaper -- the
;;;; opposite of the direction that would have undermined the design doc's group-sparsity
;;;; premise. This holds despite MNIST's element-zero-rate being comparable to, or even
;;;; slightly below, letter's at the same lambda1 (94.5%/97.4% vs letter's 96.9%/99.0% at
;;;; lambda1=10/30) -- the extra leaf- and leaf-parent-level sparsity on MNIST is not
;;;; coming from more sparsity per weight, it is coming from needing fewer classes' weights
;;;; to die together for a whole leaf (or leaf-parent) to zero out.
;;;;
;;;; Independence check (leaf-zero-rate vs element-zero-rate^n-classes), same test as the
;;;; report's "Group sparsity" section computed for letter's 26 classes:
;;;;
;;;; | lambda1 | element-zero | leaf-zero (measured) | independence (elt-zero^10) | measured/pred |
;;;; |---|---|---|---|---|
;;;; | 3  | 87.3% | 52.1% | 25.7% | ~2.03x |
;;;; | 10 | 94.5% | 73.5% | 56.8% | ~1.29x |
;;;; | 30 | 97.4% | 85.3% | 76.9% | ~1.11x |
;;;;
;;;; Measured leaf-zero-rate exceeds the independence prediction at every lambda1 here too
;;;; -- cross-class correlation is real at 10 classes, the same qualitative finding as
;;;; letter's 26-class table (ratios there: ~9.4x/1.4x/1.04x at the same three lambda1).
;;;; The *ratio* is smaller on MNIST purely because the independence baseline itself is
;;;; larger with fewer classes -- raising a fraction below 1 to the 10th power shrinks it
;;;; less than raising it to the 26th -- not because the correlation is weaker: MNIST's
;;;; absolute leaf-zero-rate is higher than letter's at every matched lambda1 despite the
;;;; smaller ratio.
;;;;
;;;; Epoch count: 20 (*EPOCHS*, unchanged from letter). Unlike the letter block above, this
;;;; run's driver called PRINT-SWEEP directly on SWEEP-LAMBDA1's return value without
;;;; printing each row's :ACCURACY-CURVE first, so there is no direct per-epoch evidence
;;;; here that lambda1=30 had flattened out by epoch 20 on MNIST the way it was confirmed
;;;; to for letter. MNIST's accuracy costs at every lambda1 (0.19/0.34/0.51pt) are smaller
;;;; than letter's already-converged-at-20-epochs costs at the same lambda1 (0.34/0.34/
;;;; 0.86pt), and MNIST's training set is 4x larger (60000 vs 15000 rows, so 4x more
;;;; gradient updates per epoch), both of which suggest convergence should be at least as
;;;; fast as letter's here, not slower -- but this was not directly measured and is a
;;;; limitation of this run, not a verified fact.
;;;;
;;;; Run time: both the sanity build and the four-row sweep were launched as their own
;;;; background `ros` processes and both completed, but wall-clock time was not recorded
;;;; precisely; qualitatively substantially longer than letter's "well under a minute", as
;;;; expected going in given MNIST's 4x row count and 49x dimension count.
