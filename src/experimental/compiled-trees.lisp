;;; -*- coding:utf-8; mode:lisp -*-
;;;
;;; Compiling decision trees into Lisp functions.
;;;
;;; A trained tree is a fixed nest of comparisons, so it can be emitted as a nest of IFs
;;; and handed to COMPILE. src/experimental/workspace.lisp carried a sketch of this
;;; (CONSTRUCT-DTREE-LAMBDA) from before the repository moved to single-float: it still
;;; declared its datamatrix DOUBLE-FLOAT, so under (safety 0) it would have read today's
;;; arrays as the wrong type rather than complain. This file is that idea rebuilt against
;;; the current representation, checked for exact agreement with the library's own
;;; predictors, and measured. The sketch has been deleted; what it had measured is
;;; recorded at the bottom of this file.
;;;
;;; Where the speed would come from
;;; -------------------------------
;;; PREDICT-DTREE walks the tree through FIND-LEAF, chasing a pointer per level and
;;; reading NODE-TEST-ATTRIBUTE and NODE-TEST-THRESHOLD out of a struct at each one, then
;;; calls NODE-CLASS-DISTRIBUTION, which recounts the leaf's class histogram from its
;;; SAMPLE-INDICES *on every single prediction* -- leaf values are deliberately not cached
;;; (see CLAUDE.md). Compiling folds all of that away: the attributes and thresholds become
;;; immediates, the traversal becomes straight-line branches, and the leaf value becomes a
;;; constant computed once at compile time.
;;;
;;; What it must not change
;;; -----------------------
;;; A compiled tree has to agree with PREDICT-DTREE on every datum, and a compiled forest
;;; with PREDICT-FOREST. That is not automatic: PREDICT-FOREST sums each tree's *normalised
;;; class distribution* and takes the argmax of the total, so folding each leaf to its own
;;; argmax -- a hard vote -- would be a different classifier that happens to look similar.
;;; The single-tree path folds to a class because for one tree argmax-of-distribution is
;;; the answer; the forest path keeps whole distributions and sums them.
;;;
;;; Not part of any system. Load by hand:
;;;
;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;   (load "src/experimental/compiled-trees.lisp")
;;;   (in-package :cl-random-forest/src/experimental/compiled-trees)
;;;   (run-benchmark)

(defpackage :cl-random-forest/src/experimental/compiled-trees
  (:use #:cl
        #:cl-random-forest)
  (:import-from #:cl-random-forest/src/random-forest
                #:dtree-root
                #:dtree-n-class
                #:node-test-attribute
                #:node-test-threshold
                #:node-left-node
                #:node-right-node
                #:node-class-distribution
                #:node-sample-indices
                #:class-distribution-forest
                #:argmax)
  (:import-from #:cl-random-forest/src/utils
                #:read-data))

(in-package :cl-random-forest/src/experimental/compiled-trees)

;;;; Reading a leaf

(defun leaf-distribution (node)
  "A private copy of NODE's normalised class distribution.

The copy is load-bearing. NODE-CLASS-DISTRIBUTION hands back the dtree's shared
CLASS-COUNT-ARRAY scratch buffer, so the next leaf read overwrites it; storing the buffer
itself would leave every leaf of a tree pointing at whichever one was visited last."
  (copy-seq (node-class-distribution node)))

(defun leaf-class (node)
  "The class PREDICT-DTREE returns at NODE, folded now rather than on every prediction.
ARGMAX is the library's, so ties break the same way -- lowest class index wins."
  (argmax (node-class-distribution node)))

;;;; Code generation

(defun node-form (node leaf-form-fn)
  "The IF-nest for the subtree at NODE, with leaves emitted by LEAF-FORM-FN.

A node with no test attribute is a leaf, which is the same condition FIND-LEAF stops on."
  (if (and node (node-test-attribute node))
      `(if (>= (aref datamatrix datum-index ,(node-test-attribute node))
               ,(node-test-threshold node))
           ,(node-form (node-left-node node) leaf-form-fn)
           ,(node-form (node-right-node node) leaf-form-fn))
      (funcall leaf-form-fn node)))

(defun predictor-declarations ()
  "The DECLARE form every generated predictor opens with.

Returned rather than wrapped around the body: a declaration is only a declaration at the
head of a body, so a macro expanding to (progn (declare ...) ...) produces a call to a
function named DECLARE instead.

The datamatrix is declared rank 2, unlike the library's own (SIMPLE-ARRAY SINGLE-FLOAT),
so the AREF compiles to an index computation rather than a generic one."
  '(declare (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0))
            (type (simple-array single-float (* *)) datamatrix)
            (type fixnum datum-index)))

(defun dtree-class-lambda-form (dtree)
  "A lambda form returning the class PREDICT-DTREE would return for a datum."
  `(lambda (datamatrix datum-index)
     ,(predictor-declarations)
     ,(node-form (dtree-root dtree) #'leaf-class)))

(defun dtree-leaf-lambda-form (dtree)
  "Return (values form distributions).

FORM is a lambda returning an ordinal identifying the leaf a datum reaches, and
DISTRIBUTIONS is a simple-vector of that many class distributions indexed by it. Splitting
it this way is what lets a compiled forest reproduce PREDICT-FOREST: the tree contributes
a whole distribution to the sum, not a vote."
  (let ((distributions '())
        (n 0))
    (flet ((emit-leaf (node)
             (push (leaf-distribution node) distributions)
             (prog1 n (incf n))))
      (let ((form `(lambda (datamatrix datum-index)
                     ,(predictor-declarations)
                     ,(node-form (dtree-root dtree) #'emit-leaf))))
        (values form (coerce (nreverse distributions) 'simple-vector))))))

;;;; Compiling

(defun compile-dtree (dtree)
  "Compile DTREE into a function of (datamatrix datum-index) returning a class."
  (compile nil (dtree-class-lambda-form dtree)))

(defstruct (compiled-forest (:constructor %make-compiled-forest))
  "A forest as one compiled predictor per tree plus its leaves' class distributions.
SCRATCH is the accumulator PREDICT-COMPILED-FOREST sums into, kept here so prediction
allocates nothing -- the same reason FOREST has a CLASS-COUNT-ARRAY."
  n-class n-tree predictors distributions scratch)

(defun compile-forest (forest)
  "Compile every tree of FOREST. Returns a COMPILED-FOREST."
  (let* ((dtrees (forest-dtree-list forest))
         (n-tree (length dtrees))
         (predictors (make-array n-tree))
         (distributions (make-array n-tree)))
    (loop for dtree in dtrees
          for i from 0
          do (multiple-value-bind (form dists) (dtree-leaf-lambda-form dtree)
               (setf (svref predictors i) (compile nil form)
                     (svref distributions i) dists)))
    (%make-compiled-forest
     :n-class (forest-n-class forest)
     :n-tree n-tree
     :predictors predictors
     :distributions distributions
     :scratch (make-array (forest-n-class forest)
                          :element-type 'single-float :initial-element 0.0))))

(defun predict-compiled-forest (compiled-forest datamatrix datum-index
                                &optional (acc (compiled-forest-scratch compiled-forest)))
  "PREDICT-FOREST's answer, computed from precompiled trees and cached leaf distributions.

ACC is the accumulator to sum into. It defaults to a buffer on the struct, which is
convenient and single-threaded only: two threads predicting from one COMPILED-FOREST would
share it and corrupt each other, exactly as PREDICT-FOREST does with
FOREST-CLASS-COUNT-ARRAY. Pass a per-thread accumulator to predict in parallel."
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array single-float (* *)) datamatrix)
           (type fixnum datum-index))
  (let ((n-class (compiled-forest-n-class compiled-forest))
        (n-tree (compiled-forest-n-tree compiled-forest))
        (predictors (compiled-forest-predictors compiled-forest))
        (distributions (compiled-forest-distributions compiled-forest)))
    (declare (type fixnum n-class n-tree)
             (type simple-vector predictors distributions)
             (type (simple-array single-float (*)) acc))
    (dotimes (k n-class) (setf (aref acc k) 0.0))
    (dotimes (tree n-tree)
      (let* ((leaf (the fixnum (funcall (the function (svref predictors tree))
                                        datamatrix datum-index)))
             (dist (svref (the simple-vector (svref distributions tree)) leaf)))
        (declare (type (simple-array single-float (*)) dist))
        (dotimes (k n-class)
          (incf (aref acc k) (aref dist k)))))
    ;; PREDICT-FOREST divides by N-TREE before its argmax. Scaling by a positive constant
    ;; cannot move the argmax, but do it anyway: the point is to reproduce that function,
    ;; and an "equivalent" shortcut is how the two drift apart later.
    (dotimes (k n-class)
      (setf (aref acc k) (/ (aref acc k) n-tree)))
    (argmax acc)))

;;;; Agreement

(defun count-dtree-disagreements (dtree compiled datamatrix)
  "How many rows of DATAMATRIX the compiled tree and PREDICT-DTREE answer differently."
  (let ((n (array-dimension datamatrix 0))
        (bad 0))
    (dotimes (i n bad)
      (unless (= (predict-dtree dtree datamatrix i)
                 (funcall compiled datamatrix i))
        (incf bad)))))

(defun count-forest-disagreements (forest compiled datamatrix)
  "How many rows of DATAMATRIX the compiled forest and PREDICT-FOREST answer differently."
  (let ((n (array-dimension datamatrix 0))
        (bad 0))
    (dotimes (i n bad)
      (unless (= (predict-forest forest datamatrix i)
                 (predict-compiled-forest compiled datamatrix i))
        (incf bad)))))

;;;; Measuring

(defmacro seconds (&body body)
  "Run BODY and return (values result elapsed-seconds)."
  (let ((start (gensym "START"))
        (result (gensym "RESULT")))
    `(let* ((,start (get-internal-real-time))
            (,result (progn ,@body)))
       (values ,result
               (/ (float (- (get-internal-real-time) ,start) 1.0d0)
                  internal-time-units-per-second)))))

(defun time-predictions (fn datamatrix &key (minimum-seconds 0.5d0))
  "Predictions per second for FN over DATAMATRIX.

Runs whole passes until at least MINIMUM-SECONDS have elapsed, and accumulates every
answer into a checksum it returns. Both parts matter. A loop whose body has no effect is
one the compiler may delete, and a fixed small number of passes can finish inside the
clock's resolution -- the first version of this did both and reported 2.5e13 predictions
per second for a depth-5 tree, a number produced entirely by dividing 25000 by zero
seconds.

Returns (values rate checksum passes)."
  (let ((n (array-dimension datamatrix 0))
        (checksum 0)
        (passes 0)
        (start (get-internal-real-time))
        (elapsed 0d0))
    (declare (type fixnum n checksum passes))
    (loop
      (dotimes (i n)
        (incf checksum (the fixnum (funcall fn datamatrix i))))
      (incf passes)
      (setf elapsed (/ (float (- (get-internal-real-time) start) 1.0d0)
                       internal-time-units-per-second))
      (when (>= elapsed minimum-seconds)
        (return)))
    (values (/ (* n passes) elapsed) checksum passes)))

(defun count-leaves (dtree)
  (let ((n 0))
    (labels ((walk (node)
               (cond ((null node) nil)
                     ((node-test-attribute node)
                      (walk (node-left-node node))
                      (walk (node-right-node node)))
                     (t (incf n)))))
      (walk (dtree-root dtree)))
    n))

(defun benchmark-dtree (n-class datamatrix target datamatrix-test max-depth n-trial)
  "Build one tree at MAX-DEPTH, compile it, check agreement, and time both predictors."
  (let ((dtree (make-dtree n-class datamatrix target
                           :max-depth max-depth :n-trial n-trial :min-region-samples 5)))
    (multiple-value-bind (compiled compile-seconds) (seconds (compile-dtree dtree))
      (let ((disagreements (count-dtree-disagreements dtree compiled datamatrix-test))
            (walk-rate (time-predictions (lambda (d i) (predict-dtree dtree d i))
                                         datamatrix-test))
            (compiled-rate (time-predictions compiled datamatrix-test)))
        (format t "~&| tree d=~D | ~D | ~,2F | ~D | ~,0F | ~,0F | ~,1Fx |~%"
                max-depth (count-leaves dtree) compile-seconds disagreements
                walk-rate compiled-rate (/ compiled-rate walk-rate))
        (force-output)))))

(defun benchmark-forest (n-class datamatrix target datamatrix-test n-tree max-depth n-trial)
  "Build a forest, compile every tree, check agreement, and time both predictors."
  (let ((forest (make-forest n-class datamatrix target
                             :n-tree n-tree :bagging-ratio 0.1
                             :max-depth max-depth :n-trial n-trial :min-region-samples 5)))
    (multiple-value-bind (compiled compile-seconds) (seconds (compile-forest forest))
      (let ((disagreements (count-forest-disagreements forest compiled datamatrix-test))
            (leaves (reduce #'+ (mapcar #'count-leaves (forest-dtree-list forest))))
            (walk-rate (time-predictions (lambda (d i) (predict-forest forest d i))
                                         datamatrix-test))
            (compiled-rate (time-predictions
                            (lambda (d i) (predict-compiled-forest compiled d i))
                            datamatrix-test)))
        (format t "~&| forest ~Dx d=~D | ~D | ~,2F | ~D | ~,0F | ~,0F | ~,1Fx |~%"
                n-tree max-depth leaves compile-seconds disagreements
                walk-rate compiled-rate (/ compiled-rate walk-rate))
        (force-output)))))

(defvar *mnist-cache* nil)

(defun mnist-data ()
  "Return (values datamatrix datamatrix-test target target-test) for MNIST, read once.

READ-DATA subtracts 1 from every LIBSVM label, which suits 1-based label files; MNIST's
already start at 0, so they arrive as -1..8 and have to be shifted back. This is what
example/classification/mnist.lisp does inline."
  (unless *mnist-cache*
    (let ((dir cl-random-forest-test/fixture:*dataset-dir*))
      (multiple-value-bind (datamatrix target)
          (read-data (merge-pathnames "mnist.scale" dir) 784)
        (multiple-value-bind (datamatrix-test target-test)
            (read-data (merge-pathnames "mnist.scale.t" dir) 784)
          (dotimes (i (length target)) (incf (aref target i)))
          (dotimes (i (length target-test)) (incf (aref target-test i)))
          (setf *mnist-cache*
                (list datamatrix datamatrix-test target target-test))))))
  (values-list *mnist-cache*))

(defun dataset-parts (dataset)
  "Return (values datamatrix datamatrix-test target n-class n-trial).

N-TRIAL follows what example/classification/ uses for each set -- 10 for letter, 28 for
MNIST -- because it decides how hard each split is searched and so how the trees are
shaped, which is the thing being compiled."
  (ecase dataset
    (:letter
     (multiple-value-bind (datamatrix target) (cl-random-forest-test/fixture:letter-train)
       (multiple-value-bind (datamatrix-test target-test)
           (cl-random-forest-test/fixture:letter-test)
         (declare (ignore target-test))
         (values datamatrix datamatrix-test target
                 cl-random-forest-test/fixture:+letter-n-class+ 10))))
    (:mnist
     (multiple-value-bind (datamatrix datamatrix-test target target-test) (mnist-data)
       (declare (ignore target-test))
       (values datamatrix datamatrix-test target 10 28)))))

(defun run-benchmark (&key (dataset :letter)
                           (depths '(5 10 15))
                           (forest-configs '((100 5) (100 10) (500 5) (500 10))))
  "Compile trees and forests of several shapes and report agreement and speed."
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (format t "~&~A: ~D train, ~D test, ~D features, ~D classes, n-trial ~D~%"
            dataset (array-dimension datamatrix 0) (array-dimension datamatrix-test 0)
            (array-dimension datamatrix 1) n-class n-trial)
    (format t "~&| model | leaves | compile s | disagreements | walk pred/s | compiled pred/s | speedup |~%")
    (format t "|---|---|---|---|---|---|---|~%")
    (force-output)
    (dolist (depth depths)
      (benchmark-dtree n-class datamatrix target datamatrix-test depth n-trial))
    (dolist (config forest-configs)
      (benchmark-forest n-class datamatrix target datamatrix-test
                        (first config) (second config) n-trial))
    (format t "~&BENCHMARK_DONE~%")
    (force-output)))

;;;; Why a forest gains less than a tree
;;;;
;;;; "A forest is the tree's work repeated N times, so N times a K-times-faster tree should
;;;; be K times faster" is the obvious expectation, and the tables above do not meet it.
;;;; Two things break the arithmetic, and they pull in the same direction.
;;;;
;;;; The first is that the standalone tree and the forest's trees are not the same trees.
;;;; BENCHMARK-DTREE trains on the whole set; MAKE-FOREST at :bagging-ratio 0.1 gives each
;;;; tree a tenth of it. PREDICT-DTREE's cost is dominated by recounting a leaf's histogram
;;;; from its SAMPLE-INDICES, which scales with how many samples landed there -- so the
;;;; standalone tree's *baseline* is inflated by roughly the bagging ratio, and the speedup
;;;; measured against it is inflated with it.
;;;;
;;;; The second is that a compiled forest is not merely N compiled trees. It still sums
;;;; N-TREE distributions of N-CLASS floats and takes an argmax, work no compiled tree
;;;; does and compiling cannot remove. That is the Amdahl residue.
;;;;
;;;; DECOMPOSE-FOREST measures both instead of arguing about them.

(defun time-traversals-only (compiled-forest datamatrix)
  "Rate at which the compiled forest reaches every leaf without aggregating anything.

Calls each tree's predictor and sums the leaf ordinals. The gap between this and
PREDICT-COMPILED-FOREST is what the distribution summing costs."
  (let ((predictors (compiled-forest-predictors compiled-forest))
        (n-tree (compiled-forest-n-tree compiled-forest)))
    (time-predictions
     (lambda (d i)
       (let ((sum 0))
         (declare (type fixnum sum))
         (dotimes (tree n-tree sum)
           (incf sum (the fixnum (funcall (the function (svref predictors tree)) d i))))))
     datamatrix)))

(defun decompose-forest (n-class datamatrix target datamatrix-test n-tree max-depth n-trial)
  "Break a forest's speedup into per-tree cost and aggregation cost.

Reports one of the forest's own trees measured alone -- same bagging, same depth -- so the
per-tree speedup can be read without the standalone tree's inflated baseline, and then how
much of the compiled forest's time is spent anywhere other than in those trees."
  (let* ((forest (make-forest n-class datamatrix target
                              :n-tree n-tree :bagging-ratio 0.1
                              :max-depth max-depth :n-trial n-trial :min-region-samples 5))
         (compiled (compile-forest forest))
         (dtree (first (forest-dtree-list forest)))
         (compiled-tree (compile-dtree dtree))
         (tree-walk-rate (time-predictions (lambda (d i) (predict-dtree dtree d i))
                                           datamatrix-test))
         (tree-compiled-rate (time-predictions compiled-tree datamatrix-test))
         (forest-walk-rate (time-predictions (lambda (d i) (predict-forest forest d i))
                                             datamatrix-test))
         (forest-compiled-rate (time-predictions
                                (lambda (d i) (predict-compiled-forest compiled d i))
                                datamatrix-test))
         (traversal-rate (time-traversals-only compiled datamatrix-test))
         ;; Seconds per prediction, so the parts can be added up.
         (tree-compiled-us (/ 1d6 tree-compiled-rate))
         (forest-compiled-us (/ 1d6 forest-compiled-rate))
         (traversal-us (/ 1d6 traversal-rate)))
    (format t "~&~%--- forest ~Dx d=~D, one of its trees measured alone ---~%" n-tree max-depth)
    (format t "~&per-tree   walk ~,0F/s  compiled ~,0F/s  speedup ~,1Fx~%"
            tree-walk-rate tree-compiled-rate (/ tree-compiled-rate tree-walk-rate))
    (format t "~&forest     walk ~,0F/s  compiled ~,0F/s  speedup ~,1Fx~%"
            forest-walk-rate forest-compiled-rate (/ forest-compiled-rate forest-walk-rate))
    (format t "~&compiled forest budget per prediction: ~,1F us total~%" forest-compiled-us)
    (format t "~&  ~,1F us  traversals only (~,0F%)~%"
            traversal-us (* 100 (/ traversal-us forest-compiled-us)))
    (format t "~&  ~,1F us  aggregation: ~D x ~D adds, divide, argmax (~,0F%)~%"
            (- forest-compiled-us traversal-us) n-tree n-class
            (* 100 (/ (- forest-compiled-us traversal-us) forest-compiled-us)))
    (format t "~&  ~,1F us  one compiled tree x ~D, for reference~%"
            (* tree-compiled-us n-tree) n-tree)
    (force-output)))

(defun run-decomposition (&key (dataset :letter) (configs '((500 5) (500 10))))
  "Run DECOMPOSE-FOREST over several forest shapes."
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (format t "~&=== decomposition on ~A ===~%" dataset)
    (force-output)
    (dolist (config configs)
      (decompose-forest n-class datamatrix target datamatrix-test
                        (first config) (second config) n-trial))
    (format t "~&DECOMPOSITION_DONE~%")
    (force-output)))

;;;; Decomposition, measured
;;;;
;;;; RUN-DECOMPOSITION, 500-tree forests, one run each. "per-tree" is one of the forest's
;;;; own trees timed alone -- same bagging, same depth -- so it can be compared with the
;;;; standalone tree without the difference in training data.
;;;;
;;;; | | per-tree walk | per-tree speedup | forest speedup |
;;;; |---|---|---|---|
;;;; | letter d=5  |  5719943/s | 52.6x | 6.4x |
;;;; | letter d=10 | 10559873/s | 19.9x | 3.9x |
;;;; | MNIST d=5   |  1752967/s | 90.9x | 20.0x |
;;;; | MNIST d=10  |  5659921/s | 22.3x | 4.3x |
;;;;
;;;; Compiled forest time per prediction, split into the traversals and everything else:
;;;;
;;;; | | total | traversals | aggregation | 500 x one tree, alone |
;;;; |---|---|---|---|---|
;;;; | letter d=5  | 18.3 us | 11.2 us (61%) |  7.1 us (39%) | 1.7 us |
;;;; | letter d=10 | 58.4 us | 39.9 us (68%) | 18.5 us (32%) | 2.4 us |
;;;; | MNIST d=5   | 15.6 us | 13.4 us (86%) |  2.1 us (14%) | 3.1 us |
;;;; | MNIST d=10  | 80.1 us | 52.3 us (65%) | 27.8 us (35%) | 4.0 us |
;;;;
;;;; Three things, not two.
;;;;
;;;; 1. The standalone tree's baseline is inflated by bagging. On MNIST a forest tree walks
;;;;    at 1.75M predictions per second against the standalone tree's 184k -- 9.5x, which
;;;;    is :bagging-ratio 0.1 showing up exactly where the histogram recount predicts it.
;;;;    Measured per-tree rather than against that inflated baseline, the speedup is 90.9x,
;;;;    not 1052x. Most of the headline single-tree number is this.
;;;;
;;;; 2. Five hundred distinct compiled functions are far slower than one compiled function
;;;;    called five hundred times. This is the effect not anticipated above, and it is the
;;;;    larger one: on MNIST at depth 5 the traversals cost 13.4 us where 500 times a
;;;;    single tree's own measured time is 3.1 us, a 4.3x penalty; at depth 10 it is 52.3
;;;;    against 4.0, 13x. A tree timed alone runs with its code hot in the instruction
;;;;    cache; in a forest each of 500 code blobs is touched once per prediction, and the
;;;;    penalty grows with total code size -- 15991 leaves at depth 5 against 259736 at
;;;;    depth 10. The walking forest does not pay this, because it is one small FIND-LEAF
;;;;    loop over different data rather than 500 different pieces of code.
;;;;
;;;; 3. Aggregation, which compiling cannot touch: 14-39% of the compiled forest's time,
;;;;    and worse where there are more classes to sum (letter's 26 against MNIST's 10).
;;;;
;;;; So the expectation that a forest inherits the tree's speedup fails on all three
;;;; counts, and the one that would be worth attacking is the second. Emitting all trees
;;;; into one function, or laying the thresholds out as data walked by one compact
;;;; interpreter loop, would trade branch-prediction for locality. Not measured here.

;;;; Measured on letter (15000 train, 5000 test, 26 classes), x86-64 SBCL, one run each.
;;;; Rates are predictions per second over the test set, timed for at least half a second.
;;;;
;;;; | model | leaves | compile s | disagreements | walk pred/s | compiled pred/s | speedup |
;;;; |---|---|---|---|---|---|---|
;;;; | tree d=5        |     29 |  0.01 | 0 |   669992 | 312156254 | 465.9x |
;;;; | tree d=10       |    476 |  0.10 | 0 |  3529958 | 207557094 |  58.8x |
;;;; | tree d=15       |   1074 |  0.26 | 0 |  4769943 | 165048019 |  34.6x |
;;;; | forest 100x d=5 |   3038 |  0.39 | 0 |    53475 |    414195 |   7.7x |
;;;; | forest 100x d=10|  22772 |  3.48 | 0 |    50590 |    154438 |   3.1x |
;;;; | forest 500x d=5 |  15128 |  1.57 | 0 |     7225 |     54644 |   7.6x |
;;;; | forest 500x d=10| 115008 | 16.58 | 0 |     3302 |     13587 |   4.1x |
;;;;
;;;; Agreement is exact everywhere, which is the result that makes the rest worth reading:
;;;; the compiled models are the same classifiers, not approximations of them.
;;;;
;;;; The single-tree speedups are large and mostly not about compilation. PREDICT-DTREE
;;;; recounts the leaf's class histogram from its SAMPLE-INDICES on every call, so its cost
;;;; scales with how many training samples landed in that leaf. That is why the shallow
;;;; tree is the *slowest* per prediction of the three (670k/s at depth 5 against 4.8M/s at
;;;; depth 15): fewer, larger leaves mean more counting. The compiled tree does none of it
;;;; -- the leaf value is a constant folded at compile time -- so its rate moves the other
;;;; way, falling with depth as the branch nest gets longer. Most of the 466x at depth 5 is
;;;; the histogram disappearing, not the traversal.
;;;;
;;;; Forests gain much less, 3-8x, and the reason is visible in the same numbers. A forest
;;;; prediction still has to sum 500 distributions of 26 floats and take an argmax, and
;;;; that arithmetic is untouched by compiling the traversals. Depth hurts the compiled
;;;; forest more than the walking one (7.6x down to 4.1x) because 115008 leaves' worth of
;;;; branch code does not fit in cache the way a few structs being chased do.
;;;;
;;;; Compile time is the cost to weigh: 16.6 seconds for 500 trees at depth 10, against
;;;; well under a second to build the forest itself. It scales with total leaves at roughly
;;;; 7 microseconds per leaf across every row here. That is fine for a model trained once
;;;; and served many times, and absurd for one used a handful of times.
;;;;
;;;; Not measured: memory. 115008 leaves of generated code is a lot of instructions, and
;;;; nothing here reports how much.

;;;; Measured on MNIST (60000 train, 10000 test, 784 features, 10 classes, n-trial 28),
;;;; same machine, one run each.
;;;;
;;;; | model | leaves | compile s | disagreements | walk pred/s | compiled pred/s | speedup |
;;;; |---|---|---|---|---|---|---|
;;;; | tree d=5        |     32 |  0.01 | 0 |   184160 | 193837286 | 1052.6x |
;;;; | tree d=10       |    887 |  0.14 | 0 |   686264 |  41139506 |   59.9x |
;;;; | tree d=15       |   4982 |  2.03 | 0 |   889318 |  33499464 |   37.7x |
;;;; | forest 100x d=5 |   3197 |  0.40 | 0 |    18832 |    475242 |   25.2x |
;;;; | forest 100x d=10|  52430 | 10.67 | 0 |    21209 |    144402 |    6.8x |
;;;; | forest 500x d=5 |  15991 |  1.95 | 0 |     2825 |     63795 |   22.6x |
;;;; | forest 500x d=10| 259736 | 47.55 | 0 |     2307 |     10672 |    4.6x |
;;;;
;;;; Agreement is again exact everywhere.
;;;;
;;;; The shape is letter's, amplified. The leaf-histogram effect that dominates the
;;;; single-tree numbers is stronger here because MNIST has four times the training data:
;;;; a depth-5 tree puts 60000 samples into 32 leaves, so PREDICT-DTREE recounts thousands
;;;; of labels per call and manages only 184k predictions per second -- slower than
;;;; letter's already-slow 670k, and a quarter of what the same code does on MNIST at
;;;; depth 15. Compiling removes that work entirely, which is where 1052x comes from. It
;;;; is a statement about how much the library recomputes at shallow depth, not about how
;;;; fast compiled branches are.
;;;;
;;;; Forests separate more sharply than on letter. At depth 5 the gain is 22-25x against
;;;; letter's 7.6x, for the same reason -- more samples per leaf to recount. At depth 10 it
;;;; collapses to 4.6x: 259736 leaves of generated code, and the summing of 500
;;;; ten-element distributions that compiling does not touch.
;;;;
;;;; Compile time tracks total leaves at about 180 microseconds per thousand leaves here,
;;;; against letter's roughly 140 -- close enough that leaf count, not feature count or
;;;; dataset size, is what predicts it. The 500-tree depth-10 forest costs 47.6 seconds to
;;;; compile, which is the number to weigh against a use that gets 4.6x back.

;;;; What the original sketch had measured
;;;;
;;;; src/experimental/workspace.lisp did carry timings, on MNIST with 500 trees at
;;;; max-depth 5 and n-trial 28, before it was deleted in favour of this file:
;;;;
;;;;   compiled predictors  0.286 s over the 10000-row test set, 9385 correct
;;;;   TEST-FOREST          2.659 s over the same set,           9433 correct (94.33%)
;;;;
;;;; About 9x, in the same range as the 7.6x measured here for a 500-tree depth-5 forest.
;;;;
;;;; The accuracy column is the part worth keeping. Those two numbers are not the same
;;;; classifier: the sketch's PREDICT-DTREE-PREDICTOR-LIST counted one vote per tree and
;;;; took the argmax of the votes, where PREDICT-FOREST sums the trees' normalised class
;;;; distributions. 48 of 10000 predictions differ because of it, and nothing in the file
;;;; remarked on the gap. That is exactly the substitution the header of this file warns
;;;; about, and why COUNT-FOREST-DISAGREEMENTS exists and has to read zero.
;;;;
;;;; (The sketch also sized its vote counter by (array-dimension datamatrix 1) -- the
;;;; feature count, 784 -- rather than the class count, which is harmless only because 784
;;;; happens to exceed 10.)

;;;; A third representation: the tree as data
;;;;
;;;; The decomposition above says the compiled forest's problem is that 500 separate code
;;;; blobs are touched once each per prediction. The way out is to stop generating code and
;;;; start generating data: lay every node of every tree into flat arrays and walk them with
;;;; one small loop. The loop stays in the instruction cache no matter how large the forest
;;;; gets, and the nodes become contiguous instead of a pointer chase through separate heap
;;;; objects.
;;;;
;;;; It keeps the thing that made compiling fast in the first place -- the leaf value is
;;;; computed once at build time instead of recounted from SAMPLE-INDICES on every
;;;; prediction -- and gives up the thing that made it fast at the margin, thresholds as
;;;; immediates in straight-line branches.
;;;;
;;;; The node struct here is a DEFSTRUCT, not a CLOS class, so there is no dispatch to
;;;; remove; the gain is layout, not dispatch.
;;;;
;;;; All the arrays are read-only after building, and prediction writes only the
;;;; accumulator the caller supplies. That makes this the only one of the three
;;;; representations that is reentrant, which the parallel section below leans on.

(defstruct (array-forest (:constructor %make-array-forest))
  "Every node of every tree of a forest, flattened into parallel arrays.

FEATURE, THRESHOLD, LEFT, RIGHT and LEAF-P are indexed by a global node number; ROOTS holds
each tree's. LEFT does double duty: on an internal node it is the child to take when the
test passes, and on a leaf it is the row of DISTRIBUTIONS holding that leaf's class
distribution. Both branches of the walk therefore read the same array, and a leaf needs no
array of its own.

The slots carry their types. Without them the accessors return T and every caller has to
re-declare what it reads or lose the optimisation -- which is exactly what happened: the
ROOTS read in PREDICT-ARRAY-TREE was the one place a declaration was missing, and SBCL
fell back to a runtime dispatch on the array's element type there."
  (n-class 0 :type fixnum)
  (n-tree 0 :type fixnum)
  (n-node 0 :type fixnum)
  (feature (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (threshold (make-array 0 :element-type 'single-float)
             :type (simple-array single-float (*)))
  (left (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (right (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (leaf-p (make-array 0 :element-type 'bit) :type simple-bit-vector)
  (roots (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (distributions (make-array '(0 0) :element-type 'single-float)
                 :type (simple-array single-float (* *))))

(defun build-array-forest (forest &key (leaf-payload :distribution))
  "Flatten FOREST into arrays.

LEAF-PAYLOAD :distribution stores each leaf's class distribution, which is what a forest
has to sum. :class stores the leaf's argmax instead, which is what PREDICT-DTREE returns
and what a single tree can therefore answer without any summing."
  (let ((feature (make-array 64 :element-type 'fixnum :adjustable t :fill-pointer 0))
        (threshold (make-array 64 :element-type 'single-float :adjustable t :fill-pointer 0))
        (left (make-array 64 :element-type 'fixnum :adjustable t :fill-pointer 0))
        (right (make-array 64 :element-type 'fixnum :adjustable t :fill-pointer 0))
        (leaf-p (make-array 64 :element-type 'bit :adjustable t :fill-pointer 0))
        (distributions '())
        (n-leaf 0)
        (roots '())
        (n-class (forest-n-class forest)))
    (labels ((emit (node)
               ;; Reserve this node's slot before recursing, so a child's index is always
               ;; greater than its parent's and the arrays can be filled in one pass.
               (let ((self (fill-pointer feature)))
                 (vector-push-extend 0 feature)
                 (vector-push-extend 0.0 threshold)
                 (vector-push-extend 0 left)
                 (vector-push-extend 0 right)
                 (vector-push-extend 0 leaf-p)
                 (if (node-test-attribute node)
                     (let ((l (emit (node-left-node node)))
                           (r (emit (node-right-node node))))
                       (setf (aref feature self) (node-test-attribute node)
                             (aref threshold self) (node-test-threshold node)
                             (aref left self) l
                             (aref right self) r))
                     (progn
                       (setf (aref leaf-p self) 1
                             (aref left self) (ecase leaf-payload
                                                (:distribution n-leaf)
                                                (:class (leaf-class node))))
                       (when (eq leaf-payload :distribution)
                         (push (leaf-distribution node) distributions))
                       (incf n-leaf)))
                 self)))
      (dolist (dtree (forest-dtree-list forest))
        (push (emit (dtree-root dtree)) roots)))
    (let ((n-node (fill-pointer feature))
          (table (make-array (list (max n-leaf 1) n-class)
                             :element-type 'single-float :initial-element 0.0)))
      (when (eq leaf-payload :distribution)
        (loop for dist in (nreverse distributions)
              for row from 0
              do (dotimes (k n-class)
                   (setf (aref table row k) (aref dist k)))))
      (%make-array-forest
       :n-class n-class
       :n-tree (forest-n-tree forest)
       :n-node n-node
       :feature (coerce feature '(simple-array fixnum (*)))
       :threshold (coerce threshold '(simple-array single-float (*)))
       :left (coerce left '(simple-array fixnum (*)))
       :right (coerce right '(simple-array fixnum (*)))
       :leaf-p (coerce leaf-p 'simple-bit-vector)
       :roots (coerce (nreverse roots) '(simple-array fixnum (*)))
       :distributions table))))

(defun make-accumulator (array-forest)
  "A fresh accumulator for PREDICT-ARRAY-FOREST. One per thread."
  (make-array (array-forest-n-class array-forest)
              :element-type 'single-float :initial-element 0.0))

(defun predict-array-forest (array-forest datamatrix datum-index acc)
  "PREDICT-FOREST's answer, walking the flattened arrays. Writes only ACC."
  (declare (optimize (speed 3) (safety 0))
           (type array-forest array-forest)
           (type (simple-array single-float (* *)) datamatrix)
           (type (simple-array single-float (*)) acc)
           (type fixnum datum-index))
  (let ((feature (array-forest-feature array-forest))
        (threshold (array-forest-threshold array-forest))
        (left (array-forest-left array-forest))
        (right (array-forest-right array-forest))
        (leaf-p (array-forest-leaf-p array-forest))
        (roots (array-forest-roots array-forest))
        (table (array-forest-distributions array-forest))
        (n-class (array-forest-n-class array-forest))
        (n-tree (array-forest-n-tree array-forest)))
    (declare (type (simple-array fixnum (*)) feature left right roots)
             (type (simple-array single-float (*)) threshold)
             (type simple-bit-vector leaf-p)
             (type (simple-array single-float (* *)) table)
             (type fixnum n-class n-tree))
    (dotimes (k n-class) (setf (aref acc k) 0.0))
    (dotimes (tree n-tree)
      (let ((node (aref roots tree)))
        (declare (type fixnum node))
        (loop
          (when (= 1 (sbit leaf-p node))
            (let ((row (aref left node)))
              (declare (type fixnum row))
              (dotimes (k n-class)
                (incf (aref acc k) (aref table row k))))
            (return))
          (setf node (if (>= (aref datamatrix datum-index (aref feature node))
                             (aref threshold node))
                         (aref left node)
                         (aref right node))))))
    (dotimes (k n-class)
      (setf (aref acc k) (/ (aref acc k) n-tree)))
    (argmax acc)))

(defun predict-array-tree (array-forest datamatrix datum-index)
  "The class a :class-payload single-tree ARRAY-FOREST gives, with no summing at all."
  (declare (optimize (speed 3) (safety 0))
           (type array-forest array-forest)
           (type (simple-array single-float (* *)) datamatrix)
           (type fixnum datum-index))
  (let ((feature (array-forest-feature array-forest))
        (threshold (array-forest-threshold array-forest))
        (left (array-forest-left array-forest))
        (right (array-forest-right array-forest))
        (leaf-p (array-forest-leaf-p array-forest))
        (node (aref (array-forest-roots array-forest) 0)))
    (declare (type (simple-array fixnum (*)) feature left right)
             (type (simple-array single-float (*)) threshold)
             (type simple-bit-vector leaf-p)
             (type fixnum node))
    (loop
      (when (= 1 (sbit leaf-p node))
        (return (aref left node)))
      (setf node (if (>= (aref datamatrix datum-index (aref feature node))
                         (aref threshold node))
                     (aref left node)
                     (aref right node))))))

(defun count-array-forest-disagreements (forest array-forest datamatrix)
  "How many rows PREDICT-FOREST and PREDICT-ARRAY-FOREST answer differently."
  (let ((acc (make-accumulator array-forest))
        (bad 0))
    (dotimes (i (array-dimension datamatrix 0) bad)
      (unless (= (predict-forest forest datamatrix i)
                 (predict-array-forest array-forest datamatrix i acc))
        (incf bad)))))

;;;; Three representations, side by side

(defun benchmark-three-ways (n-class datamatrix target datamatrix-test n-tree max-depth
                             n-trial)
  "Build a forest once and measure walking, compiled and array prediction on it."
  (let ((forest (make-forest n-class datamatrix target
                             :n-tree n-tree :bagging-ratio 0.1
                             :max-depth max-depth :n-trial n-trial :min-region-samples 5)))
    (multiple-value-bind (compiled compile-seconds) (seconds (compile-forest forest))
      (multiple-value-bind (arrayed build-seconds) (seconds (build-array-forest forest))
        (let ((acc (make-accumulator arrayed)))
          (format t "~&| forest ~Dx d=~D | ~D nodes | ~,2F s compile | ~,3F s build | ~
~D / ~D disagreements | ~,0F walk | ~,0F compiled | ~,0F array |~%"
                  n-tree max-depth (array-forest-n-node arrayed)
                  compile-seconds build-seconds
                  (count-forest-disagreements forest compiled datamatrix-test)
                  (count-array-forest-disagreements forest arrayed datamatrix-test)
                  (time-predictions (lambda (d i) (predict-forest forest d i))
                                    datamatrix-test)
                  (time-predictions (lambda (d i) (predict-compiled-forest compiled d i))
                                    datamatrix-test)
                  (time-predictions (lambda (d i) (predict-array-forest arrayed d i acc))
                                    datamatrix-test))
          (force-output)
          (values forest compiled arrayed))))))

(defun run-three-ways (&key (dataset :letter) (configs '((500 5) (500 10))))
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (format t "~&=== three representations on ~A ===~%" dataset)
    (force-output)
    (dolist (config configs)
      (benchmark-three-ways n-class datamatrix target datamatrix-test
                            (first config) (second config) n-trial))
    (format t "~&THREE_WAYS_DONE~%")
    (force-output)))

;;;; Parallel prediction
;;;;
;;;; PREDICT-FOREST accumulates into FOREST-CLASS-COUNT-ARRAY and every leaf read
;;;; overwrites DTREE-CLASS-COUNT-ARRAY, both of them slots on the shared model, so two
;;;; threads predicting from one forest write over each other. That is not a race that
;;;; merely slows things down; it changes answers. COUNT-PARALLEL-CORRUPTIONS measures how
;;;; many, rather than asserting it.

(defun chunk-bounds (n-rows n-workers)
  "N-WORKERS (start . end) pairs covering N-ROWS."
  (loop for w from 0 below n-workers
        collect (cons (floor (* w n-rows) n-workers)
                      (floor (* (1+ w) n-rows) n-workers))))

(defun time-parallel (chunk-fn n-rows n-workers &key (minimum-seconds 0.5d0))
  "Predictions per second when CHUNK-FN is run over N-WORKERS chunks in parallel.

CHUNK-FN takes (start end) and returns a checksum. A kernel of N-WORKERS is created and
shut down around the measurement."
  (let ((bounds (chunk-bounds n-rows n-workers))
        (lparallel:*kernel* (lparallel:make-kernel n-workers)))
    (unwind-protect
         (let ((passes 0)
               (checksum 0)
               (start (get-internal-real-time))
               (elapsed 0d0))
           (loop
             (incf checksum
                   (reduce #'+ (lparallel:pmapcar
                                (lambda (b) (funcall chunk-fn (car b) (cdr b)))
                                bounds)))
             (incf passes)
             (setf elapsed (/ (float (- (get-internal-real-time) start) 1.0d0)
                              internal-time-units-per-second))
             (when (>= elapsed minimum-seconds) (return)))
           (values (/ (* n-rows passes) elapsed) checksum))
      (lparallel:end-kernel :wait t))))

(defun count-parallel-corruptions (forest datamatrix n-workers)
  "How many answers PREDICT-FOREST gets wrong when N-WORKERS threads share one forest.

Serial answers first, then the same rows through a kernel, then compare."
  (let* ((n (array-dimension datamatrix 0))
         (serial (make-array n :element-type 'fixnum)))
    (dotimes (i n) (setf (aref serial i) (predict-forest forest datamatrix i)))
    (let ((parallel (make-array n :element-type 'fixnum :initial-element -1))
          (lparallel:*kernel* (lparallel:make-kernel n-workers)))
      (unwind-protect
           (lparallel:pmapc
            (lambda (b)
              (loop for i from (car b) below (cdr b)
                    do (setf (aref parallel i) (predict-forest forest datamatrix i))))
            (chunk-bounds n n-workers))
        (lparallel:end-kernel :wait t))
      (let ((bad 0))
        (dotimes (i n bad)
          (unless (= (aref serial i) (aref parallel i)) (incf bad)))))))

(defun run-parallel-scaling (&key (dataset :letter) (n-tree 500) (max-depth 10)
                                  (worker-counts '(1 2 4 8)))
  "Scaling of the compiled and array representations, and what happens to the walking one."
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (let* ((forest (make-forest n-class datamatrix target
                                :n-tree n-tree :bagging-ratio 0.1
                                :max-depth max-depth :n-trial n-trial
                                :min-region-samples 5))
           (compiled (compile-forest forest))
           (arrayed (build-array-forest forest))
           (n-rows (array-dimension datamatrix-test 0)))
      (format t "~&=== parallel scaling on ~A, ~Dx d=~D ===~%" dataset n-tree max-depth)
      (format t "~&| workers | compiled pred/s | array pred/s | walking: wrong answers |~%")
      (format t "|---|---|---|---|~%")
      (force-output)
      (dolist (w worker-counts)
        (let ((compiled-rate
                (time-parallel (lambda (start end)
                                 ;; One accumulator per chunk, so threads share nothing.
                                 (let ((acc (make-accumulator arrayed))
                                       (sum 0))
                                   (loop for i from start below end
                                         do (incf sum (predict-compiled-forest
                                                       compiled datamatrix-test i acc)))
                                   sum))
                               n-rows w))
              (array-rate
                (time-parallel (lambda (start end)
                                 (let ((acc (make-accumulator arrayed))
                                       (sum 0))
                                   (loop for i from start below end
                                         do (incf sum (predict-array-forest
                                                       arrayed datamatrix-test i acc)))
                                   sum))
                               n-rows w))
              (corruptions (if (= w 1)
                               0
                               (count-parallel-corruptions forest datamatrix-test w))))
          (format t "| ~D | ~,0F | ~,0F | ~D of ~D |~%"
                  w compiled-rate array-rate corruptions n-rows)
          (force-output)))
      (format t "~&PARALLEL_DONE~%")
      (force-output))))

;;;; Three representations and parallel scaling, measured
;;;;
;;;; (Taken with the earlier three-way benchmark, before PACKED-FOREST existed. The
;;;; four-way table further down supersedes the throughput columns here, measuring all of
;;;; them on one forest.)
;;;;
;;;; 500-tree forests, one run each. Rates are predictions per second.
;;;;
;;;; | | nodes | compile s | build s | disagreements | walk | compiled | array |
;;;; |---|---|---|---|---|---|---|---|
;;;; | letter d=5  |  29602 |  2.31 | 0.003 | 0 / 0 | 7205 | 53956 | 40716 |
;;;; | letter d=10 | 227594 | 16.22 | 0.020 | 0 / 0 | 3272 | 13227 |  8756 |
;;;; | MNIST d=5   |  31470 |  1.95 | 0.005 | 0 / 0 | 2834 | 63091 | 55658 |
;;;; | MNIST d=10  | 519302 | 47.86 | 0.035 | 0 / 0 | 2336 | 10638 |  8354 |
;;;;
;;;; Both new representations agree exactly with PREDICT-FOREST everywhere.
;;;;
;;;; Two predictions made before measuring were wrong, and it is worth saying which.
;;;;
;;;; The first was that flattening to arrays would overtake compiling on large forests,
;;;; because the instruction-cache penalty the decomposition found grows with code size.
;;;; It does grow -- but compiled still wins at every size here, by 1.1x to 1.5x, even at
;;;; 519302 nodes. Straight-line branches on immediate thresholds beat a loop doing four
;;;; or five array loads per node, and the i-cache penalty is not enough to close that.
;;;;
;;;; What arrays win instead is build time: 0.035 s against 47.86 s, a factor of 1368. For
;;;; a model built once and served forever that is irrelevant. For the iterated pruning in
;;;; src/experimental/fused-sibling-refinement.lisp, which rebuilds the forest 16 to 19
;;;; times, recompiling would cost a quarter of an hour of pure compilation and rebuilding
;;;; arrays costs under a second in total. That is the difference that decides which one is
;;;; usable there.
;;;;
;;;; Parallel scaling, 500 trees at depth 10, one accumulator per thread:
;;;;
;;;; | workers | letter compiled | letter array | MNIST compiled | MNIST array |
;;;; |---|---|---|---|---|
;;;; | 1 |  15923 | 12285 | 10236 | 12166 |
;;;; | 2 |  29411 | 18797 | 21552 | 18215 |
;;;; | 4 |  54053 | 38535 | 44053 | 35651 |
;;;; | 8 | 102995 | 59641 | 79619 | 45113 |
;;;;
;;;; The second wrong prediction: arrays would scale better, being compact read-only data
;;;; where compiled code thrashes the instruction cache. They scale worse -- 4.9x and 3.7x
;;;; on eight threads against compiled's 6.5x and 7.8x. The array version's working set is
;;;; data every core contends for in shared cache; the compiled version's is code, and each
;;;; core has its own L1i and L2. The i-cache pressure that hurts a single thread is
;;;; apparently cheaper to replicate per core than the array traffic is to share.
;;;;
;;;; Read the 1-worker column only against the rest of its own table: TIME-PARALLEL's chunk
;;;; loops over a range directly where TIME-PREDICTIONS funcalls a closure per row, so its
;;;; absolute numbers are not the three-way table's.
;;;;
;;;; And the walking representation, the same forest predicted from N threads:
;;;;
;;;; | workers | letter wrong answers | MNIST wrong answers |
;;;; |---|---|---|
;;;; | 2 | 2057 of 5000 | 3677 of 10000 |
;;;; | 4 | 3141 of 5000 | 5762 of 10000 |
;;;; | 8 | 3986 of 5000 | 6800 of 10000 |
;;;;
;;;; Up to 80% of predictions come back wrong. PREDICT-FOREST accumulates into
;;;; FOREST-CLASS-COUNT-ARRAY and every leaf read overwrites DTREE-CLASS-COUNT-ARRAY, both
;;;; slots on the shared model, so threads overwrite each other's partial sums. It does not
;;;; signal, it does not slow down, it returns confident wrong answers. Prediction
;;;; parallelism is not something these representations take away -- it is something the
;;;; library does not have, and either of them is what makes it possible.

;;;; Are the declarations actually working?
;;;;
;;;; Worth checking rather than assuming, since (safety 0) means SBCL trusts a declaration
;;;; without verifying it and a wrong one is silent. Compiling the file with notes visible
;;;; found exactly two, both on one expression:
;;;;
;;;;   ; in: DEFUN PREDICT-ARRAY-TREE
;;;;   ;     (AREF (ARRAY-FOREST-ROOTS ARRAY-FOREST) 0)
;;;;   ; note: unable to avoid runtime dispatch on array element type
;;;;   ; because: Upgraded element type of array is not known at compile time.
;;;;
;;;; PREDICT-ARRAY-FOREST -- the hot path, and the one all the forest numbers come from --
;;;; was clean, so its declarations were doing their job. The single-tree walk was reading
;;;; ROOTS without one, the only place a local declaration had been left out.
;;;;
;;;; The cause was that ARRAY-FOREST's slots had no :type, so every accessor returned T and
;;;; each caller had to re-declare whatever it read. Typing the slots fixes it at the
;;;; source: notes go to zero and no caller has to remember. Single-tree rates on letter
;;;; afterwards, against 0 disagreements with PREDICT-DTREE:
;;;;
;;;;   d=5    57 nodes  102.0M predictions/s
;;;;   d=10 1113 nodes   53.3M
;;;;   d=15 2881 nodes   39.7M
;;;;
;;;; Against the fully compiled tree's 312M / 208M / 165M from the first table, so the
;;;; array walk gives up 3-4x on a single tree -- the cost of loading feature, threshold
;;;; and child per node instead of having them as immediates in straight-line code. It is
;;;; still 20-150x the library's own PREDICT-DTREE.

;;;; LightGBM's packing
;;;;
;;;; ARRAY-FOREST above is Structure-of-Arrays, which is LightGBM's shape, but it makes two
;;;; choices LightGBM does not.
;;;;
;;;; It gives every node a slot, leaves included, and marks them in a bit vector. LightGBM
;;;; stores only internal nodes -- there are exactly n-leaf - 1 of them per tree -- and
;;;; encodes a leaf as a *negative child index*, LEFT-CHILD holding ~leaf. The loop's
;;;; continuation test then is the leaf test:
;;;;
;;;;   while (node >= 0) node = decision(...);
;;;;   return leaf_value[~node];
;;;;
;;;; No bit vector, no extra read per node, and the node arrays halve because leaves no
;;;; longer occupy slots in FEATURE, THRESHOLD and RIGHT that nothing ever reads.
;;;;
;;;; And it uses fixnum for every index, which is eight bytes to address a few hundred
;;;; thousand nodes. (signed-byte 32) is enough for both the node and leaf numbering here
;;;; and halves those arrays again.
;;;;
;;;; PACKED-FOREST is ARRAY-FOREST with both changes. Both are kept so they can be measured
;;;; against each other in one process rather than across runs.

(defstruct (packed-forest (:constructor %make-packed-forest))
  "A forest as LightGBM lays one out: internal nodes only, leaves as negative indices.

LEFT and RIGHT hold a non-negative internal-node index, or the bitwise complement of a leaf
number. ROOTS may hold either, since a tree that never split is a bare leaf. FEATURE and
THRESHOLD are indexed by internal-node number only, so they are half the length
ARRAY-FOREST needs."
  (n-class 0 :type fixnum)
  (n-tree 0 :type fixnum)
  (n-internal 0 :type fixnum)
  (n-leaf 0 :type fixnum)
  (feature (make-array 0 :element-type '(unsigned-byte 32))
           :type (simple-array (unsigned-byte 32) (*)))
  (threshold (make-array 0 :element-type 'single-float)
             :type (simple-array single-float (*)))
  (left (make-array 0 :element-type '(signed-byte 32))
        :type (simple-array (signed-byte 32) (*)))
  (right (make-array 0 :element-type '(signed-byte 32))
         :type (simple-array (signed-byte 32) (*)))
  (roots (make-array 0 :element-type '(signed-byte 32))
         :type (simple-array (signed-byte 32) (*)))
  (distributions (make-array '(0 0) :element-type 'single-float)
                 :type (simple-array single-float (* *)))
  (leaf-classes (make-array 0 :element-type '(unsigned-byte 32))
                :type (simple-array (unsigned-byte 32) (*))))

(defun build-packed-forest (forest &key (leaf-payload :distribution))
  "Flatten FOREST the way LightGBM lays a model out.

LEAF-PAYLOAD :distribution fills DISTRIBUTIONS, which is what a forest sums; :class fills
LEAF-CLASSES with each leaf's argmax, which is what a single tree answers with."
  (let ((feature (make-array 64 :element-type '(unsigned-byte 32)
                                :adjustable t :fill-pointer 0))
        (threshold (make-array 64 :element-type 'single-float
                                  :adjustable t :fill-pointer 0))
        (left (make-array 64 :element-type '(signed-byte 32) :adjustable t :fill-pointer 0))
        (right (make-array 64 :element-type '(signed-byte 32) :adjustable t :fill-pointer 0))
        (leaf-data '())
        (n-leaf 0)
        (roots '())
        (n-class (forest-n-class forest)))
    (labels ((emit (node)
               "Return NODE's index: non-negative for an internal node, ~leaf for a leaf."
               (cond
                 ((node-test-attribute node)
                  ;; Reserve before recursing, so a child's index exceeds its parent's.
                  (let ((self (fill-pointer feature)))
                    (vector-push-extend 0 feature)
                    (vector-push-extend 0.0 threshold)
                    (vector-push-extend 0 left)
                    (vector-push-extend 0 right)
                    (let ((l (emit (node-left-node node)))
                          (r (emit (node-right-node node))))
                      (setf (aref feature self) (node-test-attribute node)
                            (aref threshold self) (node-test-threshold node)
                            (aref left self) l
                            (aref right self) r))
                    self))
                 (t
                  (push (ecase leaf-payload
                          (:distribution (leaf-distribution node))
                          (:class (leaf-class node)))
                        leaf-data)
                  (prog1 (lognot n-leaf) (incf n-leaf))))))
      (dolist (dtree (forest-dtree-list forest))
        (push (emit (dtree-root dtree)) roots)))
    (setf leaf-data (nreverse leaf-data))
    (let ((table (make-array (list (if (eq leaf-payload :distribution) (max n-leaf 1) 0)
                                   (if (eq leaf-payload :distribution) n-class 0))
                             :element-type 'single-float :initial-element 0.0))
          (classes (make-array (if (eq leaf-payload :class) n-leaf 0)
                               :element-type '(unsigned-byte 32) :initial-element 0)))
      (ecase leaf-payload
        (:distribution
         (loop for dist in leaf-data
               for row from 0
               do (dotimes (k n-class) (setf (aref table row k) (aref dist k)))))
        (:class
         (loop for class in leaf-data
               for i from 0
               do (setf (aref classes i) class))))
      (%make-packed-forest
       :n-class n-class
       :n-tree (forest-n-tree forest)
       :n-internal (fill-pointer feature)
       :n-leaf n-leaf
       :feature (coerce feature '(simple-array (unsigned-byte 32) (*)))
       :threshold (coerce threshold '(simple-array single-float (*)))
       :left (coerce left '(simple-array (signed-byte 32) (*)))
       :right (coerce right '(simple-array (signed-byte 32) (*)))
       :roots (coerce (nreverse roots) '(simple-array (signed-byte 32) (*)))
       :distributions table
       :leaf-classes classes))))

(defun make-packed-accumulator (packed-forest)
  "A fresh accumulator for PREDICT-PACKED-FOREST. One per thread."
  (make-array (packed-forest-n-class packed-forest)
              :element-type 'single-float :initial-element 0.0))

(defun predict-packed-forest (packed-forest datamatrix datum-index acc)
  "PREDICT-FOREST's answer from the packed layout. Writes only ACC."
  (declare (optimize (speed 3) (safety 0))
           (type packed-forest packed-forest)
           (type (simple-array single-float (* *)) datamatrix)
           (type (simple-array single-float (*)) acc)
           (type fixnum datum-index))
  (let ((feature (packed-forest-feature packed-forest))
        (threshold (packed-forest-threshold packed-forest))
        (left (packed-forest-left packed-forest))
        (right (packed-forest-right packed-forest))
        (roots (packed-forest-roots packed-forest))
        (table (packed-forest-distributions packed-forest))
        (n-class (packed-forest-n-class packed-forest))
        (n-tree (packed-forest-n-tree packed-forest)))
    (declare (type (simple-array (unsigned-byte 32) (*)) feature)
             (type (simple-array single-float (*)) threshold)
             (type (simple-array (signed-byte 32) (*)) left right roots)
             (type (simple-array single-float (* *)) table)
             (type fixnum n-class n-tree))
    (dotimes (k n-class) (setf (aref acc k) 0.0))
    (dotimes (tree n-tree)
      (let ((node (aref roots tree)))
        (declare (type (signed-byte 32) node))
        ;; The loop's own test is the leaf test: no bit vector, no extra read.
        (loop while (>= node 0)
              do (setf node (if (>= (aref datamatrix datum-index (aref feature node))
                                    (aref threshold node))
                                (aref left node)
                                (aref right node))))
        (let ((row (lognot node)))
          (declare (type fixnum row))
          (dotimes (k n-class)
            (incf (aref acc k) (aref table row k))))))
    (dotimes (k n-class)
      (setf (aref acc k) (/ (aref acc k) n-tree)))
    (argmax acc)))

(defun predict-packed-tree (packed-forest datamatrix datum-index)
  "The class a :class-payload single-tree PACKED-FOREST gives."
  (declare (optimize (speed 3) (safety 0))
           (type packed-forest packed-forest)
           (type (simple-array single-float (* *)) datamatrix)
           (type fixnum datum-index))
  (let ((feature (packed-forest-feature packed-forest))
        (threshold (packed-forest-threshold packed-forest))
        (left (packed-forest-left packed-forest))
        (right (packed-forest-right packed-forest))
        (classes (packed-forest-leaf-classes packed-forest))
        (node (aref (packed-forest-roots packed-forest) 0)))
    (declare (type (simple-array (unsigned-byte 32) (*)) feature classes)
             (type (simple-array single-float (*)) threshold)
             (type (simple-array (signed-byte 32) (*)) left right)
             (type (signed-byte 32) node))
    (loop while (>= node 0)
          do (setf node (if (>= (aref datamatrix datum-index (aref feature node))
                                (aref threshold node))
                            (aref left node)
                            (aref right node))))
    (aref classes (lognot node))))

(defun count-packed-forest-disagreements (forest packed datamatrix)
  "How many rows PREDICT-FOREST and PREDICT-PACKED-FOREST answer differently."
  (let ((acc (make-packed-accumulator packed))
        (bad 0))
    (dotimes (i (array-dimension datamatrix 0) bad)
      (unless (= (predict-forest forest datamatrix i)
                 (predict-packed-forest packed datamatrix i acc))
        (incf bad)))))

;;;; Bytes

(defun array-bytes (array element-bits)
  (ceiling (* (array-total-size array) element-bits) 8))

(defun array-forest-bytes (af)
  "Bytes in ARRAY-FOREST's node arrays, and in its leaf table, separately."
  (values (+ (array-bytes (array-forest-feature af) 64)
             (array-bytes (array-forest-threshold af) 32)
             (array-bytes (array-forest-left af) 64)
             (array-bytes (array-forest-right af) 64)
             (array-bytes (array-forest-leaf-p af) 1))
          (array-bytes (array-forest-distributions af) 32)))

(defun packed-forest-bytes (pf)
  "Bytes in PACKED-FOREST's node arrays, and in its leaf table, separately."
  (values (+ (array-bytes (packed-forest-feature pf) 32)
             (array-bytes (packed-forest-threshold pf) 32)
             (array-bytes (packed-forest-left pf) 32)
             (array-bytes (packed-forest-right pf) 32))
          (array-bytes (packed-forest-distributions pf) 32)))

;;;; array against packed, in one process

(defun compare-layouts (n-class datamatrix target datamatrix-test n-tree max-depth n-trial)
  "Build one forest, lay it out both ways, and report agreement, bytes and speed."
  (let* ((forest (make-forest n-class datamatrix target
                              :n-tree n-tree :bagging-ratio 0.1
                              :max-depth max-depth :n-trial n-trial :min-region-samples 5)))
    (multiple-value-bind (af af-seconds) (seconds (build-array-forest forest))
      (multiple-value-bind (pf pf-seconds) (seconds (build-packed-forest forest))
        (multiple-value-bind (af-nodes af-leaves) (array-forest-bytes af)
          (multiple-value-bind (pf-nodes) (packed-forest-bytes pf)
            (let ((af-acc (make-accumulator af))
                  (pf-acc (make-packed-accumulator pf)))
              (format t "~&| ~Dx d=~D | ~D / ~D | ~,1F / ~,1F MB nodes | ~,1F MB leaves | ~
~,3F / ~,3F s build | ~,0F / ~,0F pred/s | ~,2Fx |~%"
                      n-tree max-depth
                      (count-array-forest-disagreements forest af datamatrix-test)
                      (count-packed-forest-disagreements forest pf datamatrix-test)
                      (/ af-nodes 1048576.0) (/ pf-nodes 1048576.0)
                      (/ af-leaves 1048576.0)
                      af-seconds pf-seconds
                      (time-predictions (lambda (d i) (predict-array-forest af d i af-acc))
                                        datamatrix-test)
                      (time-predictions (lambda (d i) (predict-packed-forest pf d i pf-acc))
                                        datamatrix-test)
                      (/ (time-predictions (lambda (d i) (predict-packed-forest pf d i pf-acc))
                                           datamatrix-test)
                         (time-predictions (lambda (d i) (predict-array-forest af d i af-acc))
                                           datamatrix-test)))
              (force-output))))))))

(defun compare-layout-trees (n-class datamatrix target datamatrix-test max-depth n-trial)
  "The same comparison for a single tree, where the leaf payload is a class."
  (let* ((forest (make-forest n-class datamatrix target
                              :n-tree 1 :bagging-ratio 1.0
                              :max-depth max-depth :n-trial n-trial :min-region-samples 5))
         (dtree (first (forest-dtree-list forest)))
         (af (build-array-forest forest :leaf-payload :class))
         (pf (build-packed-forest forest :leaf-payload :class))
         (af-bad 0)
         (pf-bad 0))
    (dotimes (i (array-dimension datamatrix-test 0))
      (let ((expected (predict-dtree dtree datamatrix-test i)))
        (unless (= expected (predict-array-tree af datamatrix-test i)) (incf af-bad))
        (unless (= expected (predict-packed-tree pf datamatrix-test i)) (incf pf-bad))))
    (format t "~&| tree d=~D | ~D / ~D | ~,0F / ~,0F pred/s | ~,2Fx |~%"
            max-depth af-bad pf-bad
            (time-predictions (lambda (d i) (predict-array-tree af d i)) datamatrix-test)
            (time-predictions (lambda (d i) (predict-packed-tree pf d i)) datamatrix-test)
            (/ (time-predictions (lambda (d i) (predict-packed-tree pf d i)) datamatrix-test)
               (time-predictions (lambda (d i) (predict-array-tree af d i)) datamatrix-test)))
    (force-output)))

(defun run-layout-comparison (&key (dataset :letter) (depths '(5 10 15))
                                   (configs '((500 5) (500 10))))
  "array against packed: agreement, bytes, build time, speed. Ratios are packed / array."
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (format t "~&=== array vs packed on ~A ===~%" dataset)
    (format t "~&| model | disagreements a/p | node bytes a/p | leaf bytes | build a/p | pred/s a/p | ratio |~%")
    (format t "|---|---|---|---|---|---|---|~%")
    (force-output)
    (dolist (depth depths)
      (compare-layout-trees n-class datamatrix target datamatrix-test depth n-trial))
    (dolist (config configs)
      (compare-layouts n-class datamatrix target datamatrix-test
                       (first config) (second config) n-trial))
    (format t "~&LAYOUT_DONE~%")
    (force-output)))

;;;; array against packed, measured
;;;;
;;;; One forest per row, laid out both ways in the same process so the two are not being
;;;; compared across runs. Ratios are packed / array. Both agree exactly with the library
;;;; everywhere.
;;;;
;;;; letter:
;;;;
;;;; | model | node bytes a/p | build s a/p | pred/s array | pred/s packed | ratio |
;;;; |---|---|---|---|---|---|
;;;; | tree d=5   |     |             | 103219381 | 123479012 | 1.19x |
;;;; | tree d=10  |     |             |  54659563 |  65199609 | 1.19x |
;;;; | tree d=15  |     |             |  41219588 |  46189815 | 1.14x |
;;;; | 500x d=5   | 0.8 / 0.2 MB | 0.010 / 0.007 |  38910 |  55350 | 1.43x |
;;;; | 500x d=10  | 6.2 / 1.7 MB | 0.069 / 0.055 |   8361 |  15267 | 1.84x |
;;;;
;;;; MNIST:
;;;;
;;;; | model | node bytes a/p | build s a/p | pred/s array | pred/s packed | ratio |
;;;; |---|---|---|---|---|---|
;;;; | tree d=5   |      |             |  49419703 |  84219326 | 1.79x |
;;;; | tree d=10  |      |             |  21139873 |  31399686 | 1.55x |
;;;; | tree d=15  |      |             |  12439925 |  17839822 | 1.49x |
;;;; | 500x d=5   |  0.8 / 0.2 MB | 0.007 / 0.006 |  56074 |  68610 | 1.23x |
;;;; | 500x d=10  | 14.0 / 4.0 MB | 0.057 / 0.049 |   8077 |  19047 | 2.35x |
;;;;
;;;; The node arrays fall to between a quarter and a third: 14.0 MB to 4.0 MB on MNIST's
;;;; largest forest. Dropping the leaves' unused slots accounts for about half of that and
;;;; narrowing the indices from fixnum to 32 bits for the rest, which is what the two
;;;; changes predict.
;;;;
;;;; Speed follows size. The gain is smallest on a single tree (1.14-1.19x on letter),
;;;; where the arrays fit in cache either way and all that is saved is the bit-vector read
;;;; per node, and largest on the biggest forest (2.35x on MNIST at depth 10), where 14 MB
;;;; of node data did not fit and 4 MB comes closer. This is the first change in this file
;;;; whose effect grows with model size rather than shrinking.
;;;;
;;;; It also closes most of the gap to the compiled representation. On MNIST 500x d=10 the
;;;; earlier three-way table had walk 2336, array 8354, compiled 10638; packed reaches
;;;; 19047 on its own forest, past compiled -- though on a different forest build, so the
;;;; two need measuring side by side before that is claimed.

;;;; All four on one forest
;;;;
;;;; Every comparison so far has been between two representations built from the same
;;;; forest, but across pairs the forests differed -- MAKE-FOREST is unseeded, and the
;;;; suggestion that packed had overtaken compiled rested on numbers from two different
;;;; builds. This measures all four on one.

(defun code-bytes (function)
  "Machine code size of FUNCTION, or NIL if this implementation will not say.
SBCL-specific and looked up by name so the file still reads elsewhere."
  (let ((size-fn (find-symbol "%CODE-CODE-SIZE" "SB-KERNEL"))
        (header-fn (find-symbol "FUN-CODE-HEADER" "SB-KERNEL")))
    (when (and size-fn header-fn (fboundp size-fn) (fboundp header-fn))
      (ignore-errors (funcall size-fn (funcall header-fn function))))))

(defun compiled-forest-code-bytes (cf)
  "Total machine code of a COMPILED-FOREST's predictors, or NIL."
  (let ((total 0))
    (dotimes (i (compiled-forest-n-tree cf) total)
      (let ((bytes (code-bytes (svref (compiled-forest-predictors cf) i))))
        (unless bytes (return-from compiled-forest-code-bytes nil))
        (incf total bytes)))))

(defun benchmark-four-ways (n-class datamatrix target datamatrix-test n-tree max-depth
                            n-trial)
  "One forest, four ways: walking, compiled, array, packed."
  (let ((forest (make-forest n-class datamatrix target
                             :n-tree n-tree :bagging-ratio 0.1
                             :max-depth max-depth :n-trial n-trial :min-region-samples 5)))
    (multiple-value-bind (compiled compile-seconds) (seconds (compile-forest forest))
      (let* ((af (build-array-forest forest))
             (pf (build-packed-forest forest))
             (af-acc (make-accumulator af))
             (pf-acc (make-packed-accumulator pf))
             (code (compiled-forest-code-bytes compiled))
             (walk (time-predictions (lambda (d i) (predict-forest forest d i))
                                     datamatrix-test))
             (comp (time-predictions (lambda (d i) (predict-compiled-forest compiled d i))
                                     datamatrix-test))
             (arr (time-predictions (lambda (d i) (predict-array-forest af d i af-acc))
                                    datamatrix-test))
             (pack (time-predictions (lambda (d i) (predict-packed-forest pf d i pf-acc))
                                     datamatrix-test)))
        (multiple-value-bind (af-nodes af-leaves) (array-forest-bytes af)
          (declare (ignore af-nodes))
          (multiple-value-bind (pf-nodes pf-leaves) (packed-forest-bytes pf)
            (declare (ignore pf-leaves))
            (format t "~&| ~Dx d=~D | ~D | ~D/~D/~D | ~,1F s | ~@[~,1F MB code~] | ~
~,1F MB packed | ~,0F | ~,0F | ~,0F | ~,0F |~%"
                    n-tree max-depth (packed-forest-n-internal pf)
                    (count-forest-disagreements forest compiled datamatrix-test)
                    (count-array-forest-disagreements forest af datamatrix-test)
                    (count-packed-forest-disagreements forest pf datamatrix-test)
                    compile-seconds
                    (and code (/ code 1048576.0))
                    (/ (+ pf-nodes af-leaves) 1048576.0)
                    walk comp arr pack)
            (force-output)))))))

(defun run-four-ways (&key (dataset :letter) (configs '((500 5) (500 10))))
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (format t "~&=== four representations, one forest each, on ~A ===~%" dataset)
    (format t "~&| model | internal nodes | disagreements c/a/p | compile | code | packed bytes | walk | compiled | array | packed |~%")
    (format t "|---|---|---|---|---|---|---|---|---|---|~%")
    (force-output)
    (dolist (config configs)
      (benchmark-four-ways n-class datamatrix target datamatrix-test
                           (first config) (second config) n-trial))
    (format t "~&FOUR_WAYS_DONE~%")
    (force-output)))

;;;; All four on one forest, measured
;;;;
;;;; One forest per row, all four representations built from it. Disagreements are
;;;; compiled / array / packed against PREDICT-FOREST, and zero everywhere.
;;;;
;;;; | | internal nodes | compile | code MB | packed MB | walk | compiled | array | packed |
;;;; |---|---|---|---|---|---|---|---|---|
;;;; | letter 500x d=5  |  14551 |  2.3 s |  0.8 |  1.7 | 7396 | 52173 | 40257 | 53860 |
;;;; | letter 500x d=10 | 113547 | 16.1 s |  5.9 | 13.0 | 3274 | 13495 |  8696 | 16051 |
;;;; | MNIST 500x d=5   |  15485 |  1.9 s |  0.8 |  0.8 | 2834 | 61255 | 56074 | 68259 |
;;;; | MNIST 500x d=10  | 259401 | 47.9 s | 13.5 | 13.9 | 2278 | 10438 |  8217 | 19120 |
;;;;
;;;; Packed is fastest in all four, and the earlier suggestion that it had overtaken
;;;; compiled -- which rested on two different forest builds -- holds up on one: 1.03x at
;;;; letter depth 5, 1.19x at letter depth 10, 1.11x at MNIST depth 5, and 1.83x at MNIST
;;;; depth 10. The margin grows with the model, which is the same pattern the array-versus-
;;;; packed comparison showed and the opposite of what compiling does.
;;;;
;;;; So the prediction that started this -- that laying trees out as data would beat
;;;; compiling them for large forests -- was right after all. It failed the first time
;;;; because of how the data was laid out, not because the idea was wrong: giving leaves
;;;; node slots and addressing them with 64-bit indices cost enough to lose 1.5x to
;;;; compiled code, and removing both turns it into a 1.8x win.
;;;;
;;;; Memory, the thing left unmeasured until now. SBCL will report machine code size, so
;;;; both sides can be counted. On MNIST's largest forest the compiled predictors are 13.5
;;;; MB of code and the packed layout is 13.9 MB of arrays -- nearly identical, and in both
;;;; cases most of it is unavoidable: 259401 internal nodes have to be described somehow.
;;;; The packed total is 4.0 MB of nodes and 9.9 MB of leaf distributions, so two thirds of
;;;; it is the class distributions rather than the tree structure, and a forest with fewer
;;;; classes or a regression forest would be much smaller. letter, with 26 classes, spends
;;;; 11.4 of its 13.0 MB the same way.
;;;;
;;;; What each is for, on this evidence:
;;;;
;;;;   packed    fastest everywhere, 0.05 s to build, and the only one worth using in a
;;;;             loop that rebuilds the model -- iterated pruning rebuilds 16 to 19 times
;;;;   compiled  no longer fastest at any size measured here, and costs 47.9 s to build
;;;;   array     superseded by packed; kept only so the two can be compared
;;;;   walking   what the library does today, 3 to 8 times slower than any of them, and
;;;;             unsafe to call from more than one thread

;;;; Second cut: topology separated from leaf payload
;;;;
;;;; PACKED-FOREST carries both a classifier's distribution table and a single tree's class
;;;; array, one of which is always empty, and it computes whichever the caller asked for
;;;; while building. Review of docs/packed-forest-layout.md pointed out what that costs:
;;;;
;;;;   - Refinement needs only the leaf a datum reaches, so building a payload at all is
;;;;     waste -- and the iterated pruning loop rebuilds 16 to 19 times.
;;;;   - Regression wants a scalar per leaf, which neither slot fits.
;;;;   - The leaf table is two thirds of the memory, and leaves average two non-zero
;;;;     classes of 26, so a sparse payload is worth far more than any further squeezing
;;;;     of the node arrays.
;;;;
;;;; So: PACKED-TOPOLOGY is the tree structure and nothing else, and a payload is attached
;;;; separately. The builder is two-pass, sizing exactly and validating before anything
;;;; reaches a (safety 0) loop.

(defstruct (packed-topology (:constructor %make-packed-topology))
  "The structure of a forest: internal nodes, and which leaf a datum lands in.

Leaf numbering is not left to the traversal order. Leaves are numbered
tree-leaf-offsets[tree] + the tree's own leaf index, which is by construction the index
Global Refinement uses (FOREST-INDEX-OFFSET plus NODE-LEAF-INDEX). Deriving it rather than
letting two walks happen to agree is what lets the node ordering change later -- to
breadth-first, or anything cache-conscious -- without silently renumbering the refine
feature space."
  (n-tree 0 :type fixnum)
  (n-internal 0 :type fixnum)
  (n-leaf 0 :type fixnum)
  (feature (make-array 0 :element-type '(unsigned-byte 32))
           :type (simple-array (unsigned-byte 32) (*)))
  (threshold (make-array 0 :element-type 'single-float)
             :type (simple-array single-float (*)))
  (left (make-array 0 :element-type '(signed-byte 32))
        :type (simple-array (signed-byte 32) (*)))
  (right (make-array 0 :element-type '(signed-byte 32))
         :type (simple-array (signed-byte 32) (*)))
  (roots (make-array 0 :element-type '(signed-byte 32))
         :type (simple-array (signed-byte 32) (*)))
  (tree-leaf-offsets (make-array 0 :element-type '(unsigned-byte 32))
                     :type (simple-array (unsigned-byte 32) (*))))

(define-condition packed-build-error (error)
  ((detail :initarg :detail :reader packed-build-error-detail))
  (:report (lambda (c s) (format s "cannot pack this forest: ~A"
                                 (packed-build-error-detail c))))
  (:documentation
   "Signalled for anything the packed traversal could not survive.

The traversal runs at (safety 0) and trusts its declarations, so every assumption it makes
has to hold before it starts: indices in range, both children present, and -- the one that
is not obvious -- a leaf that still has its sample indices. CLASS-DISTRIBUTION returns a
*uniform* distribution rather than signalling when a leaf has none, which pruning creates
by default (issue #14), so a builder that just read it would freeze a uniform distribution
into the model with nothing to show for it."))

(defun check-packable (forest)
  "Signal PACKED-BUILD-ERROR unless FOREST can be packed. Returns the datum dimension."
  (let ((dtrees (forest-dtree-list forest))
        (dim (forest-datum-dim forest)))
    (unless (= (length dtrees) (forest-n-tree forest))
      (error 'packed-build-error
             :detail (format nil "~D trees in the list, FOREST-N-TREE says ~D"
                             (length dtrees) (forest-n-tree forest))))
    (labels ((walk (node depth)
               (cond
                 ((null node)
                  (error 'packed-build-error :detail "a nil node"))
                 ((node-test-attribute node)
                  (let ((f (node-test-attribute node)))
                    (unless (and (typep f 'fixnum) (<= 0 f) (< f dim))
                      (error 'packed-build-error
                             :detail (format nil "feature ~S out of [0,~D) at depth ~D"
                                             f dim depth)))
                    (unless (typep (node-test-threshold node) 'single-float)
                      (error 'packed-build-error
                             :detail (format nil "threshold ~S is not a single-float"
                                             (node-test-threshold node))))
                    (unless (and (node-left-node node) (node-right-node node))
                      (error 'packed-build-error
                             :detail "an internal node with only one child"))
                    (walk (node-left-node node) (1+ depth))
                    (walk (node-right-node node) (1+ depth))))
                 (t
                  ;; A leaf. Its distribution has to be derivable, not invented.
                  (unless (node-sample-indices node)
                    (error 'packed-build-error
                           :detail (format nil "a leaf at depth ~D has no sample indices, ~
so its class distribution would silently come out uniform (issue #14) -- rebuild the ~
forest with :remove-sample-indices? nil" depth)))))))
      (dolist (dtree dtrees dim)
        (walk (dtree-root dtree) 0)))))

(defun count-tree-nodes (dtree)
  "Return (values n-internal n-leaf) for DTREE."
  (let ((internal 0) (leaves 0))
    (labels ((walk (node)
               (if (node-test-attribute node)
                   (progn (incf internal)
                          (walk (node-left-node node))
                          (walk (node-right-node node)))
                   (incf leaves))))
      (walk (dtree-root dtree)))
    (values internal leaves)))

(defun build-packed-topology (forest)
  "Flatten FOREST's structure. Two passes: count and validate, then fill exact-size arrays."
  (check-packable forest)
  (let* ((dtrees (forest-dtree-list forest))
         (n-tree (length dtrees))
         (internal-counts (make-array n-tree))
         (leaf-counts (make-array n-tree))
         (n-internal 0)
         (n-leaf 0))
    (loop for dtree in dtrees
          for tree from 0
          do (multiple-value-bind (i l) (count-tree-nodes dtree)
               (setf (svref internal-counts tree) i
                     (svref leaf-counts tree) l)
               (incf n-internal i)
               (incf n-leaf l)))
    (unless (< n-internal (expt 2 31))
      (error 'packed-build-error
             :detail (format nil "~D internal nodes exceeds a (signed-byte 32) index"
                             n-internal)))
    (unless (< n-leaf (expt 2 31))
      (error 'packed-build-error
             :detail (format nil "~D leaves exceeds a (signed-byte 32) index" n-leaf)))
    (let ((feature (make-array n-internal :element-type '(unsigned-byte 32)))
          (threshold (make-array n-internal :element-type 'single-float))
          (left (make-array n-internal :element-type '(signed-byte 32)))
          (right (make-array n-internal :element-type '(signed-byte 32)))
          (roots (make-array n-tree :element-type '(signed-byte 32)))
          (offsets (make-array n-tree :element-type '(unsigned-byte 32)))
          (node-cursor 0)
          (leaf-base 0))
      (loop for dtree in dtrees
            for tree from 0
            do (setf (aref offsets tree) leaf-base)
               (let ((local-leaf 0))
                 (labels ((emit (node)
                            (cond
                              ((node-test-attribute node)
                               (let ((self node-cursor))
                                 (incf node-cursor)
                                 (setf (aref feature self) (node-test-attribute node)
                                       (aref threshold self) (node-test-threshold node))
                                 (setf (aref left self) (emit (node-left-node node)))
                                 (setf (aref right self) (emit (node-right-node node)))
                                 self))
                              (t
                               ;; The leaf's global number, by construction the refine index.
                               (prog1 (lognot (+ leaf-base local-leaf))
                                 (incf local-leaf))))))
                   (setf (aref roots tree) (emit (dtree-root dtree))))
                 (incf leaf-base local-leaf)))
      (%make-packed-topology
       :n-tree n-tree :n-internal n-internal :n-leaf n-leaf
       :feature feature :threshold threshold :left left :right right
       :roots roots :tree-leaf-offsets offsets))))

(declaim (inline topology-leaf))
(defun topology-leaf (topology datamatrix datum-index root)
  "The leaf number a datum reaches from ROOT. This is the refine index."
  (declare (optimize (speed 3) (safety 0))
           (type packed-topology topology)
           (type (simple-array single-float (* *)) datamatrix)
           (type fixnum datum-index)
           (type (signed-byte 32) root))
  (let ((feature (packed-topology-feature topology))
        (threshold (packed-topology-threshold topology))
        (left (packed-topology-left topology))
        (right (packed-topology-right topology))
        (node root))
    (declare (type (simple-array (unsigned-byte 32) (*)) feature)
             (type (simple-array single-float (*)) threshold)
             (type (simple-array (signed-byte 32) (*)) left right)
             (type (signed-byte 32) node))
    (loop while (>= node 0)
          do (setf node (if (>= (aref datamatrix datum-index (aref feature node))
                                (aref threshold node))
                            (aref left node)
                            (aref right node))))
    (lognot node)))

(defun topology-leaves (topology datamatrix datum-index out)
  "Fill OUT with the leaf each tree sends a datum to. This is MAKE-REFINE-VECTOR's answer."
  (declare (optimize (speed 3) (safety 0))
           (type packed-topology topology)
           (type (simple-array single-float (* *)) datamatrix)
           (type (simple-array fixnum (*)) out)
           (type fixnum datum-index))
  (let ((roots (packed-topology-roots topology))
        (n-tree (packed-topology-n-tree topology)))
    (declare (type (simple-array (signed-byte 32) (*)) roots)
             (type fixnum n-tree))
    (dotimes (tree n-tree out)
      (setf (aref out tree) (topology-leaf topology datamatrix datum-index
                                           (aref roots tree))))))

;;;; Leaf payloads
;;;;
;;;; Two classifier payloads over one topology. Dense is the n-leaf x n-class table
;;;; PACKED-FOREST used. CSR keeps only the non-zero classes, which the measurement in
;;;; docs/packed-forest-layout.md says is 2.04 of 26 on letter and 2.06 of 10 on MNIST.
;;;;
;;;; Skipping the zeros is exact rather than approximate. Every value is non-negative, so
;;;; omitting a `+ 0.0` changes no sum, and the accumulator ends bit-identical to the dense
;;;; one -- which the tests check element by element, not merely after ARGMAX.

(defstruct (dense-classifier (:constructor %make-dense-classifier))
  "A distribution per leaf, stored in full."
  (topology (%make-packed-topology) :type packed-topology)
  (n-class 0 :type fixnum)
  (table (make-array '(0 0) :element-type 'single-float)
         :type (simple-array single-float (* *))))

(defstruct (csr-classifier (:constructor %make-csr-classifier))
  "A distribution per leaf, stored as its non-zero entries.

OFFSETS has n-leaf + 1 entries; leaf L occupies [offsets[L], offsets[L+1]) of CLASS and
PROBABILITY."
  (topology (%make-packed-topology) :type packed-topology)
  (n-class 0 :type fixnum)
  (offsets (make-array 1 :element-type '(unsigned-byte 32))
           :type (simple-array (unsigned-byte 32) (*)))
  (class (make-array 0 :element-type '(unsigned-byte 16))
         :type (simple-array (unsigned-byte 16) (*)))
  (probability (make-array 0 :element-type 'single-float)
               :type (simple-array single-float (*))))

(defun collect-leaf-distributions (forest topology)
  "A simple-vector of every leaf's distribution, indexed by the topology's leaf number."
  (let ((out (make-array (packed-topology-n-leaf topology)))
        (offsets (packed-topology-tree-leaf-offsets topology)))
    (loop for dtree in (forest-dtree-list forest)
          for tree from 0
          do (let ((local 0)
                   (base (aref offsets tree)))
               (labels ((walk (node)
                          (if (node-test-attribute node)
                              (progn (walk (node-left-node node))
                                     (walk (node-right-node node)))
                              (progn (setf (svref out (+ base local))
                                           (leaf-distribution node))
                                     (incf local)))))
                 (walk (dtree-root dtree)))))
    out))

(defun build-dense-classifier (forest &optional (topology (build-packed-topology forest)))
  (let* ((n-class (forest-n-class forest))
         (n-leaf (packed-topology-n-leaf topology))
         (dists (collect-leaf-distributions forest topology))
         (table (make-array (list (max n-leaf 1) n-class)
                            :element-type 'single-float :initial-element 0.0)))
    (dotimes (row n-leaf)
      (let ((d (svref dists row)))
        (dotimes (k n-class) (setf (aref table row k) (aref d k)))))
    (%make-dense-classifier :topology topology :n-class n-class :table table)))

(defun build-csr-classifier (forest &optional (topology (build-packed-topology forest)))
  (let* ((n-class (forest-n-class forest))
         (n-leaf (packed-topology-n-leaf topology))
         (dists (collect-leaf-distributions forest topology))
         (nnz 0))
    (unless (< n-class 65536)
      (error 'packed-build-error
             :detail (format nil "~D classes exceeds a (unsigned-byte 16) class index"
                             n-class)))
    (dotimes (row n-leaf)
      (let ((d (svref dists row)))
        (dotimes (k n-class) (unless (zerop (aref d k)) (incf nnz)))))
    (let ((offsets (make-array (1+ n-leaf) :element-type '(unsigned-byte 32)))
          (class (make-array (max nnz 1) :element-type '(unsigned-byte 16)))
          (probability (make-array (max nnz 1) :element-type 'single-float))
          (cursor 0))
      (dotimes (row n-leaf)
        (setf (aref offsets row) cursor)
        (let ((d (svref dists row)))
          (dotimes (k n-class)
            (let ((p (aref d k)))
              (unless (zerop p)
                (setf (aref class cursor) k
                      (aref probability cursor) p)
                (incf cursor))))))
      (setf (aref offsets n-leaf) cursor)
      (%make-csr-classifier :topology topology :n-class n-class
                            :offsets offsets :class class :probability probability))))

(defun predict-dense (classifier datamatrix datum-index acc)
  "PREDICT-FOREST's answer from a dense payload. Writes only ACC."
  (declare (optimize (speed 3) (safety 0))
           (type dense-classifier classifier)
           (type (simple-array single-float (* *)) datamatrix)
           (type (simple-array single-float (*)) acc)
           (type fixnum datum-index))
  (let* ((topology (dense-classifier-topology classifier))
         (table (dense-classifier-table classifier))
         (n-class (dense-classifier-n-class classifier))
         (roots (packed-topology-roots topology))
         (n-tree (packed-topology-n-tree topology)))
    (declare (type (simple-array single-float (* *)) table)
             (type (simple-array (signed-byte 32) (*)) roots)
             (type fixnum n-class n-tree))
    (dotimes (k n-class) (setf (aref acc k) 0.0))
    (dotimes (tree n-tree)
      (let ((row (topology-leaf topology datamatrix datum-index (aref roots tree))))
        (declare (type fixnum row))
        (dotimes (k n-class) (incf (aref acc k) (aref table row k)))))
    (dotimes (k n-class) (setf (aref acc k) (/ (aref acc k) n-tree)))
    (argmax acc)))

(defun predict-csr (classifier datamatrix datum-index acc)
  "PREDICT-FOREST's answer from a CSR payload. Writes only ACC."
  (declare (optimize (speed 3) (safety 0))
           (type csr-classifier classifier)
           (type (simple-array single-float (* *)) datamatrix)
           (type (simple-array single-float (*)) acc)
           (type fixnum datum-index))
  (let* ((topology (csr-classifier-topology classifier))
         (offsets (csr-classifier-offsets classifier))
         (class (csr-classifier-class classifier))
         (probability (csr-classifier-probability classifier))
         (n-class (csr-classifier-n-class classifier))
         (roots (packed-topology-roots topology))
         (n-tree (packed-topology-n-tree topology)))
    (declare (type (simple-array (unsigned-byte 32) (*)) offsets)
             (type (simple-array (unsigned-byte 16) (*)) class)
             (type (simple-array single-float (*)) probability)
             (type (simple-array (signed-byte 32) (*)) roots)
             (type fixnum n-class n-tree))
    (dotimes (k n-class) (setf (aref acc k) 0.0))
    (dotimes (tree n-tree)
      (let ((leaf (topology-leaf topology datamatrix datum-index (aref roots tree))))
        (declare (type fixnum leaf))
        (loop for i of-type fixnum from (aref offsets leaf) below (aref offsets (1+ leaf))
              do (incf (aref acc (aref class i)) (aref probability i)))))
    (dotimes (k n-class) (setf (aref acc k) (/ (aref acc k) n-tree)))
    (argmax acc)))

;;;; Agreement, on distributions as well as classes

(defun count-payload-disagreements (forest predict-fn n-class datamatrix)
  "Rows where PREDICT-FN's class or distribution differs from the library's.

Returns (values class-differing distribution-differing worst-absolute-difference)."
  (let ((acc (make-array n-class :element-type 'single-float :initial-element 0.0))
        (class-bad 0) (dist-bad 0) (worst 0.0))
    (dotimes (i (array-dimension datamatrix 0) (values class-bad dist-bad worst))
      (let ((reference-class (predict-forest forest datamatrix i))
            (reference-dist (copy-seq (class-distribution-forest forest datamatrix i))))
        (unless (= reference-class (funcall predict-fn datamatrix i acc))
          (incf class-bad))
        (let ((row-bad nil))
          (dotimes (k n-class)
            (let ((d (abs (- (aref reference-dist k) (aref acc k)))))
              (when (> d 0.0) (setf row-bad t))
              (setf worst (max worst d))))
          (when row-bad (incf dist-bad)))))))

(defun classifier-bytes (thing)
  "Bytes in the leaf payload of a dense or CSR classifier."
  (etypecase thing
    (dense-classifier (array-bytes (dense-classifier-table thing) 32))
    (csr-classifier (+ (array-bytes (csr-classifier-offsets thing) 32)
                       (array-bytes (csr-classifier-class thing) 16)
                       (array-bytes (csr-classifier-probability thing) 32)))))

(defun topology-bytes (topology)
  (+ (array-bytes (packed-topology-feature topology) 32)
     (array-bytes (packed-topology-threshold topology) 32)
     (array-bytes (packed-topology-left topology) 32)
     (array-bytes (packed-topology-right topology) 32)
     (array-bytes (packed-topology-roots topology) 32)
     (array-bytes (packed-topology-tree-leaf-offsets topology) 32)))

;;;; dense against CSR

(defun compare-payloads (n-class datamatrix target datamatrix-test n-tree max-depth n-trial)
  "One forest, one topology, both payloads: agreement, bytes and speed."
  (let ((forest (make-forest n-class datamatrix target
                             :n-tree n-tree :bagging-ratio 0.1 :max-depth max-depth
                             :n-trial n-trial :min-region-samples 5
                             ;; The builder refuses a forest whose leaves lost their
                             ;; indices, so ask for them up front.
                             :remove-sample-indices? nil)))
    (multiple-value-bind (topology topology-seconds) (seconds (build-packed-topology forest))
      (multiple-value-bind (dense dense-seconds)
          (seconds (build-dense-classifier forest topology))
        (multiple-value-bind (csr csr-seconds) (seconds (build-csr-classifier forest topology))
          (multiple-value-bind (dense-class dense-dist dense-worst)
              (count-payload-disagreements forest
                                           (lambda (d i acc) (predict-dense dense d i acc))
                                           n-class datamatrix-test)
            (multiple-value-bind (csr-class csr-dist csr-worst)
                (count-payload-disagreements forest
                                             (lambda (d i acc) (predict-csr csr d i acc))
                                             n-class datamatrix-test)
              (let ((dense-acc (make-array n-class :element-type 'single-float
                                                   :initial-element 0.0))
                    (csr-acc (make-array n-class :element-type 'single-float
                                                 :initial-element 0.0)))
                (format t "~&| ~Dx d=~D | ~D/~D/~,1F ~D/~D/~,1F | ~,2F topo ~,2F dense ~,2F csr | ~
~,1F / ~,1F / ~,1F MB | ~,0F / ~,0F | ~,2Fx |~%"
                        n-tree max-depth
                        dense-class dense-dist dense-worst csr-class csr-dist csr-worst
                        topology-seconds dense-seconds csr-seconds
                        (/ (topology-bytes topology) 1048576.0)
                        (/ (classifier-bytes dense) 1048576.0)
                        (/ (classifier-bytes csr) 1048576.0)
                        (time-predictions (lambda (d i) (predict-dense dense d i dense-acc))
                                          datamatrix-test)
                        (time-predictions (lambda (d i) (predict-csr csr d i csr-acc))
                                          datamatrix-test)
                        (/ (time-predictions (lambda (d i) (predict-csr csr d i csr-acc))
                                             datamatrix-test)
                           (time-predictions (lambda (d i) (predict-dense dense d i dense-acc))
                                             datamatrix-test)))
                (force-output)))))))))

(defun run-payload-comparison (&key (dataset :letter) (configs '((500 5) (500 10))))
  "dense against CSR over one topology. Disagreements are class/distribution/worst-diff."
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (format t "~&=== dense vs csr leaf payload on ~A ===~%" dataset)
    (format t "~&| model | dense c/d/w  csr c/d/w | build s | topo / dense / csr MB | dense / csr pred/s | ratio |~%")
    (format t "|---|---|---|---|---|---|~%")
    (force-output)
    (dolist (config configs)
      (compare-payloads n-class datamatrix target datamatrix-test
                        (first config) (second config) n-trial))
    (format t "~&PAYLOAD_DONE~%")
    (force-output)))

;;;; dense against CSR, measured
;;;;
;;;; One forest and one topology per row, both payloads built from it. Disagreements are
;;;; class / distribution / worst absolute difference, against PREDICT-FOREST and
;;;; CLASS-DISTRIBUTION-FOREST. Every one is 0/0/0.0: the payloads are bit-identical to the
;;;; library, not merely equal after ARGMAX.
;;;;
;;;; | | topo MB | dense MB | csr MB | dense pred/s | csr pred/s | ratio |
;;;; |---|---|---|---|---|---|---|
;;;; | letter 500x d=5  | 0.2 |  1.5 | 0.7 | 50933 | 49407 | 0.97x |
;;;; | letter 500x d=10 | 1.7 | 11.3 | 1.8 | 16207 | 24154 | 1.50x |
;;;; | MNIST 500x d=5   | 0.2 |  0.6 | 0.7 | 67113 | 57142 | 0.85x |
;;;; | MNIST 500x d=10  | 4.0 |  9.9 | 4.1 | 19342 | 15823 | 0.82x |
;;;;
;;;; The memory prediction held: letter's leaf table falls 11.3 MB to 1.8 MB at depth 10,
;;;; a 6.3x against the 7.3x the non-zero profile predicted. MNIST's barely moves, 9.9 to
;;;; 4.1, and at depth 5 CSR is *larger* than dense -- with 10 classes, 40 bytes a leaf
;;;; dense against 4 + 5s with s just over 2, there is almost nothing to win.
;;;;
;;;; Speed does not follow memory the way it did for the node arrays. CSR wins where the
;;;; saving is large (1.50x on letter at depth 10) and loses where it is not (0.82-0.85x on
;;;; MNIST), because it trades a straight-line run over n-class contiguous floats for an
;;;; indirect one: two offset loads, then a class index per non-zero to scatter through.
;;;; With 10 classes the dense run is 40 bytes -- under a cache line -- and the indirection
;;;; costs more than the bytes saved.
;;;;
;;;; So the rule is class count, not sparsity. CSR pays when n-class is large enough that a
;;;; dense row spans several cache lines; dense pays when a row is a cache line or less.
;;;; letter's 26 classes are past the crossover and MNIST's 10 are not. A builder choosing
;;;; between them on n-class and the measured mean non-zero count, as the review suggested,
;;;; is the right shape -- but it should choose on the dense row's size rather than on the
;;;; compression ratio, which is what would have picked wrong here.

;;;; Parallel scaling of the separated representations
;;;;
;;;; The array layout reached 3.7-4.9x on eight threads and the guess was that its working
;;;; set -- data every core contends for -- was the limit. Packing cut the node arrays to a
;;;; third, so if that guess was right the scaling should improve. Measured rather than
;;;; assumed, since the last two guesses about this were wrong.

(defun run-packed-parallel-scaling (&key (dataset :letter) (n-tree 500) (max-depth 10)
                                         (worker-counts '(1 2 4 8)))
  "Scaling of dense and CSR payloads over a shared topology, with the old layouts alongside."
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (let* ((forest (make-forest n-class datamatrix target
                                :n-tree n-tree :bagging-ratio 0.1 :max-depth max-depth
                                :n-trial n-trial :min-region-samples 5
                                :remove-sample-indices? nil))
           (topology (build-packed-topology forest))
           (dense (build-dense-classifier forest topology))
           (csr (build-csr-classifier forest topology))
           (old (build-packed-forest forest))
           (n-rows (array-dimension datamatrix-test 0)))
      (format t "~&=== packed parallel scaling on ~A, ~Dx d=~D ===~%" dataset n-tree max-depth)
      (format t "~&| workers | dense pred/s | csr pred/s | packed-forest pred/s |~%")
      (format t "|---|---|---|---|~%")
      (force-output)
      (flet ((rate (predict-fn)
               (lambda (w)
                 (time-parallel
                  (lambda (start end)
                    (let ((acc (make-array n-class :element-type 'single-float
                                                   :initial-element 0.0))
                          (sum 0))
                      (loop for i from start below end
                            do (incf sum (funcall predict-fn datamatrix-test i acc)))
                      sum))
                  n-rows w))))
        (let ((dense-rate (rate (lambda (d i acc) (predict-dense dense d i acc))))
              (csr-rate (rate (lambda (d i acc) (predict-csr csr d i acc))))
              (old-rate (rate (lambda (d i acc) (predict-packed-forest old d i acc)))))
          (dolist (w worker-counts)
            (format t "| ~D | ~,0F | ~,0F | ~,0F |~%"
                    w (funcall dense-rate w) (funcall csr-rate w) (funcall old-rate w))
            (force-output))))
      (format t "~&PACKED_PARALLEL_DONE~%")
      (force-output))))

;;;; Batch prediction: datum-major against tree-major
;;;;
;;;; Everything so far predicts one datum at a time, which walks all N trees per datum --
;;;; so each tree's node arrays are revisited once per datum and nothing stays hot. The
;;;; other order is to take a tile of data and push it through one tree before moving to
;;;; the next, so a tree's nodes are touched once per tile.
;;;;
;;;; The cost is memory: accumulating across trees means one accumulator per datum in the
;;;; tile, tile x n-class floats, live for the whole tile. That is the trade -- locality
;;;; over working set -- and which way it goes depends on whether a tree's nodes are
;;;; smaller than the tile's accumulators.

(defun predict-dense-tile-datum-major (classifier datamatrix start end acc out)
  "Predict rows [START,END) a datum at a time. OUT is indexed from 0."
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array single-float (*)) acc)
           (type (simple-array fixnum (*)) out)
           (type fixnum start end))
  (loop for i of-type fixnum from start below end
        do (setf (aref out (- i start))
                 (predict-dense classifier datamatrix i acc))))

(defun predict-dense-tile-tree-major (classifier datamatrix start end accs out)
  "Predict rows [START,END) a tree at a time, accumulating into per-datum rows of ACCS.

ACCS is (tile x n-class). A tree's feature, threshold and child arrays are walked for every
datum in the tile before the next tree is touched."
  (declare (optimize (speed 3) (safety 0))
           (type dense-classifier classifier)
           (type (simple-array single-float (* *)) datamatrix accs)
           (type (simple-array fixnum (*)) out)
           (type fixnum start end))
  (let* ((topology (dense-classifier-topology classifier))
         (table (dense-classifier-table classifier))
         (n-class (dense-classifier-n-class classifier))
         (roots (packed-topology-roots topology))
         (n-tree (packed-topology-n-tree topology))
         (tile (- end start)))
    (declare (type (simple-array single-float (* *)) table)
             (type (simple-array (signed-byte 32) (*)) roots)
             (type fixnum n-class n-tree tile))
    (dotimes (r tile)
      (dotimes (k n-class) (setf (aref accs r k) 0.0)))
    (dotimes (tree n-tree)
      (let ((root (aref roots tree)))
        (dotimes (r tile)
          (let ((row (topology-leaf topology datamatrix (+ start r) root)))
            (declare (type fixnum row))
            (dotimes (k n-class)
              (incf (aref accs r k) (aref table row k)))))))
    (dotimes (r tile)
      (let ((best most-negative-single-float)
            (best-k 0))
        (declare (type single-float best) (type fixnum best-k))
        (dotimes (k n-class)
          (let ((v (/ (aref accs r k) n-tree)))
            (when (> v best) (setf best v best-k k))))
        (setf (aref out r) best-k)))))

(defun time-batch (fn n-rows tile &key (minimum-seconds 0.5d0))
  "Predictions per second for FN, which takes (start end) and predicts that tile."
  (let ((passes 0)
        (start-time (get-internal-real-time))
        (elapsed 0d0))
    (loop
      (let ((i 0))
        (loop while (< i n-rows)
              do (let ((end (min n-rows (+ i tile))))
                   (funcall fn i end)
                   (setf i end))))
      (incf passes)
      (setf elapsed (/ (float (- (get-internal-real-time) start-time) 1.0d0)
                       internal-time-units-per-second))
      (when (>= elapsed minimum-seconds) (return)))
    (/ (* n-rows passes) elapsed)))

(defun compare-batch-orders (n-class datamatrix target datamatrix-test n-tree max-depth
                             n-trial tiles)
  "One forest: datum-major against tree-major at several tile sizes, and agreement."
  (let* ((forest (make-forest n-class datamatrix target
                              :n-tree n-tree :bagging-ratio 0.1 :max-depth max-depth
                              :n-trial n-trial :min-region-samples 5
                              :remove-sample-indices? nil))
         (dense (build-dense-classifier forest))
         (n-rows (array-dimension datamatrix-test 0)))
    ;; Agreement first: tree-major must give exactly what the library gives.
    (let* ((tile (min 256 n-rows))
           (accs (make-array (list tile n-class) :element-type 'single-float))
           (out (make-array tile :element-type 'fixnum))
           (bad 0))
      (let ((i 0))
        (loop while (< i n-rows)
              do (let ((end (min n-rows (+ i tile))))
                   (predict-dense-tile-tree-major dense datamatrix-test i end accs out)
                   (loop for j from i below end
                         do (unless (= (aref out (- j i))
                                       (predict-forest forest datamatrix-test j))
                              (incf bad)))
                   (setf i end))))
      (format t "~&| ~Dx d=~D | tree-major disagreements ~D of ~D |~%"
              n-tree max-depth bad n-rows))
    (dolist (tile tiles)
      (let* ((tile (min tile n-rows))
             (acc (make-array n-class :element-type 'single-float :initial-element 0.0))
             (accs (make-array (list tile n-class) :element-type 'single-float))
             (out (make-array tile :element-type 'fixnum))
             (datum-rate (time-batch (lambda (s e)
                                       (predict-dense-tile-datum-major dense datamatrix-test
                                                                       s e acc out))
                                     n-rows tile))
             (tree-rate (time-batch (lambda (s e)
                                      (predict-dense-tile-tree-major dense datamatrix-test
                                                                     s e accs out))
                                    n-rows tile)))
        (format t "~&| tile ~D | ~,0F datum-major | ~,0F tree-major | ~,2Fx | accs ~,2F MB |~%"
                tile datum-rate tree-rate (/ tree-rate datum-rate)
                (/ (* tile n-class 4) 1048576.0))
        (force-output)))))

(defun run-batch-comparison (&key (dataset :letter) (n-tree 500) (max-depth 10)
                                  (tiles '(1 32 256 5000)))
  (multiple-value-bind (datamatrix datamatrix-test target n-class n-trial)
      (dataset-parts dataset)
    (format t "~&=== batch order on ~A, ~Dx d=~D ===~%" dataset n-tree max-depth)
    (force-output)
    (compare-batch-orders n-class datamatrix target datamatrix-test
                          n-tree max-depth n-trial tiles)
    (format t "~&BATCH_DONE~%")
    (force-output)))

;;;; Parallel scaling and batch order, measured
;;;;
;;;; 500-tree depth-10 forests, one accumulator per thread.
;;;;
;;;; | workers | letter dense | letter csr | MNIST dense | MNIST csr |
;;;; |---|---|---|---|---|
;;;; | 1 |  15577 |  24001 | 18554 | 16557 |
;;;; | 2 |  35028 |  47895 | 38988 | 34015 |
;;;; | 4 |  64460 |  90914 | 71052 | 66339 |
;;;; | 8 | 124290 | 153855 | 91079 | 83199 |
;;;;
;;;; letter reaches 8.0x on eight threads for dense and 6.4x for CSR, against the array
;;;; layout's 4.9x measured earlier -- so the guess that its working set was the limit was
;;;; right, and cutting the node arrays to a third fixed it. MNIST does not: 4.9x for
;;;; dense, and the curve flattens between four and eight threads (71052 to 91079) where
;;;; letter's is still nearly linear. MNIST's topology is 4.0 MB against letter's 1.7, and
;;;; its leaf table 9.9 against 11.3 -- so the working set that still does not fit is the
;;;; one CSR shrank on letter and could not on MNIST.
;;;;
;;;; Batch order, dense payload, tree-major against datum-major:
;;;;
;;;; | tile | letter datum | letter tree | ratio | MNIST datum | MNIST tree | ratio |
;;;; |---|---|---|---|---|---|---|
;;;; | 1     | 15106 | 13388 | 0.89x | 18798 | 14948 | 0.80x |
;;;; | 32    | 15315 | 18316 | 1.20x | 18869 | 23867 | 1.26x |
;;;; | 256   | 15106 | 29184 | 1.93x | 18869 | 32950 | 1.75x |
;;;; | whole | 15198 | 35463 | 2.33x | 18869 | 18976 | 1.01x |
;;;;
;;;; Tree-major agrees exactly with PREDICT-FOREST at every tile: 0 of 5000 and 0 of 10000.
;;;;
;;;; A tile of one is tree-major at its worst -- all of the accumulator handling, none of
;;;; the reuse -- and it duly loses. From 32 up it wins, because a tree's node arrays are
;;;; walked once per tile instead of once per datum. letter keeps improving to the whole
;;;; test set, 2.33x. MNIST peaks at 256 and then collapses back to parity: 10000 x 10
;;;; accumulators is 0.38 MB being swept once per tree, 500 times, and past some tile size
;;;; that costs more than the tree reuse saves. letter's 5000 x 26 is 0.50 MB and does not
;;;; hit it, which says the limit is not accumulator size alone but its size against the
;;;; tree being reused.
;;;;
;;;; Tiling is the largest single-thread gain left in this file, and it composes with the
;;;; parallel result rather than competing: chunks are already per-thread, so a chunk is a
;;;; tile. Combining them was not measured.
