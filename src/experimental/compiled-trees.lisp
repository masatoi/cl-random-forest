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
