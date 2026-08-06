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

(defun predict-compiled-forest (compiled-forest datamatrix datum-index)
  "PREDICT-FOREST's answer, computed from precompiled trees and cached leaf distributions."
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array single-float (* *)) datamatrix)
           (type fixnum datum-index))
  (let ((n-class (compiled-forest-n-class compiled-forest))
        (n-tree (compiled-forest-n-tree compiled-forest))
        (predictors (compiled-forest-predictors compiled-forest))
        (distributions (compiled-forest-distributions compiled-forest))
        (acc (compiled-forest-scratch compiled-forest)))
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
