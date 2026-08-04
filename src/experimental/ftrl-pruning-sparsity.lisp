;;; -*- coding:utf-8; mode:lisp -*-
;;;
;;; Does FTRL-Proximal's L1 sparsity make global pruning easier?
;;;
;;; Load with the test fixture system on the path, then LOAD this file:
;;;
;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;   (load "src/experimental/ftrl-pruning-sparsity.lisp")
;;;   (run-letter-sweep)
;;;
;;; Not part of any system. See docs/superpowers/specs/2026-08-04-*.md for the design.

;;; Not :CL-RANDOM-FOREST, the way src/experimental/multi-grained-scanning.lisp does it.
;;; That facade re-exports only the exported symbols, and MAKE-L2-NORM,
;;; COLLECT-LEAF-PARENT, CHILDREN-L2-NORM and DTREE-MAX-LEAF-INDEX -- everything the
;;; pruning criterion is made of -- are internal. Interning them fresh in another
;;; package is exactly the bug CLAUDE.md's "Known broken code" section describes.
(in-package :cl-random-forest/src/random-forest)

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
