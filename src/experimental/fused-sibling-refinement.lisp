;;; -*- coding:utf-8; mode:lisp -*-
;;;
;;; Refinement in a sibling (sum, difference) basis, so that L1 drives sibling leaves
;;; toward each other instead of toward zero.
;;;
;;; Why this basis
;;; --------------
;;; PRUNING! deleting a leaf-parent merges its two leaves into one, so every datum that
;;; reached either child now gets a single weight. The merge is therefore lossless exactly
;;; when w_L = w_R for every class -- "both zero" is only the special case c = 0.
;;;
;;; Measured on letter with the plain one-leaf-per-coordinate encoding: under L1 the set
;;; of leaf-parents with w_L = w_R is *identical* to the set with w_L = w_R = 0. From
;;; RUN-PLAIN-SIBLING-CHECK at the bottom of this file, lambda1 10.0, seed 42:
;;;
;;;   (:N-LEAF-PARENT 54177 :BOTH-ZERO 22201 :ALL-EQUAL 22201 :EQUAL-BUT-NOT-ZERO 0)
;;;
;;; L1 never produces a pair that is equal but non-zero -- two independently updated FTRL
;;; coordinates do not land on the same float. So a criterion that scores ||w_L - w_R||
;;; has nothing extra to find; the structure has to be created, not just detected.
;;;
;;; This file creates it. For a leaf-parent p with children L and R, replace their two
;;; coordinates with
;;;
;;;   s_p = (w_L + w_R) / 2      d_p = (w_L - w_R) / 2
;;;
;;; and encode a datum reaching L as (s_p: +1, d_p: +1), reaching R as (s_p: +1, d_p: -1).
;;; Then s + d = w_L and s - d = w_R, so the model class is unchanged -- this is a change
;;; of basis, not a different model. But now L1 acts on d directly, and d_p = 0 across all
;;; classes means exactly "this split is useless, merge it", whatever s is.
;;;
;;; A leaf whose sibling is an internal node belongs to no pair and keeps a single
;;; coordinate. Since each leaf-parent owns exactly two leaves and contributes exactly two
;;; coordinates, the dimension is unchanged: 2*n_pairs + n_unpaired = n_leaves.
;;;
;;; Nothing in cl-online-learning changes. SPARSE-LR+FTRL-UPDATE already reads the sparse
;;; vector's value vector, so the +-1 entries work as they stand; only cl-random-forest's
;;; refine encoding, which hardwires a shared all-1.0 value array, had to be replaced.
;;;
;;; Not part of any system. Load by hand:
;;;
;;;   (ql:quickload :cl-random-forest-test/fixture)
;;;   (setf lparallel:*kernel* (lparallel:make-kernel 4))
;;;   (load "src/experimental/fused-sibling-refinement.lisp")
;;;   (in-package :cl-random-forest/src/experimental/fused-sibling-refinement)
;;;   (run-fused-letter)

(defpackage :cl-random-forest/src/experimental/fused-sibling-refinement
  (:use #:cl
        #:cl-random-forest)
  (:import-from #:cl-random-forest/src/random-forest
                #:find-leaf
                #:dtree-root
                #:dtree-id
                #:dtree-max-leaf-index
                #:node-leaf-index
                #:node-left-node
                #:node-right-node
                #:node-dtree
                #:node-depth
                #:collect-leaf-parent
                #:delete-children!
                #:set-leaf-index-forest!
                #:make-l2-norm
                #:children-l2-norm)
  (:import-from #:cl-random-forest/src/utils
                #:dotimes/pdotimes
                #:read-data))

(in-package :cl-random-forest/src/experimental/fused-sibling-refinement)

;;;; The basis

(defstruct (reparam-map (:constructor %make-reparam-map))
  "Maps each tree's leaf indices onto the (sum, difference) coordinate space.

S-INDEX, D-INDEX and D-SIGN are one array per tree, indexed by that tree's leaf index.
D-INDEX is -1 and D-SIGN 0.0 for a leaf with no leaf sibling. PARENTS is in
COLLECT-LEAF-PARENT order and PARENT-S-INDEX / PARENT-D-INDEX are aligned with it, so a
pruning criterion can go from a leaf-parent straight to the coordinate that says whether
it is mergeable."
  dimension s-index d-index d-sign parents parent-s-index parent-d-index)

(defun make-reparam-map (forest)
  "Assign (sum, difference) coordinates to FOREST's leaves.
The resulting dimension equals the plain leaf-index space's, since each leaf-parent
trades its two leaves for one sum and one difference coordinate."
  (let* ((dtrees (forest-dtree-list forest))
         (n-tree (length dtrees))
         (parents (collect-leaf-parent forest))
         (n-parent (length parents))
         (s-index (make-array n-tree))
         (d-index (make-array n-tree))
         (d-sign (make-array n-tree))
         (parent-s-index (make-array n-parent :element-type 'fixnum :initial-element -1))
         (parent-d-index (make-array n-parent :element-type 'fixnum :initial-element -1))
         (by-tree (make-array n-tree :initial-element nil))
         (global 0))
    ;; Bucket the parents by tree, keeping each one's position in PARENTS so
    ;; PARENT-D-INDEX stays aligned with the list a caller will iterate.
    (loop for node in parents
          for position from 0
          do (push (cons position node) (aref by-tree (dtree-id (node-dtree node)))))
    (loop for dtree in dtrees
          for tree-id from 0
          do ;; MAKE-REFINE-DATASET indexes FOREST-INDEX-OFFSET by DTREE-ID while building
             ;; it from FOREST-DTREE-LIST in order, i.e. it already assumes these agree.
             ;; Assert it rather than inherit the assumption silently.
             (assert (= (dtree-id dtree) tree-id))
             (let* ((n-leaf (dtree-max-leaf-index dtree))
                    (si (make-array n-leaf :element-type 'fixnum :initial-element -1))
                    (di (make-array n-leaf :element-type 'fixnum :initial-element -1))
                    (sg (make-array n-leaf :element-type 'single-float :initial-element 0.0)))
               (loop for (position . node) in (aref by-tree tree-id)
                     do (let ((left (node-leaf-index (node-left-node node)))
                              (right (node-leaf-index (node-right-node node)))
                              (s global)
                              (d (1+ global)))
                          (incf global 2)
                          (setf (aref si left) s
                                (aref si right) s
                                (aref di left) d
                                (aref di right) d
                                (aref sg left) 1.0
                                (aref sg right) -1.0
                                (aref parent-s-index position) s
                                (aref parent-d-index position) d)))
               (dotimes (i n-leaf)
                 (when (minusp (aref si i))
                   (setf (aref si i) global)
                   (incf global)))
               (setf (svref s-index tree-id) si
                     (svref d-index tree-id) di
                     (svref d-sign tree-id) sg)))
    (assert (= global (reduce #'+ (mapcar #'dtree-max-leaf-index dtrees))))
    (%make-reparam-map :dimension global
                       :s-index s-index
                       :d-index d-index
                       :d-sign d-sign
                       :parents parents
                       :parent-s-index parent-s-index
                       :parent-d-index parent-d-index)))

;;;; Encoding

(defun reparam-sparse-vector (forest map datamatrix datum-index)
  "The (sum, difference) sparse vector for one datum: one or two entries per tree."
  (let* ((dtrees (forest-dtree-list forest))
         (n-tree (length dtrees))
         (index (make-array (* 2 n-tree) :element-type 'fixnum))
         (value (make-array (* 2 n-tree) :element-type 'single-float))
         (k 0))
    (loop for dtree in dtrees
          for tree-id from 0
          do (let* ((leaf (node-leaf-index (find-leaf (dtree-root dtree) datamatrix datum-index)))
                    (si (svref (reparam-map-s-index map) tree-id))
                    (di (svref (reparam-map-d-index map) tree-id))
                    (sg (svref (reparam-map-d-sign map) tree-id)))
               (setf (aref index k) (aref si leaf)
                     (aref value k) 1.0)
               (incf k)
               (when (>= (aref di leaf) 0)
                 (setf (aref index k) (aref di leaf)
                       (aref value k) (aref sg leaf))
                 (incf k))))
    (clol.vector:make-sparse-vector (subseq index 0 k) (subseq value 0 k))))

(defun make-reparam-dataset (forest map datamatrix)
  "A simple-vector of sparse vectors, one per row of DATAMATRIX.
Unlike MAKE-REFINE-DATASET this has to keep values, not just indices, so it stores whole
sparse vectors rather than one shared index array per datum."
  (let* ((len (array-dimension datamatrix 0))
         (dataset (make-array len)))
    (dotimes/pdotimes (i len)
      (setf (svref dataset i) (reparam-sparse-vector forest map datamatrix i)))
    dataset))

;;;; Training and testing
;;;;
;;;; TRAIN-REFINE-LEARNER cannot be reused: it rebuilds a sparse vector per class from a
;;;; shared all-1.0 value array and swaps only the index vector in. Here each datum owns
;;;; its own values, so the loop hands the learner the stored vector directly.

(defun train-fused-learner (learner dataset target)
  "One epoch over DATASET. Parallel over classes, as TRAIN-REFINE-LEARNER-MULTICLASS is."
  (let ((n-class (clol::one-vs-rest-n-class learner))
        (learners (clol::one-vs-rest-learners-vector learner))
        (update (clol::one-vs-rest-learner-update learner))
        (len (length dataset)))
    (dotimes/pdotimes (class-id n-class)
      (loop for i fixnum from 0 below len
            do (funcall update
                        (svref learners class-id)
                        (svref dataset i)
                        (if (= (aref target i) class-id) 1.0 -1.0))))))

(defun test-fused-learner (learner dataset target)
  "Top-1 accuracy in percent."
  (let* ((n-class (clol::one-vs-rest-n-class learner))
         (learners (clol::one-vs-rest-learners-vector learner))
         (activate (clol::one-vs-rest-learner-activate learner))
         (weight-of (clol::one-vs-rest-learner-weight learner))
         (bias-of (clol::one-vs-rest-learner-bias learner))
         (len (length dataset))
         (n-correct 0))
    (loop for i fixnum from 0 below len
          do (let ((best most-negative-single-float)
                   (best-class 0))
               (dotimes (k n-class)
                 (let ((activation (funcall activate
                                            (svref dataset i)
                                            (funcall weight-of (svref learners k))
                                            (funcall bias-of (svref learners k)))))
                   (when (> activation best)
                     (setf best activation
                           best-class k))))
               (when (= best-class (aref target i))
                 (incf n-correct))))
    (* 100.0 (/ (float n-correct) len))))

(defun train-fused-epochs (learner dataset target test-dataset test-target epochs)
  "Train for EPOCHS epochs, returning the per-epoch test accuracy list."
  (loop repeat epochs
        do (train-fused-learner learner dataset target)
        collect (test-fused-learner learner test-dataset test-target)))

;;;; Verification
;;;;
;;;; The encoding is where this can break silently: a swapped sign or a stale offset still
;;;; trains and still reports a plausible accuracy. So check the algebra directly rather
;;;; than inferring correctness from a number that looks reasonable.

(defun verify-encoding (forest map datamatrix &key (n-check 300))
  "Signal an error unless the reparametrised encoding scores identically to the plain one.

Takes an arbitrary deterministic weight vector W in the plain leaf-index space, maps it to
the (sum, difference) basis by s = (w_L + w_R)/2 and d = (w_L - w_R)/2, and checks that
<x_reparam, w_reparam> equals sum-over-trees w[leaf + offset] for N-CHECK data. Any index,
offset or sign error breaks this identity."
  (let* ((dimension (reparam-map-dimension map))
         (plain (make-array dimension :element-type 'single-float))
         (reparam (make-array dimension :element-type 'single-float :initial-element 0.0))
         (index-offset (forest-index-offset forest))
         (dtrees (forest-dtree-list forest))
         (worst 0.0))
    ;; An arbitrary but reproducible spread of magnitudes and signs.
    (dotimes (i dimension)
      (setf (aref plain i) (- (mod (* 37 (1+ i)) 101) 50.0)))
    ;; Paired coordinates, straight from the parent nodes.
    (loop for node in (reparam-map-parents map)
          for position from 0
          do (let* ((offset (aref index-offset (dtree-id (node-dtree node))))
                    (wl (aref plain (+ (node-leaf-index (node-left-node node)) offset)))
                    (wr (aref plain (+ (node-leaf-index (node-right-node node)) offset))))
               (setf (aref reparam (aref (reparam-map-parent-s-index map) position))
                     (* 0.5 (+ wl wr))
                     (aref reparam (aref (reparam-map-parent-d-index map) position))
                     (* 0.5 (- wl wr)))))
    ;; Leaves with no leaf sibling keep their single coordinate unchanged.
    (loop for dtree in dtrees
          for tree-id from 0
          do (let ((offset (aref index-offset tree-id))
                   (si (svref (reparam-map-s-index map) tree-id))
                   (di (svref (reparam-map-d-index map) tree-id)))
               (dotimes (leaf (dtree-max-leaf-index dtree))
                 (when (minusp (aref di leaf))
                   (setf (aref reparam (aref si leaf)) (aref plain (+ leaf offset)))))))
    (dotimes (datum (min n-check (array-dimension datamatrix 0)))
      (let ((plain-score 0.0)
            (reparam-score 0.0))
        (loop for dtree in dtrees
              for tree-id from 0
              do (let ((leaf (node-leaf-index
                              (find-leaf (dtree-root dtree) datamatrix datum))))
                   (incf plain-score (aref plain (+ leaf (aref index-offset tree-id))))))
        (let ((sv (reparam-sparse-vector forest map datamatrix datum)))
          (dotimes (k (clol.vector:sparse-vector-length sv))
            (incf reparam-score
                  (* (aref (clol.vector:sparse-vector-value-vector sv) k)
                     (aref reparam (aref (clol.vector:sparse-vector-index-vector sv) k))))))
        (setf worst (max worst (abs (- plain-score reparam-score))))))
    (assert (< worst 0.05) (worst)
            "Reparametrised encoding disagrees with the plain one by ~,6F" worst)
    worst))

;;;; What the basis bought

(defun fused-report (map learner)
  "Classify every leaf-parent by whether its difference and sum coordinates are zero.

MERGEABLE counts leaf-parents whose difference coordinate is exactly zero in every class:
w_L = w_R, so deleting the split changes no prediction. MERGEABLE-AND-INFORMATIVE counts
the subset whose sum coordinate is still non-zero -- pairs that carry real weight yet are
free to merge. Those are the structure the plain encoding could not produce at all."
  (let* ((parent-s-index (reparam-map-parent-s-index map))
         (parent-d-index (reparam-map-parent-d-index map))
         (n-parent (length parent-d-index))
         (n-class (clol::one-vs-rest-n-class learner))
         (learners (clol::one-vs-rest-learners-vector learner))
         (weight-of (clol::one-vs-rest-learner-weight learner))
         (weights (make-array n-class))
         (mergeable 0)
         (mergeable-and-informative 0)
         (all-zero 0)
         (n-element 0)
         (n-element-zero 0))
    (dotimes (k n-class)
      (setf (svref weights k) (funcall weight-of (svref learners k))))
    (dotimes (position n-parent)
      (let ((d (aref parent-d-index position))
            (s (aref parent-s-index position))
            (d-zero t)
            (s-zero t))
        (dotimes (k n-class)
          (let ((w (svref weights k)))
            (unless (zerop (aref w d)) (setf d-zero nil))
            (unless (zerop (aref w s)) (setf s-zero nil))))
        (when d-zero
          (incf mergeable)
          (if s-zero (incf all-zero) (incf mergeable-and-informative)))))
    (dotimes (k n-class)
      (let ((w (svref weights k)))
        (dotimes (j (length w))
          (incf n-element)
          (when (zerop (aref w j)) (incf n-element-zero)))))
    (list :n-leaf-parent n-parent
          :mergeable-rate (/ (float mergeable) n-parent)
          :mergeable-and-informative-rate (/ (float mergeable-and-informative) n-parent)
          :both-zero-rate (/ (float all-zero) n-parent)
          :element-zero-rate (/ (float n-element-zero) n-element))))

;;;; Pruning on the difference coordinate

(defun prune-mergeable! (forest map learner &optional (min-depth 1))
  "Delete every leaf-parent whose difference coordinate is zero in all classes.

No pruning rate: the criterion supplies its own cutoff, which is the point of the basis.
Mutates FOREST and re-runs SET-LEAF-INDEX-FOREST!, so the caller must rebuild the map, the
dataset and the learner afterwards. Returns the number of leaf-parents deleted."
  (let* ((parents (reparam-map-parents map))
         (parent-d-index (reparam-map-parent-d-index map))
         (n-class (clol::one-vs-rest-n-class learner))
         (learners (clol::one-vs-rest-learners-vector learner))
         (weight-of (clol::one-vs-rest-learner-weight learner))
         (weights (make-array n-class))
         (deleted 0))
    (dotimes (k n-class)
      (setf (svref weights k) (funcall weight-of (svref learners k))))
    (loop for node in parents
          for position from 0
          do (let ((d (aref parent-d-index position))
                   (d-zero t))
               (dotimes (k n-class)
                 (unless (zerop (aref (svref weights k) d)) (setf d-zero nil)))
               (when (and d-zero (>= (node-depth node) min-depth))
                 (delete-children! node)
                 (incf deleted))))
    (set-leaf-index-forest! forest)
    deleted))

(defun leaf-count (forest)
  "Total leaves across FOREST. Not FOREST-N-LEAF, which goes stale on pruning (issue #15)."
  (reduce #'+ (mapcar #'dtree-max-leaf-index (forest-dtree-list forest))))

;;;; Driver

(defparameter *epochs* 20
  "Fixed epoch count, matching src/experimental/ftrl-pruning-sparsity.lisp.
TRAIN-REFINE-LEARNER-PROCESS is unusable: its rollback keeps a shallow struct copy sharing
the weight arrays, so it always returns the last epoch (issue #20).")

(defun letter-forest ()
  "Return (values forest datamatrix datamatrix-test train-target test-target).
Settings from example/classification/letter.lisp plus :remove-sample-indices? nil, which
pruning needs (issue #14)."
  (multiple-value-bind (datamatrix target) (cl-random-forest-test/fixture:letter-train)
    (multiple-value-bind (datamatrix-test target-test) (cl-random-forest-test/fixture:letter-test)
      (values (make-forest cl-random-forest-test/fixture:+letter-n-class+ datamatrix target
                           :n-tree 500 :bagging-ratio 0.1
                           :min-region-samples 5 :n-trial 10 :max-depth 15
                           :remove-sample-indices? nil)
              datamatrix datamatrix-test target target-test))))

(defun run-fused-letter (&optional (lambda1-list '(3.0 10.0 30.0)))
  "Train in the sibling basis on letter at each lambda1, then prune the mergeable set.

Prints, per lambda1: accuracy, how many leaf-parents the difference coordinate says are
mergeable, how many of those still carry weight, and what pruning exactly that set costs."
  (multiple-value-bind (forest datamatrix datamatrix-test train-target test-target)
      (letter-forest)
    (let ((map (make-reparam-map forest)))
      (format t "~&encoding check: worst |plain - reparam| = ~,8F over 300 data~%"
              (verify-encoding forest map datamatrix))
      (format t "~&dimension ~D, leaf-parents ~D, leaves ~D~%"
              (reparam-map-dimension map)
              (length (reparam-map-parents map))
              (leaf-count forest))
      (force-output)
      (let ((train (make-reparam-dataset forest map datamatrix))
            (test (make-reparam-dataset forest map datamatrix-test)))
        (dolist (lambda1 lambda1-list)
          (let* ((learner (make-refine-learner-of-type
                           forest 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0))
                 (curve (train-fused-epochs learner train train-target
                                            test test-target *epochs*))
                 (report (fused-report map learner)))
            (format t "~&FUSED lambda1 ~,1F accuracy ~,2F ~S~%" lambda1 (car (last curve)) report)
            (format t "~&  curve ~{~,2F ~}~%" curve)
            (force-output)
            ;; Prune exactly the mergeable set on a private copy of the forest, then
            ;; rebuild everything the new leaf index space invalidates and retrain.
            (multiple-value-bind (forest2 dm2 dmt2 tg2 tgt2) (letter-forest)
              (declare (ignore dm2 dmt2))
              (let* ((map2 (make-reparam-map forest2))
                     (train2 (make-reparam-dataset forest2 map2 datamatrix))
                     (test2 (make-reparam-dataset forest2 map2 datamatrix-test))
                     (learner2 (make-refine-learner-of-type
                                forest2 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0))
                     (before (car (last (train-fused-epochs learner2 train2 tg2
                                                            test2 tgt2 *epochs*))))
                     (leaves-before (leaf-count forest2))
                     (n-parent (length (reparam-map-parents map2)))
                     (deleted (prune-mergeable! forest2 map2 learner2)))
                (let* ((map3 (make-reparam-map forest2))
                       (train3 (make-reparam-dataset forest2 map3 datamatrix))
                       (test3 (make-reparam-dataset forest2 map3 datamatrix-test))
                       (learner3 (make-refine-learner-of-type
                                  forest2 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0))
                       (after (car (last (train-fused-epochs learner3 train3 tg2
                                                             test3 tgt2 *epochs*)))))
                  (format t "~&PRUNE lambda1 ~,1F deleted ~D/~D leaf-parents (~,1F%) ~
accuracy ~,2F -> ~,2F leaves ~D -> ~D~%"
                          lambda1 deleted n-parent (* 100.0 (/ (float deleted) n-parent))
                          before after leaves-before (leaf-count forest2))
                  (force-output))))))))
    (format t "~&FUSED_DONE~%")
    (force-output)))

(defun mnist-forest ()
  "Return (values forest datamatrix datamatrix-test train-target test-target).

Forest settings are MNIST-FOREST's from example/classification/mnist.lisp -- the forest
that file's PRUNING! calls operate on -- plus :remove-sample-indices? nil.

READ-DATA subtracts 1 from every LIBSVM label, which is right for 1-based label files.
MNIST's labels already start at 0, so they arrive as -1..8 and have to be shifted back,
exactly as example/classification/mnist.lisp does inline."
  (let ((dir cl-random-forest-test/fixture:*dataset-dir*))
    (multiple-value-bind (datamatrix target)
        (read-data (merge-pathnames "mnist.scale" dir) 784)
      (multiple-value-bind (datamatrix-test target-test)
          (read-data (merge-pathnames "mnist.scale.t" dir) 784)
        (dotimes (i (length target))
          (incf (aref target i)))
        (dotimes (i (length target-test))
          (incf (aref target-test i)))
        (values (make-forest 10 datamatrix target
                             :n-tree 500 :bagging-ratio 0.1
                             :min-region-samples 5 :n-trial 10 :max-depth 10
                             :remove-sample-indices? nil)
                datamatrix datamatrix-test target target-test)))))

(defun run-fused-mnist (&optional (lambda1-list '(3.0 10.0 30.0)) (prune-lambda1 10.0))
  "Reproduce the letter result on MNIST: 10 classes instead of 26.

Builds the forest once and prunes the very forest the kept learner was trained on, so the
before/after accuracy pair is paired rather than two independent unseeded builds -- which
is the main methodological weakness of RUN-FUSED-LETTER."
  (multiple-value-bind (forest datamatrix datamatrix-test train-target test-target)
      (mnist-forest)
    (format t "~&forest accuracy ~,2F (example/classification/mnist.lisp records 93.38)~%"
            (test-forest forest datamatrix-test test-target :quiet-p t))
    (force-output)
    (let ((map (make-reparam-map forest)))
      (format t "~&encoding check: worst |plain - reparam| = ~,8F over 300 data~%"
              (verify-encoding forest map datamatrix))
      (format t "~&dimension ~D, leaf-parents ~D, leaves ~D~%"
              (reparam-map-dimension map)
              (length (reparam-map-parents map))
              (leaf-count forest))
      (force-output)
      (let ((train (make-reparam-dataset forest map datamatrix))
            (test (make-reparam-dataset forest map datamatrix-test))
            (prune-learner nil)
            (prune-before 0.0))
        (dolist (lambda1 lambda1-list)
          (let* ((learner (make-refine-learner-of-type
                           forest 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0))
                 (curve (train-fused-epochs learner train train-target
                                            test test-target *epochs*)))
            (format t "~&FUSED lambda1 ~,1F accuracy ~,2F ~S~%"
                    lambda1 (car (last curve)) (fused-report map learner))
            (format t "~&  curve ~{~,2F ~}~%" curve)
            (force-output)
            (when (= lambda1 prune-lambda1)
              (setf prune-learner learner
                    prune-before (car (last curve))))))
        (when prune-learner
          (let ((n-parent (length (reparam-map-parents map)))
                (leaves-before (leaf-count forest)))
            (let ((deleted (prune-mergeable! forest map prune-learner)))
              ;; Release the pre-pruning datasets before building their replacements; on
              ;; MNIST each one is hundreds of megabytes of sparse vectors.
              (setf train nil test nil)
              (let* ((map2 (make-reparam-map forest))
                     (train2 (make-reparam-dataset forest map2 datamatrix))
                     (test2 (make-reparam-dataset forest map2 datamatrix-test))
                     (learner2 (make-refine-learner-of-type
                                forest 'clol::sparse-lr+ftrl 0.1 1.0 prune-lambda1 1.0))
                     (after (car (last (train-fused-epochs learner2 train2 train-target
                                                           test2 test-target *epochs*)))))
                (format t "~&PRUNE lambda1 ~,1F deleted ~D/~D (~,1F%) ~
accuracy ~,2F -> ~,2F leaves ~D -> ~D~%"
                        prune-lambda1 deleted n-parent
                        (* 100.0 (/ (float deleted) n-parent))
                        prune-before after leaves-before (leaf-count forest))
                (format t "~&forest accuracy after pruning ~,2F~%"
                        (test-forest forest datamatrix-test test-target :quiet-p t))
                (force-output)))))))
    (format t "~&FUSED_MNIST_DONE~%")
    (force-output)))


;;;; Measured on letter, 500 trees, max-depth 15, 20 epochs, 4-worker lparallel kernel.
;;;; dimension 157431 = leaf count 157431 (the basis change is dimension-preserving, as
;;;; it must be), 53631 leaf-parents. Encoding check: worst |plain - reparam| = 0.00000000
;;;; over 300 data.
;;;;
;;;; | lambda1 | accuracy | mergeable | of which informative | both-zero | element-zero |
;;;; |---|---|---|---|---|---|
;;;; | 3.0  | 96.76 | 27.3% | 9.2%  | 18.1% | 88.6% |
;;;; | 10.0 | 96.58 | 49.9% | 10.5% | 39.5% | 96.4% |
;;;; | 30.0 | 96.44 | 70.3% | 9.5%  | 60.9% | 98.8% |
;;;;
;;;; Against the plain one-leaf-per-coordinate encoding measured in
;;;; src/experimental/ftrl-pruning-sparsity.lisp on the same forest configuration:
;;;;
;;;; | lambda1 | plain acc / prunable | fused acc / prunable |
;;;; |---|---|---|
;;;; | 3.0  | 96.80 / 18.3% | 96.76 / 27.3% |
;;;; | 10.0 | 96.80 / 40.9% | 96.58 / 49.9% |
;;;; | 30.0 | 96.28 / 63.4% | 96.44 / 70.3% |
;;;;
;;;; The fused basis makes roughly 9 more percentage points of leaf-parents prunable at
;;;; every lambda1, for an accuracy difference inside run-to-run noise (an unseeded forest
;;;; moves the AROW baseline by +-0.15 between builds; the largest gap here is 0.22 and
;;;; lambda1=30 goes the other way).
;;;;
;;;; Those 9 points are exactly the MERGEABLE-AND-INFORMATIVE column, and the arithmetic
;;;; closes: at lambda1=10, both-zero 39.5% + informative 10.5% = mergeable 49.9%, and the
;;;; both-zero figure matches the plain encoding's 40.9% prunable rate. So the plain
;;;; encoding already found every pair that is zero, and the basis change adds precisely
;;;; the pairs that are equal without being zero -- the ones the plain encoding produced
;;;; exactly zero of, no matter which criterion was used to look for them.
;;;;
;;;; Threshold pruning, no rate parameter, deleting exactly the mergeable set:
;;;;
;;;;   lambda1 3.0  deleted 14838/54077 (27.4%)  accuracy 97.00 -> 97.02  leaves 160778 -> 145940
;;;;   lambda1 10.0 deleted 27010/54150 (49.9%)  accuracy 96.76 -> 96.92  leaves 160574 -> 133564
;;;;   lambda1 30.0 deleted 37898/53783 (70.5%)  accuracy 96.64 -> 96.50  leaves 158580 -> 120682
;;;;
;;;; The before/after pairs come from their own freshly built forests, so the "before"
;;;; column differs from the table above for the usual unseeded-bagging reason.
;;;;
;;;; This also confirms the S/D labelling is not swapped, which VERIFY-ENCODING alone
;;;; cannot rule out: a consistent relabelling would still satisfy the algebraic identity,
;;;; but then PRUNE-MERGEABLE! would be deleting pairs with w_L = -w_R -- large opposite
;;;; weights -- and accuracy would collapse rather than hold flat.
;;;;
;;;; Convergence: lambda1=30 was still gaining (+0.02 on epoch 20, curve 83.48 -> 96.44),
;;;; so its accuracy is a slight underestimate, the same effect the plain encoding showed.
;;;; A stronger L1 delays learning because |z| must cross lambda1 before a weight moves at
;;;; all.

;;;; Measured on MNIST, 500 trees, max-depth 10, 20 epochs, 8-worker lparallel kernel.
;;;; Forest accuracy 93.27 against the 93.38 example/classification/mnist.lisp records, so
;;;; the labels and dimension are right. dimension 251521 = leaf count 251521, 99047
;;;; leaf-parents. Encoding check: worst |plain - reparam| = 0.00000000 over 300 data.
;;;;
;;;; | lambda1 | accuracy | mergeable | of which informative | both-zero | element-zero |
;;;; |---|---|---|---|---|---|
;;;; | 3.0  | 97.98 | 42.6% | 13.2% | 29.4% | 87.1% |
;;;; | 10.0 | 97.97 | 68.1% | 11.7% | 56.3% | 94.4% |
;;;; | 30.0 | 97.79 | 82.0% | 8.7%  | 73.3% | 97.2% |
;;;;
;;;; Against the plain encoding's MNIST numbers in
;;;; src/experimental/ftrl-pruning-sparsity.lisp (AROW baseline there: 98.27 / 0.0%):
;;;;
;;;; | lambda1 | plain acc / prunable | fused acc / prunable | prunable gained |
;;;; |---|---|---|---|
;;;; | 3.0  | 98.08 / 25.5% | 97.98 / 42.6% | +17.1pt |
;;;; | 10.0 | 97.93 / 52.2% | 97.97 / 68.1% | +15.9pt |
;;;; | 30.0 | 97.76 / 71.0% | 97.79 / 82.0% | +11.0pt |
;;;;
;;;; The letter result reproduces and strengthens: the accuracy differences (-0.10, +0.04,
;;;; +0.03) are inside run-to-run noise and go both ways, while the prunable fraction gains
;;;; 11 to 17 points rather than letter's ~9. The BOTH-ZERO column tracks the plain
;;;; encoding's prunable rate as it did on letter (29.4/56.3/73.3 here against
;;;; 25.5/52.2/71.0 there, two independently built forests), so once again the whole gain
;;;; is the MERGEABLE-AND-INFORMATIVE column -- pairs that are equal without being zero.
;;;;
;;;; Ten classes make that column larger than 26 did (13.2% at lambda1=3 against letter's
;;;; 9.2%), the same direction as the plain encoding's group-sparsity result: needing
;;;; agreement across fewer classes makes an all-class condition easier to satisfy.
;;;;
;;;; Threshold pruning at lambda1 10.0, no rate parameter. Unlike RUN-FUSED-LETTER this is
;;;; paired -- one forest, pruned with the learner trained on it -- so the before/after
;;;; difference is not confounded by unseeded bagging:
;;;;
;;;;   deleted 67404/99047 leaf-parents (68.1%)
;;;;   refined accuracy 97.97 -> 97.98
;;;;   leaves 251521 -> 184117 (-26.8%)
;;;;   raw forest accuracy 93.27 -> 92.97
;;;;
;;;; Deleting two thirds of the leaf-parents costs the refined model nothing, which is what
;;;; the criterion optimises for; the raw forest, which nothing here optimises for, gives up
;;;; 0.30 points.

;;;; Iterated pruning
;;;;
;;;; One pass can only ever reach leaf-parents, i.e. nodes whose *both* children are
;;;; leaves. Deleting one turns it into a leaf itself, so its own parent may now have two
;;;; leaf children and become a candidate. Repeating therefore peels the forest a layer at
;;;; a time, and because the criterion carries its own cutoff the loop needs no schedule
;;;; and no pruning rate -- it stops when nothing is mergeable any more.

(defun run-iterated-pruning (forest datamatrix datamatrix-test train-target test-target
                             &key (lambda1 10.0) (max-rounds 20))
  "Prune, rebuild, retrain, repeat until no leaf-parent is mergeable or MAX-ROUNDS.

Each round trains a fresh learner on the current forest, records its accuracy, then
deletes every leaf-parent whose difference coordinate is zero in all classes. The accuracy
in a row is therefore the accuracy of the forest that row's pruning was chosen from.
FOREST is mutated. Prints one row per round and returns the rows in order."
  (let ((rows '())
        (exhausted t))
    (loop for round from 1 to max-rounds
          do (let* ((map (make-reparam-map forest))
                    (train (make-reparam-dataset forest map datamatrix))
                    (test (make-reparam-dataset forest map datamatrix-test))
                    (learner (make-refine-learner-of-type
                              forest 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0))
                    (accuracy (car (last (train-fused-epochs learner train train-target
                                                             test test-target *epochs*))))
                    (n-parent (length (reparam-map-parents map)))
                    (leaves (leaf-count forest))
                    (deleted (prune-mergeable! forest map learner)))
               (push (list :round round :accuracy accuracy :leaves leaves
                           :leaf-parents n-parent :deleted deleted
                           :leaves-after (leaf-count forest))
                     rows)
               (format t "~&ROUND ~2D accuracy ~,2F leaves ~D leaf-parents ~D ~
deleted ~D (~,1F%) leaves -> ~D~%"
                       round accuracy leaves n-parent deleted
                       (* 100.0 (/ (float deleted) (max 1 n-parent)))
                       (leaf-count forest))
               (force-output)
               (when (zerop deleted)
                 (setf exhausted nil)
                 (return))))
    ;; Exiting on MAX-ROUNDS leaves the last round's pruning unmeasured, since a row's
    ;; accuracy is recorded before its own deletions. Measure the final forest.
    (when exhausted
      (let* ((map (make-reparam-map forest))
             (train (make-reparam-dataset forest map datamatrix))
             (test (make-reparam-dataset forest map datamatrix-test))
             (learner (make-refine-learner-of-type
                       forest 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0))
             (accuracy (car (last (train-fused-epochs learner train train-target
                                                      test test-target *epochs*)))))
        (push (list :round :final :accuracy accuracy :leaves (leaf-count forest)
                    :leaf-parents (length (reparam-map-parents map)) :deleted 0
                    :leaves-after (leaf-count forest))
              rows)
        (format t "~&ROUND final accuracy ~,2F leaves ~D (hit max-rounds, not exhausted)~%"
                accuracy (leaf-count forest))
        (force-output)))
    (nreverse rows)))

(defun run-iterated-letter (&key (lambda1 10.0) (max-rounds 20))
  "Iterated pruning on letter."
  (multiple-value-bind (forest datamatrix datamatrix-test train-target test-target)
      (letter-forest)
    (format t "~&letter, lambda1 ~,1F, forest accuracy ~,2F~%"
            lambda1 (test-forest forest datamatrix-test test-target :quiet-p t))
    (force-output)
    (prog1 (run-iterated-pruning forest datamatrix datamatrix-test
                                 train-target test-target
                                 :lambda1 lambda1 :max-rounds max-rounds)
      (format t "~&raw forest accuracy after all rounds ~,2F~%"
              (test-forest forest datamatrix-test test-target :quiet-p t))
      (format t "~&ITERATED_DONE~%")
      (force-output))))

(defun run-iterated-mnist (&key (lambda1 10.0) (max-rounds 20))
  "Iterated pruning on MNIST."
  (multiple-value-bind (forest datamatrix datamatrix-test train-target test-target)
      (mnist-forest)
    (format t "~&MNIST, lambda1 ~,1F, forest accuracy ~,2F~%"
            lambda1 (test-forest forest datamatrix-test test-target :quiet-p t))
    (force-output)
    (prog1 (run-iterated-pruning forest datamatrix datamatrix-test
                                 train-target test-target
                                 :lambda1 lambda1 :max-rounds max-rounds)
      (format t "~&raw forest accuracy after all rounds ~,2F~%"
              (test-forest forest datamatrix-test test-target :quiet-p t))
      (format t "~&ITERATED_DONE~%")
      (force-output))))

;;;; Matched comparison against rate-based pruning
;;;;
;;;; The threshold criterion picks its own number of deletions, so comparing it with
;;;; PRUNING!'s fixed quantile means choosing what to hold constant. Holding the *rate*
;;;; constant compares two different amounts of pruning; holding the *count* constant
;;;; compares two different choices of which nodes to delete, which is the actual question.
;;;;
;;;; So: run the fused loop, take its per-round deletion counts as a schedule, and make the
;;;; rate-based runs delete exactly the same number each round. All three trajectories then
;;;; pass through identical forest sizes and differ only in which leaf-parents they picked.
;;;; They also start from a bit-identical forest, which needs a seeded *RANDOM-STATE* and a
;;;; serial kernel -- MAKE-FOREST's bagging is parallelised and each lparallel worker has
;;;; its own random state, so a seed alone does not determine the result.

(defun prune-by-l2-norm-count! (forest learner count &optional (min-depth 1))
  "Delete the COUNT lowest-scoring leaf-parents under the incumbent CHILDREN-L2-NORM.

PRUNING! with the number supplied instead of a rate. Like PRUNING!, a candidate shallower
than MIN-DEPTH is skipped without extending the search, so the deletions actually made can
fall short of COUNT. Returns how many were made."
  (let* ((parents (collect-leaf-parent forest))
         (l2 (make-l2-norm learner))
         (indexed (make-array (length parents)))
         (deleted 0))
    (loop for node in parents
          for i from 0
          do (setf (aref indexed i) (cons (children-l2-norm node l2 forest) node)))
    (let ((sorted (sort indexed #'< :key #'car)))
      (loop for i from 0 below (min count (length sorted))
            do (let ((node (cdr (aref sorted i))))
                 (when (>= (node-depth node) min-depth)
                   (delete-children! node)
                   (incf deleted)))))
    (set-leaf-index-forest! forest)
    deleted))

(defun run-plain-iterated-by-counts (forest datamatrix datamatrix-test
                                     train-target test-target counts
                                     &key learner-args (label :plain))
  "Iterated rate-based pruning in the plain encoding, on a supplied deletion schedule.

LEARNER-ARGS is NIL for the incumbent AROW refine learner, or a (type . params) list for
MAKE-REFINE-LEARNER-OF-TYPE. A row's accuracy is the accuracy of the forest that row's
pruning was chosen from, matching RUN-ITERATED-PRUNING. FOREST is mutated."
  (let ((rows '()))
    (loop for count in counts
          for round from 1
          do (let* ((refine-train (make-refine-dataset forest datamatrix))
                    (refine-test (make-refine-dataset forest datamatrix-test))
                    (learner (if learner-args
                                 (apply #'make-refine-learner-of-type forest learner-args)
                                 (make-refine-learner forest)))
                    (leaves (leaf-count forest)))
               (dotimes (epoch *epochs*)
                 (train-refine-learner learner refine-train train-target))
               (let ((accuracy (test-refine-learner learner refine-test test-target
                                                    :quiet-p t)))
                 (let ((deleted (prune-by-l2-norm-count! forest learner count)))
                   (push (list :round round :accuracy accuracy :leaves leaves
                               :deleted deleted :leaves-after (leaf-count forest))
                         rows)
                   (format t "~&~A ROUND ~2D accuracy ~,2F leaves ~D deleted ~D leaves -> ~D~%"
                           label round accuracy leaves deleted (leaf-count forest))
                   (force-output)))))
    ;; The last round's pruning is unmeasured, since a row records accuracy before its own
    ;; deletions. Measure the final forest so the trajectories end comparably.
    (let* ((refine-train (make-refine-dataset forest datamatrix))
           (refine-test (make-refine-dataset forest datamatrix-test))
           (learner (if learner-args
                        (apply #'make-refine-learner-of-type forest learner-args)
                        (make-refine-learner forest))))
      (dotimes (epoch *epochs*)
        (train-refine-learner learner refine-train train-target))
      (let ((accuracy (test-refine-learner learner refine-test test-target :quiet-p t)))
        (push (list :round :final :accuracy accuracy :leaves (leaf-count forest)
                    :deleted 0 :leaves-after (leaf-count forest))
              rows)
        (format t "~&~A ROUND final accuracy ~,2F leaves ~D~%" label accuracy (leaf-count forest))
        (force-output)))
    (nreverse rows)))

(defvar *mnist-cache* nil)

(defun mnist-data ()
  "Return (values datamatrix datamatrix-test train-target test-target), read once.
The label shift is READ-DATA's off-by-one against 0-based LIBSVM labels, as in
example/classification/mnist.lisp."
  (unless *mnist-cache*
    (let ((dir cl-random-forest-test/fixture:*dataset-dir*))
      (multiple-value-bind (datamatrix target)
          (read-data (merge-pathnames "mnist.scale" dir) 784)
        (multiple-value-bind (datamatrix-test target-test)
            (read-data (merge-pathnames "mnist.scale.t" dir) 784)
          (dotimes (i (length target)) (incf (aref target i)))
          (dotimes (i (length target-test)) (incf (aref target-test i)))
          (setf *mnist-cache* (list datamatrix datamatrix-test target target-test))))))
  (values-list *mnist-cache*))

(defun build-seeded-forest (dataset datamatrix target seed)
  "Build DATASET's forest reproducibly. SBCL-specific; src/experimental/ is not in CI."
  (let ((lparallel:*kernel* nil))
    (setf *random-state* (sb-ext:seed-random-state seed))
    (ecase dataset
      (:letter (make-forest cl-random-forest-test/fixture:+letter-n-class+ datamatrix target
                            :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5
                            :n-trial 10 :max-depth 15 :remove-sample-indices? nil))
      (:mnist (make-forest 10 datamatrix target
                           :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5
                           :n-trial 10 :max-depth 10 :remove-sample-indices? nil)))))

(defun run-matched-comparison (dataset &key (lambda1 10.0) (max-rounds 20) (seed 42))
  "Compare threshold pruning in the fused basis against rate-based pruning at matched size.

Runs three trajectories from the same seeded forest: the fused loop, which chooses both
which nodes to delete and how many; then the incumbent CHILDREN-L2-NORM criterion under
FTRL and under AROW, each held to the fused loop's per-round deletion counts. Prints the
three accuracy columns against a shared leaf count."
  (multiple-value-bind (datamatrix datamatrix-test train-target test-target)
      (ecase dataset
        (:letter (multiple-value-bind (dm tg) (cl-random-forest-test/fixture:letter-train)
                   (multiple-value-bind (dmt tgt) (cl-random-forest-test/fixture:letter-test)
                     (values dm dmt tg tgt))))
        (:mnist (mnist-data)))
    (let ((ftrl-args (list 'clol::sparse-lr+ftrl 0.1 1.0 lambda1 1.0)))
      (format t "~&=== ~A, lambda1 ~,1F, seed ~D ===~%" dataset lambda1 seed)
      (force-output)
      (let* ((fused-rows
               (let ((forest (build-seeded-forest dataset datamatrix train-target seed)))
                 (format t "~&FUSED trajectory, forest accuracy ~,2F~%"
                         (test-forest forest datamatrix-test test-target :quiet-p t))
                 (force-output)
                 (run-iterated-pruning forest datamatrix datamatrix-test
                                       train-target test-target
                                       :lambda1 lambda1 :max-rounds max-rounds)))
             (counts (remove 0 (mapcar (lambda (row) (getf row :deleted)) fused-rows)))
             (ftrl-rows
               (let ((forest (build-seeded-forest dataset datamatrix train-target seed)))
                 (run-plain-iterated-by-counts forest datamatrix datamatrix-test
                                               train-target test-target counts
                                               :learner-args ftrl-args :label :plain-ftrl)))
             (arow-rows
               (let ((forest (build-seeded-forest dataset datamatrix train-target seed)))
                 (run-plain-iterated-by-counts forest datamatrix datamatrix-test
                                               train-target test-target counts
                                               :learner-args nil :label :plain-arow))))
        (format t "~&~%| round | fused leaves | fused acc | plain+FTRL leaves | plain+FTRL acc | ~
plain+AROW leaves | plain+AROW acc |~%")
        (format t "|---|---|---|---|---|---|---|~%")
        (loop for i from 0 below (max (length fused-rows) (length ftrl-rows) (length arow-rows))
              do (let ((f (nth i fused-rows))
                       (p (nth i ftrl-rows))
                       (a (nth i arow-rows)))
                   (format t "| ~A | ~@[~D~] | ~@[~,2F~] | ~@[~D~] | ~@[~,2F~] | ~@[~D~] | ~@[~,2F~] |~%"
                           (if f (getf f :round) (if p (getf p :round) (getf a :round)))
                           (and f (getf f :leaves)) (and f (getf f :accuracy))
                           (and p (getf p :leaves)) (and p (getf p :accuracy))
                           (and a (getf a :leaves)) (and a (getf a :accuracy)))))
        (format t "~&MATCHED_DONE~%")
        (force-output)
        (list :fused fused-rows :plain-ftrl ftrl-rows :plain-arow arow-rows)))))

;;;; The hybrid: FTRL picks the structure, AROW fits the final model
;;;;
;;;; RUN-MATCHED-COMPARISON found that at equal forest size the pruning criterion does not
;;;; move accuracy measurably, while the learner does -- AROW sits about 0.4 points above
;;;; either FTRL variant throughout. So the two halves of the problem want different tools.
;;;; FTRL's exact zeros in the fused basis supply a stopping point with no pruning rate to
;;;; choose; AROW supplies the better linear model. Use FTRL to decide the structure, throw
;;;; it away, and refit with AROW on what is left.
;;;;
;;;; What this leaves untested elsewhere: the matched comparison only ever had one learner
;;;; both choosing the pruning and being scored on it. Here FTRL chooses and AROW is
;;;; scored, so whether a structure selected as redundant-for-FTRL is still good for AROW
;;;; is exactly the question, not an inference.

(defun train-plain-arow (forest datamatrix datamatrix-test train-target test-target)
  "Train the incumbent AROW refine learner on FOREST for *EPOCHS* epochs, return accuracy."
  (let ((refine-train (make-refine-dataset forest datamatrix))
        (refine-test (make-refine-dataset forest datamatrix-test))
        (learner (make-refine-learner forest)))
    (dotimes (epoch *epochs*)
      (train-refine-learner learner refine-train train-target))
    (test-refine-learner learner refine-test test-target :quiet-p t)))

(defun run-hybrid (dataset &key (lambda1 10.0) (max-rounds 20) (seed 42))
  "Prune with FTRL in the fused basis until it stops, then refit with AROW.

Reports AROW on the unpruned forest, the fused loop's own final accuracy, and AROW on the
pruned forest -- so the hybrid can be read against both the size it started from and the
learner it replaced."
  (multiple-value-bind (datamatrix datamatrix-test train-target test-target)
      (ecase dataset
        (:letter (multiple-value-bind (dm tg) (cl-random-forest-test/fixture:letter-train)
                   (multiple-value-bind (dmt tgt) (cl-random-forest-test/fixture:letter-test)
                     (values dm dmt tg tgt))))
        (:mnist (mnist-data)))
    (let ((forest (build-seeded-forest dataset datamatrix train-target seed)))
      (format t "~&=== HYBRID ~A, lambda1 ~,1F, seed ~D ===~%" dataset lambda1 seed)
      (let ((leaves-before (leaf-count forest)))
        (format t "~&AROW on the unpruned forest: ~,2F at ~D leaves~%"
                (train-plain-arow forest datamatrix datamatrix-test train-target test-target)
                leaves-before)
        (force-output)
        (let* ((rows (run-iterated-pruning forest datamatrix datamatrix-test
                                           train-target test-target
                                           :lambda1 lambda1 :max-rounds max-rounds))
               (fused-final (getf (car (last rows)) :accuracy))
               (leaves-after (leaf-count forest))
               (arow-final (train-plain-arow forest datamatrix datamatrix-test
                                             train-target test-target)))
          (format t "~&FTRL fused, its own final accuracy: ~,2F at ~D leaves~%"
                  fused-final leaves-after)
          (format t "~&AROW refitted on the pruned forest: ~,2F at ~D leaves (~,1F% of original)~%"
                  arow-final leaves-after
                  (* 100.0 (/ (float leaves-after) leaves-before)))
          (format t "~&raw forest accuracy after pruning ~,2F~%"
                  (test-forest forest datamatrix-test test-target :quiet-p t))
          (format t "~&HYBRID_DONE~%")
          (force-output)
          (list :leaves-before leaves-before :leaves-after leaves-after
                :fused-final fused-final :arow-final arow-final))))))

;;;; Iterated pruning, matched comparison, and the hybrid: measurements
;;;;
;;;; All at lambda1 10.0, *EPOCHS* 20. The iterated runs used unseeded parallel forest
;;;; builds; the matched and hybrid runs used seed 42 with a serial build, so within each
;;;; of those the three trajectories start from a bit-identical forest.
;;;;
;;;; 1. Iterated pruning, self-terminating
;;;; ------------------------------------
;;;; Deletion counts fall off geometrically and the loop stops on its own when nothing is
;;;; mergeable. No pruning rate and no round budget were supplied.
;;;;
;;;;   letter, 16 rounds: leaves 156700 -> 109769 (-29.9%), accuracy 96.90 -> 96.82
;;;;     deleted per round 26836 11258 4749 2084 961 478 232 114 86 52 37 23 13 6 2 0
;;;;     raw forest accuracy 90.96 -> 90.36
;;;;   MNIST, 19 rounds: leaves 249435 -> 128361 (-48.5%), accuracy 97.89 -> 97.92
;;;;     deleted per round 66858 31241 13219 5128 2192 1014 518 353 226 131 73 39 30 26
;;;;                       17 5 3 1 0
;;;;     raw forest accuracy 93.32 -> 92.63
;;;;
;;;; Iterating roughly doubles what a single pass reaches (letter -16.8% -> -29.9%), and
;;;; accuracy is flat across every round rather than decaying.
;;;;
;;;; 2. Matched comparison: does the criterion matter?
;;;; ------------------------------------------------
;;;; Same forest, same per-round deletion counts, so the only difference is which
;;;; leaf-parents get chosen. First and last rounds:
;;;;
;;;;   letter, 14 rounds, leaves 160733 -> 113007
;;;;     fused       96.78 -> 96.82
;;;;     plain+FTRL  96.80 -> 96.84
;;;;     plain+AROW  97.14 -> 97.24
;;;;   MNIST, 17 rounds, leaves 250626 -> 129080
;;;;     fused       97.76 -> 97.89
;;;;     plain+FTRL  97.83 -> 97.85
;;;;     plain+AROW  98.22 -> 98.20
;;;;
;;;; The answer is no. At equal size the fused and plain criteria are indistinguishable
;;;; (within +-0.06 on both datasets, sign varying), while AROW sits 0.35-0.40 above both
;;;; FTRL variants at every size. That gap is exactly FTRL's known accuracy cost against
;;;; AROW, not an effect of the pruning criterion. Accuracy is flat along each trajectory,
;;;; so no criterion "breaks later" than another either.
;;;;
;;;; So the fused basis does not buy accuracy. What it buys is the schedule: it decides how
;;;; far to prune without a rate, and stops.
;;;;
;;;; 3. The hybrid
;;;; -------------
;;;; Which suggests splitting the job. FTRL in the fused basis chooses the structure, then
;;;; AROW is fitted on what is left:
;;;;
;;;;   letter  AROW unpruned          97.14 at 160733 leaves
;;;;           FTRL fused, own model  96.82 at 113007
;;;;           AROW refitted          97.18 at 113007 (70.3% of the leaves)
;;;;           raw forest 91.54 -> 90.84
;;;;   MNIST   AROW unpruned          98.22 at 250626 leaves
;;;;           FTRL fused, own model  97.89 at 129080
;;;;           AROW refitted          98.16 at 129080 (51.5% of the leaves)
;;;;           raw forest 93.31 -> 92.69
;;;;
;;;; FTRL's 0.33-0.36 deficit is an artefact of the search, not of the result: refitting
;;;; recovers it entirely. letter ends *above* its unpruned baseline (97.18 vs 97.14) at
;;;; 70% of the size; MNIST gives up 0.06 for half the size.
;;;;
;;;; This also settles what the matched comparison could not. There, one learner always
;;;; both chose the pruning and was scored on it, so "is a structure selected as
;;;; redundant-for-FTRL still good for AROW?" was open. It is: the hybrid lands within 0.06
;;;; and 0.04 of the structure AROW picked for itself (97.24 on letter, 98.20 on MNIST).

;;;; The observation this basis is a response to
;;;;
;;;; The header claims that under the plain encoding L1 never produces a sibling pair that
;;;; is equal without being zero, which is why a cheaper criterion change -- scoring
;;;; ||w_L - w_R|| instead of ||w_L||^2 + ||w_R||^2 -- cannot help, and the basis has to
;;;; change instead. That claim is load-bearing, so here is the measurement behind it.

(defun plain-sibling-zero-counts (forest learner)
  "Count, for a learner trained in the PLAIN leaf-index encoding, how many leaf-parents
have both children all-class-zero and how many merely have them all-class-equal.

The first is the set CHILDREN-L2-NORM scores zero. The second is the set a mergeability
criterion would score zero, and it always contains the first. Under L1 the two come out
identical, which is the finding: there is nothing for a difference-based criterion to add."
  (let* ((parents (collect-leaf-parent forest))
         (n-class (clol::one-vs-rest-n-class learner))
         (learners (clol::one-vs-rest-learners-vector learner))
         (weight-of (clol::one-vs-rest-learner-weight learner))
         (weights (make-array n-class))
         (index-offset (forest-index-offset forest))
         (both-zero 0)
         (equal-pairs 0))
    (dotimes (k n-class)
      (setf (svref weights k) (funcall weight-of (svref learners k))))
    (dolist (node parents)
      (let* ((offset (aref index-offset (dtree-id (node-dtree node))))
             (left (+ (node-leaf-index (node-left-node node)) offset))
             (right (+ (node-leaf-index (node-right-node node)) offset))
             (all-zero t)
             (all-equal t))
        (dotimes (k n-class)
          (let ((wl (aref (svref weights k) left))
                (wr (aref (svref weights k) right)))
            (unless (and (zerop wl) (zerop wr)) (setf all-zero nil))
            (unless (= wl wr) (setf all-equal nil))))
        (when all-zero (incf both-zero))
        (when all-equal (incf equal-pairs))))
    (list :n-leaf-parent (length parents)
          :both-zero both-zero
          :all-equal equal-pairs
          :equal-but-not-zero (- equal-pairs both-zero))))

(defun run-plain-sibling-check (&key (lambda1 10.0) (seed 42))
  "Reproduce the header's claim on letter, in the plain encoding."
  (multiple-value-bind (datamatrix target) (cl-random-forest-test/fixture:letter-train)
    (multiple-value-bind (datamatrix-test target-test) (cl-random-forest-test/fixture:letter-test)
      (let* ((forest (build-seeded-forest :letter datamatrix target seed))
             (refine-train (make-refine-dataset forest datamatrix))
             (refine-test (make-refine-dataset forest datamatrix-test))
             (learner (make-refine-learner-of-type forest 'clol::sparse-lr+ftrl
                                                   0.1 1.0 lambda1 1.0)))
        (dotimes (epoch *epochs*)
          (train-refine-learner learner refine-train target))
        (format t "~&plain encoding, letter, lambda1 ~,1F, accuracy ~,2F~%  ~S~%"
                lambda1
                (test-refine-learner learner refine-test target-test :quiet-p t)
                (plain-sibling-zero-counts forest learner))
        (format t "~&SIBLING_CHECK_DONE~%")
        (force-output)))))

;;;; Is lambda1 a size dial, or a pruning rate wearing a hat?
;;;;
;;;; The hybrid's selling point is that no pruning rate is chosen. That is only a real
;;;; simplification if lambda1 is easier to set than a rate would be. Two things decide it:
;;;; how widely the final size swings with lambda1, and whether accuracy survives across
;;;; the range. A rate set too high costs accuracy; if AROW refits to full accuracy at
;;;; every lambda1, then lambda1 picks a size without risking correctness, which a rate
;;;; does not.
;;;;
;;;; Caveat to read the high end with: a larger lambda1 delays learning, because |z| has to
;;;; cross it before a weight leaves zero at all. At *EPOCHS* 20 the lambda1 100 runs were
;;;; still climbing. An under-trained round over-prunes -- a coordinate can sit at zero
;;;; because it has not accumulated yet, not because the pair is mergeable -- and iterating
;;;; compounds that. The AROW column is unaffected (AROW converges fast on whatever
;;;; structure it is handed); the FTRL column and the sizes at high lambda1 are not.

(defun run-lambda1-sweep (dataset &key (lambda1-list '(0.0 1.0 3.0 10.0 30.0 100.0))
                                       (max-rounds 20) (seed 42))
  "Run the whole hybrid pipeline at each lambda1 from a bit-identical seeded forest."
  (multiple-value-bind (datamatrix datamatrix-test train-target test-target)
      (ecase dataset
        (:letter (multiple-value-bind (dm tg) (cl-random-forest-test/fixture:letter-train)
                   (multiple-value-bind (dmt tgt) (cl-random-forest-test/fixture:letter-test)
                     (values dm dmt tg tgt))))
        (:mnist (mnist-data)))
    (format t "~&=== LAMBDA1 SWEEP ~A, seed ~D, epochs ~D ===~%" dataset seed *epochs*)
    (let* ((baseline-forest (build-seeded-forest dataset datamatrix train-target seed))
           (baseline-leaves (leaf-count baseline-forest))
           (baseline (train-plain-arow baseline-forest datamatrix datamatrix-test
                                       train-target test-target))
           (rows '()))
      (format t "~&AROW baseline, unpruned: ~,2F at ~D leaves~%" baseline baseline-leaves)
      (force-output)
      (dolist (lambda1 lambda1-list)
        (let ((forest (build-seeded-forest dataset datamatrix train-target seed)))
          (format t "~&--- lambda1 ~,1F ---~%" lambda1)
          (force-output)
          (let* ((iterated (run-iterated-pruning forest datamatrix datamatrix-test
                                                 train-target test-target
                                                 :lambda1 lambda1 :max-rounds max-rounds))
                 (rounds (length iterated))
                 (fused-final (getf (car (last iterated)) :accuracy))
                 (leaves (leaf-count forest))
                 (arow (train-plain-arow forest datamatrix datamatrix-test
                                         train-target test-target)))
            (push (list :lambda1 lambda1 :rounds rounds :leaves leaves
                        :fraction (/ (float leaves) baseline-leaves)
                        :fused fused-final :arow arow)
                  rows)
            (format t "~&SWEEP lambda1 ~,1F rounds ~D leaves ~D (~,1F%) fused ~,2F arow ~,2F~%"
                    lambda1 rounds leaves (* 100.0 (/ (float leaves) baseline-leaves))
                    fused-final arow)
            (force-output))))
      (setf rows (nreverse rows))
      (format t "~&~%| lambda1 | rounds | leaves | % of original | FTRL own | AROW refit | vs baseline |~%")
      (format t "|---|---|---|---|---|---|---|~%")
      (dolist (row rows)
        (format t "| ~,1F | ~D | ~D | ~,1F% | ~,2F | ~,2F | ~,2F |~%"
                (getf row :lambda1) (getf row :rounds) (getf row :leaves)
                (* 100.0 (getf row :fraction)) (getf row :fused) (getf row :arow)
                (- (getf row :arow) baseline)))
      (format t "~&baseline ~,2F at ~D leaves~%" baseline baseline-leaves)
      (format t "~&SWEEP_DONE~%")
      (force-output)
      rows)))

;;;; Which learner should do the final fit?
;;;;
;;;; The hybrid's split -- FTRL chooses the structure, something else fits the model --
;;;; leaves the second half open. AROW is the incumbent, but nothing about the split
;;;; requires it, and cl-online-learning has other confidence-weighted learners. SCW is the
;;;; obvious next one: same family, an explicit aggressiveness parameter C and confidence
;;;; parameter eta where AROW has a single gamma.
;;;;
;;;; Measuring each candidate on the unpruned forest as well as the pruned one separates
;;;; "this learner is better" from "this learner copes with pruning better".

(defun train-plain-final (forest datamatrix datamatrix-test train-target test-target
                          &optional learner-args)
  "Train a final refine learner on FOREST in the plain encoding, return its accuracy.
LEARNER-ARGS is NIL for the incumbent AROW, or a (type . params) list handed to
MAKE-REFINE-LEARNER-OF-TYPE."
  (let ((refine-train (make-refine-dataset forest datamatrix))
        (refine-test (make-refine-dataset forest datamatrix-test))
        (learner (if learner-args
                     (apply #'make-refine-learner-of-type forest learner-args)
                     (make-refine-learner forest))))
    (dotimes (epoch *epochs*)
      (train-refine-learner learner refine-train train-target))
    (test-refine-learner learner refine-test test-target :quiet-p t)))

(defparameter *final-learners*
  '((:arow      . nil)
    (:scw-c0.1  . (clol::sparse-scw 0.9 0.1))
    (:scw-c1.0  . (clol::sparse-scw 0.9 1.0))
    (:scw-c10   . (clol::sparse-scw 0.9 10.0)))
  "Candidate final learners for the hybrid's refit step, as (label . learner-args).
The SCW parameters are (eta C); eta 0.9 is what cl-online-learning's own examples use, and
C sweeps the aggressiveness. NIL means the incumbent AROW at its default gamma.")

(defun run-final-learner-comparison (dataset &key (lambda1-list '(10.0 30.0))
                                                  (max-rounds 20) (seed 42))
  "Compare candidate final learners on the unpruned forest and on FTRL-pruned ones.

One pruning run per lambda1, every candidate measured on each resulting forest, so the
comparison is within a forest rather than across independent builds."
  (multiple-value-bind (datamatrix datamatrix-test train-target test-target)
      (ecase dataset
        (:letter (multiple-value-bind (dm tg) (cl-random-forest-test/fixture:letter-train)
                   (multiple-value-bind (dmt tgt) (cl-random-forest-test/fixture:letter-test)
                     (values dm dmt tg tgt))))
        (:mnist (mnist-data)))
    (format t "~&=== FINAL LEARNER COMPARISON ~A, seed ~D, epochs ~D ===~%"
            dataset seed *epochs*)
    (force-output)
    (let ((rows '())
          (baseline-leaves nil))
      ;; Unpruned reference for every candidate.
      (let ((forest (build-seeded-forest dataset datamatrix train-target seed)))
        (setf baseline-leaves (leaf-count forest))
        (dolist (candidate *final-learners*)
          (let ((accuracy (train-plain-final forest datamatrix datamatrix-test
                                             train-target test-target (cdr candidate))))
            (push (list :lambda1 nil :learner (car candidate)
                        :leaves baseline-leaves :accuracy accuracy)
                  rows)
            (format t "~&UNPRUNED ~A ~,2F at ~D leaves~%"
                    (car candidate) accuracy baseline-leaves)
            (force-output))))
      ;; One pruning run per lambda1, then every candidate on the result.
      (dolist (lambda1 lambda1-list)
        (let ((forest (build-seeded-forest dataset datamatrix train-target seed)))
          (format t "~&--- pruning at lambda1 ~,1F ---~%" lambda1)
          (force-output)
          (run-iterated-pruning forest datamatrix datamatrix-test train-target test-target
                                :lambda1 lambda1 :max-rounds max-rounds)
          (let ((leaves (leaf-count forest)))
            (dolist (candidate *final-learners*)
              (let ((accuracy (train-plain-final forest datamatrix datamatrix-test
                                                 train-target test-target (cdr candidate))))
                (push (list :lambda1 lambda1 :learner (car candidate)
                            :leaves leaves :accuracy accuracy)
                      rows)
                (format t "~&PRUNED lambda1 ~,1F ~A ~,2F at ~D leaves (~,1F%)~%"
                        lambda1 (car candidate) accuracy leaves
                        (* 100.0 (/ (float leaves) baseline-leaves)))
                (force-output))))))
      (setf rows (nreverse rows))
      (format t "~&~%| lambda1 | leaves | ~{~A | ~}~%"
              (mapcar (lambda (c) (string-downcase (symbol-name (car c)))) *final-learners*))
      (format t "|---|---|~{~A~}~%" (mapcar (constantly "---|") *final-learners*))
      (dolist (lambda1 (cons nil lambda1-list))
        (let ((group (remove-if-not (lambda (row) (eql (getf row :lambda1) lambda1)) rows)))
          (when group
            (format t "| ~@[~,1F~]~:[unpruned~;~] | ~D | ~{~,2F | ~}~%"
                    lambda1 lambda1 (getf (car group) :leaves)
                    (mapcar (lambda (row) (getf row :accuracy)) group)))))
      (format t "~&FINAL_LEARNER_DONE~%")
      (force-output)
      rows)))

;;;; lambda1 sweep and final-learner choice: measurements
;;;;
;;;; letter, seed 42, *EPOCHS* 20, one bit-identical forest for every row.
;;;; Read every difference below against the test set size: letter's is 5000, so 0.02
;;;; accuracy points is one test sample.
;;;;
;;;; 1. Is lambda1 a size dial, or a pruning rate wearing a hat?
;;;; ----------------------------------------------------------
;;;; RUN-LAMBDA1-SWEEP, whole hybrid pipeline per lambda1. Baseline: AROW unpruned 97.14
;;;; at 160733 leaves.
;;;;
;;;; | lambda1 | rounds | leaves | % of original | FTRL own | AROW refit | vs baseline |
;;;; |---|---|---|---|---|---|---|
;;;; |   0.0 |  1 | 160733 | 100.0% | 96.90 | 97.14 |  0.00 |
;;;; |   1.0 |  8 | 155013 |  96.4% | 97.02 | 97.14 |  0.00 |
;;;; |   3.0 | 12 | 139097 |  86.5% | 96.80 | 97.20 | +0.06 |
;;;; |  10.0 | 14 | 113007 |  70.3% | 96.82 | 97.18 | +0.04 |
;;;; |  30.0 | 20 |  81777 |  50.9% | 96.66 | 97.24 | +0.10 |
;;;; | 100.0 | 21 |  45780 |  28.5% | 96.02 | 97.00 | -0.14 |
;;;;
;;;; It is a dial: 100% down to 28.5%, a 3.5x range. So "no pruning rate to choose" is
;;;; partly a rename -- how aggressively to prune is still chosen, just in another unit.
;;;;
;;;; What is not a rename is the accuracy column. Across that whole 3.5x range the refit
;;;; accuracy moves 0.24 points total and sits at or above the unpruned baseline at five
;;;; of six settings. Getting lambda1 wrong by a factor of ten (3 to 30) moves the size
;;;; from 86.5% to 50.9% and the accuracy from 97.20 to 97.24 -- up, not down. A rate set
;;;; too high is a mistake; a lambda1 set too high is a smaller forest.
;;;;
;;;; The best accuracy in the table is not the unpruned forest but lambda1 30 at half the
;;;; leaves, which is the CVPR2015 global-pruning claim behaving as advertised: the pruning
;;;; is acting as regularisation.
;;;;
;;;; lambda1 0.0 is the sanity anchor -- no L1, nothing ever mergeable, one round, zero
;;;; deletions, and the refit reproduces the baseline exactly.
;;;;
;;;; Caveat: lambda1 100 did not converge, it hit MAX-ROUNDS 20 while still deleting 31
;;;; leaf-parents in its last round, so 45780 is an upper bound on where it would stop.
;;;; A larger lambda1 also delays learning (|z| must cross it before a weight leaves zero),
;;;; so at 20 epochs its rounds are under-trained, which over-prunes and compounds. Its
;;;; -0.14 is the one row that should not be read as converged.
;;;;
;;;; 2. Does the final learner matter? AROW against SCW
;;;; --------------------------------------------------
;;;; RUN-FINAL-LEARNER-COMPARISON. C first, at eta 0.9:
;;;;
;;;; | lambda1 | leaves | arow | scw C=0.1 | scw C=1.0 | scw C=10 |
;;;; |---|---|---|---|---|---|
;;;; | unpruned | 160733 | 97.14 | 97.08 | 97.08 | 97.08 |
;;;; |     10.0 | 113007 | 97.18 | 97.12 | 97.12 | 97.12 |
;;;; |     30.0 |  81777 | 97.24 | 97.18 | 97.18 | 97.18 |
;;;;
;;;; C over a 100x range changes nothing, and that is correct rather than a bug. C caps
;;;; alpha in SPARSE-SCW-UPDATE, and cl-online-learning's 2026-08-02 fix (restoring
;;;; Proposition 1's 1/(v zeta) factor) shrank the uncapped alpha by roughly v*zeta, so the
;;;; cap no longer binds here. Before that fix it bound on 663 of a1a's 679 updates and was
;;;; effectively setting a constant step size.
;;;;
;;;; eta, the knob that does bind, at C 1.0:
;;;;
;;;; | lambda1 | leaves | arow | eta .6 | eta .75 | eta .9 | eta .95 | eta .99 |
;;;; |---|---|---|---|---|---|---|---|
;;;; | unpruned | 160733 | 97.14 | 97.16 | 97.12 | 97.08 | 97.12 | 97.10 |
;;;; |     30.0 |  81777 | 97.24 | 97.26 | 97.26 | 97.18 | 97.20 | 97.16 |
;;;;
;;;; eta 0.6-0.75 edges AROW out by 0.02, which is one test sample. The full spread across
;;;; AROW and every SCW setting is 97.08 to 97.26, nine samples. There is no reason here to
;;;; prefer either learner. Incidentally eta 0.9, the value cl-online-learning's own
;;;; examples use, is the weakest of the SCW settings for this task.
;;;;
;;;; The one thing the sweep does establish is that the hybrid does not depend on which
;;;; learner finishes it: every learner tested gains from the pruning rather than merely
;;;; tolerating it (AROW 97.14 -> 97.24, SCW eta .6 97.16 -> 97.26, eta .9 97.08 -> 97.18,
;;;; eta .99 97.10 -> 97.16). Halving the leaves is worth +0.06 to +0.10 whoever fits the
;;;; model, which is the regularisation reading surviving a change of learner.
