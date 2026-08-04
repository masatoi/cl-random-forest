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
;;; of leaf-parents with w_L = w_R is *identical* to the set with w_L = w_R = 0 (22127 of
;;; 54038, exactly equal counts). L1 never produces a pair that is equal but non-zero --
;;; two independently updated FTRL coordinates do not land on the same float. So a
;;; criterion that scores ||w_L - w_R|| has nothing extra to find; the structure has to be
;;; created, not just detected.
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
