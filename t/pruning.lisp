(in-package :cl-user)

(defpackage cl-random-forest-test/pruning
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture)
  (:import-from #:cl-random-forest/src/random-forest
                #:dtree-max-leaf-index
                #:collect-leaf-parent
                #:do-leaf
                #:node-sample-indices
                #:dtree-root))
(in-package :cl-random-forest-test/pruning)

(defun leaf-count (forest)
  "Total number of leaves across FOREST.
Deliberately not FOREST-N-LEAF: that slot is written only at construction, so it
goes stale the moment anything prunes the forest (issue #15)."
  (reduce #'+ (mapcar #'dtree-max-leaf-index (forest-dtree-list forest))))

(defun trained-forest-and-learner (&key (remove-sample-indices? t))
  "Return (values forest refine-learner) trained on the synthetic classification set."
  (multiple-value-bind (datamatrix target) (synthetic-classification-train)
    (let* ((forest (make-forest +synthetic-n-class+ datamatrix target
                                :n-tree 50 :bagging-ratio 0.3
                                :max-depth 8 :n-trial 10
                                :remove-sample-indices? remove-sample-indices?))
           (refine-dataset (make-refine-dataset forest datamatrix))
           (learner (make-refine-learner forest)))
      (dotimes (epoch 5)
        (train-refine-learner learner refine-dataset target))
      (values forest learner))))

(defun leaves-without-sample-indices (forest)
  "Count the leaves in FOREST whose NODE-SAMPLE-INDICES is NIL.
NODE-CLASS-DISTRIBUTION recomputes a leaf's distribution from those indices on
every prediction, so a leaf without them cannot answer a query."
  (let ((count 0))
    (dolist (dtree (forest-dtree-list forest))
      (do-leaf (lambda (node)
                 (unless (node-sample-indices node) (incf count)))
        (dtree-root dtree)))
    count))

(deftest pruning-reduces-leaf-count
  (multiple-value-bind (forest learner) (trained-forest-and-learner)
    (let* ((rate 0.2)
           (n-parents (length (collect-leaf-parent forest)))
           (expected (floor (* n-parents rate)))
           (before (leaf-count forest)))
      (pruning! forest learner rate)
      (let* ((after (leaf-count forest))
             (deleted (- before after)))
        (ok (< after before)
            (format nil "leaf count fell ~D -> ~D" before after))
        ;; pruning-rate is the fraction of LEAF-PARENT nodes to prune, not of
        ;; leaves. Each pruned parent loses its two children and becomes a leaf
        ;; itself, so exactly one leaf goes per parent. The README says otherwise
        ;; -- see issue #18. Verified exact at rates 0.1, 0.2 and 0.5.
        ;;
        ;; Exact because no leaf-parent here sits at depth 0: pruning! skips any
        ;; candidate shallower than min-depth (default 1), i.e. a root that split
        ;; once and stopped. Measured 0 such parents across 14 forest builds at
        ;; these fixture settings; lowering :max-depth could change that.
        (ok (= deleted expected)
            (format nil "deleted ~D leaves = floor(~D leaf-parents * ~A) = ~D"
                    deleted n-parents rate expected))))))

(deftest pruning-preserves-refine-accuracy
  (multiple-value-bind (datamatrix target) (synthetic-classification-train)
    (multiple-value-bind (datamatrix-test target-test) (synthetic-classification-test)
      (multiple-value-bind (forest learner) (trained-forest-and-learner)
        (let ((before (test-refine-learner learner
                                           (make-refine-dataset forest datamatrix-test)
                                           target-test :quiet-p t)))
          (pruning! forest learner 0.2)
          ;; Pruning changes the leaf index space, so both the refine dataset and
          ;; the learner must be rebuilt before re-training (the README procedure).
          (let ((refine-train (make-refine-dataset forest datamatrix))
                (refine-test (make-refine-dataset forest datamatrix-test))
                (relearner (make-refine-learner forest)))
            (dotimes (epoch 5)
              (train-refine-learner relearner refine-train target))
            (let ((after (test-refine-learner relearner refine-test target-test
                                              :quiet-p t)))
              ;; Measured +0.33 and -0.17 points; 5.0 is a generous guard.
              (ok (< (abs (- after before)) 5.0)
                  (format nil "refine accuracy ~,4F -> ~,4F after pruning and re-learning"
                          before after)))))))))

(deftest pruning-keeps-forest-usable-when-sample-indices-retained
  (multiple-value-bind (datamatrix-test target-test) (synthetic-classification-test)
    (multiple-value-bind (forest learner)
        (trained-forest-and-learner :remove-sample-indices? nil)
      (let ((before (test-forest forest datamatrix-test target-test :quiet-p t)))
        (pruning! forest learner 0.2)
        (let ((after (test-forest forest datamatrix-test target-test :quiet-p t)))
          (ok (numberp after)
              "test-forest still returns an accuracy after pruning")
          (ok (< (abs (- after before)) 5.0)
              (format nil "forest accuracy ~,4F -> ~,4F after pruning" before after))
          ;; The accuracy comparison alone has teeth only where a stranded leaf
          ;; signals. SBCL checks class-distribution's (simple-array fixnum)
          ;; declaration and does; CCL does not, and would return a quietly wrong
          ;; number instead. Assert the mechanism too.
          (ok (zerop (leaves-without-sample-indices forest))
              "no leaf was stranded, because the forest kept its sample indices"))))))

(deftest pruning-strands-leaves-without-sample-indices
  ;; KNOWN BUG (issue #14): set-best-children! nils node-sample-indices on every
  ;; node it splits when :remove-sample-indices? is t -- the default -- and
  ;; delete-children! turns such a node back into a leaf without restoring them.
  ;; Every pruned parent therefore becomes a leaf that cannot answer a query.
  ;;
  ;; How that surfaces is implementation-dependent: SBCL checks the
  ;; (simple-array fixnum) declaration in class-distribution and signals a
  ;; type-error, while CCL does not check it, quietly returns a zero
  ;; distribution, and gives a wrong answer instead of a loud one. So this test
  ;; asserts the mechanism rather than either symptom.
  ;;
  ;; This test pins the current behaviour -- when it starts FAILING, the bug has
  ;; been fixed and this test should be replaced by a positive assertion.
  (multiple-value-bind (forest learner) (trained-forest-and-learner) ; default flags
    (let* ((rate 0.2)
           (n-parents (length (collect-leaf-parent forest)))
           (expected (floor (* n-parents rate))))
      (ok (zerop (leaves-without-sample-indices forest))
          "before pruning, every leaf carries its sample indices")
      (pruning! forest learner rate)
      (let ((stranded (leaves-without-sample-indices forest)))
        ;; Exact because no leaf-parent here sits at depth 0: pruning! skips any
        ;; candidate shallower than min-depth (default 1), i.e. a root that split
        ;; once and stopped. Measured 0 such parents across 14 forest builds at
        ;; these fixture settings; lowering :max-depth could change that.
        (ok (= stranded expected)
            (format nil "after pruning, ~D leaves have no sample indices = the ~D parents pruned"
                    stranded expected))))))

(deftest pruning-does-not-update-forest-n-leaf
  ;; KNOWN BUG (issue #15): forest-n-leaf is written only by make-forest and
  ;; make-regression-forest. pruning! calls set-leaf-index-forest!, which updates
  ;; every dtree-max-leaf-index but leaves the cached total alone. This test pins
  ;; the current behaviour -- when it starts FAILING, the bug has been fixed and
  ;; this test should be replaced by a positive assertion.
  (multiple-value-bind (forest learner) (trained-forest-and-learner)
    (let ((slot-before (forest-n-leaf forest))
          (true-before (leaf-count forest)))
      (pruning! forest learner 0.2)
      (let ((slot-after (forest-n-leaf forest))
            (true-after (leaf-count forest)))
        (ok (< true-after true-before)
            (format nil "the real leaf count did fall ~D -> ~D" true-before true-after))
        (ok (= slot-after slot-before)
            (format nil "forest-n-leaf stayed at ~D while the real count is ~D"
                    slot-after true-after))))))
