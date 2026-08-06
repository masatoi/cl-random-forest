(in-package :cl-user)

(defpackage cl-random-forest-test/packed
  (:use :cl :rove :cl-random-forest :cl-random-forest/src/packed
        :cl-random-forest-test/fixture)
  (:import-from #:cl-random-forest/src/random-forest
                #:find-leaf
                #:dtree-root
                #:node-leaf-index
                #:node-test-attribute
                #:node-left-node
                #:node-sample-indices))
(in-package :cl-random-forest-test/packed)

(defun synthetic-forest (&key (remove-sample-indices? nil) (n-tree 30))
  "A small forest on the fixture's deterministic classification set."
  (multiple-value-bind (datamatrix target) (synthetic-classification-train)
    (make-forest +synthetic-n-class+ datamatrix target
                 :n-tree n-tree :bagging-ratio 0.3 :max-depth 7 :n-trial 10
                 :remove-sample-indices? remove-sample-indices?)))

(deftest packed-topology-leaf-numbers-are-the-refine-index
  ;; The leaf number a topology reports must be the index Global Refinement uses, so a
  ;; refine dataset can be built straight from it. The spec makes this a guarantee rather
  ;; than a coincidence of two traversal orders agreeing.
  (with-serial-kernel
    (multiple-value-bind (datamatrix target) (synthetic-classification-train)
      (declare (ignore target))
      (let* ((forest (synthetic-forest))
             (topology (build-packed-topology forest))
             (offsets (forest-index-offset forest))
             (out (make-array (forest-n-tree forest) :element-type 'fixnum))
             (checked 0)
             (mismatch 0))
        (dotimes (i 200)
          (packed-leaf-indices topology datamatrix i out)
          (loop for dtree in (forest-dtree-list forest)
                for tree from 0
                do (incf checked)
                   (unless (= (aref out tree)
                              (+ (node-leaf-index
                                  (find-leaf (dtree-root dtree) datamatrix i))
                                 (aref offsets tree)))
                     (incf mismatch))))
        (ok (plusp checked) (format nil "checked ~D (datum, tree) pairs" checked))
        (ok (zerop mismatch)
            (format nil "~D leaf numbers differ from the refine index" mismatch))))))

(deftest packed-topology-refuses-a-forest-with-stranded-leaves
  ;; CLASS-DISTRIBUTION returns a *uniform* distribution for a leaf with no sample indices
  ;; rather than signalling, so a builder that read one would freeze that into the model
  ;; with nothing to show for it.
  ;;
  ;; Pruning is what creates such a leaf, not forest construction. SET-BEST-CHILDREN! nils
  ;; the indices of the node it *splits*, and DELETE-CHILDREN! later turns that node back
  ;; into a leaf without restoring them (issue #14). A leaf is never split, so it keeps its
  ;; indices whatever :remove-sample-indices? says -- which is why this test prunes.
  (with-serial-kernel
    (multiple-value-bind (datamatrix target) (synthetic-classification-train)
      (let* ((forest (make-forest +synthetic-n-class+ datamatrix target
                                  :n-tree 30 :bagging-ratio 0.3 :max-depth 7 :n-trial 10
                                  :remove-sample-indices? t))
             (refine-dataset (make-refine-dataset forest datamatrix))
             (learner (make-refine-learner forest)))
        (ok (packed-topology-p (build-packed-topology forest))
            "before pruning, every leaf still has its indices and the forest packs")
        (dotimes (epoch 3)
          (train-refine-learner learner refine-dataset target))
        (pruning! forest learner 0.3)
        (ok (handler-case (progn (build-packed-topology forest) nil)
              (packed-build-error () t))
            "after pruning, build-packed-topology signals on the stranded leaves")))))

(deftest packed-topology-refuses-a-leaf-with-no-samples-to-count
  ;; A leaf can have nothing to count without its SAMPLE-INDICES being NIL. When a split's
  ;; sampled attribute is constant over the node's rows, MAKE-RANDOM-TEST produces
  ;; threshold = min = max and every row goes left, leaving the right child a leaf holding
  ;; a real but zero-length array. CLASS-DISTRIBUTION divides by a zero sum in both cases
  ;; and returns a uniform distribution rather than signalling, so both must be rejected.
  ;; It is rare but not hypothetical: 3 of letter's 26936 leaves at :max-depth 20.
  (with-serial-kernel
    (multiple-value-bind (datamatrix target) (synthetic-classification-train)
      (declare (ignore target))
      (declare (ignorable datamatrix))
      (let ((forest (synthetic-forest)))
        (ok (packed-topology-p (build-packed-topology forest))
            "the forest packs while every leaf has samples")
        ;; Empty one leaf's indices, which is what such a split leaves behind.
        (labels ((leftmost-leaf (node)
                   (if (node-test-attribute node)
                       (leftmost-leaf (node-left-node node))
                       node)))
          (setf (node-sample-indices
                 (leftmost-leaf (dtree-root (first (forest-dtree-list forest)))))
                (make-array 0 :element-type 'fixnum)))
        (ok (handler-case (progn (build-packed-topology forest) nil)
              (packed-build-error () t))
            "a leaf with an empty sample-indices array is rejected")))))

(deftest packed-topology-counts-add-up
  (with-serial-kernel
    (let* ((forest (synthetic-forest))
           (topology (build-packed-topology forest)))
      (ok (= (packed-topology-n-tree topology) (forest-n-tree forest))
          "one root per tree")
      ;; A binary tree with L leaves has L-1 internal nodes, so a forest of T trees has
      ;; n-leaf - T of them.
      (ok (= (packed-topology-n-internal topology)
             (- (packed-topology-n-leaf topology) (packed-topology-n-tree topology)))
          (format nil "~D internal, ~D leaves, ~D trees"
                  (packed-topology-n-internal topology)
                  (packed-topology-n-leaf topology)
                  (packed-topology-n-tree topology))))))
