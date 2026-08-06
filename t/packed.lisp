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

(deftest packed-classifier-agrees-bit-for-bit
  ;; Agreement on the class alone is too weak -- two different distributions can share an
  ;; argmax. Compare the whole distribution, and require exact equality: the same floats
  ;; are summed in the same tree and class order, so anything else is a defect.
  (with-serial-kernel
    (multiple-value-bind (datamatrix target) (synthetic-classification-test)
      (declare (ignore target))
      (let ((forest (synthetic-forest)))
        (dolist (payload '(:dense :csr))
          (let ((classifier (build-packed-classifier forest :payload payload)))
            (multiple-value-bind (class-bad dist-bad worst)
                (packed-verify classifier forest datamatrix)
              (ok (zerop class-bad)
                  (format nil "~A: ~D classes differ" payload class-bad))
              (ok (zerop dist-bad)
                  (format nil "~A: ~D distributions differ" payload dist-bad))
              (ok (zerop worst)
                  (format nil "~A: worst absolute difference ~,10F" payload worst)))))))))

(deftest packed-classifier-auto-picks-on-the-dense-row-size
  ;; CSR once a dense row exceeds a cache line, dense below. Not the compression ratio:
  ;; a 10-class model compresses 2.4x on that measure and runs 0.82x as fast.
  ;;
  ;; The rule is unit-tested on both sides of the boundary, because no fixture forest has
  ;; enough classes to reach the CSR side -- the synthetic set has 4.
  (let ((rule #'cl-random-forest/src/packed/classifier::choose-payload))
    (ok (eq :dense (funcall rule 1)) "1 class is dense")
    (ok (eq :dense (funcall rule 16)) "16 classes is dense, a row being exactly 64 bytes")
    (ok (eq :csr (funcall rule 17)) "17 classes is csr, a row exceeding a cache line")
    (ok (eq :csr (funcall rule 26)) "26 classes is csr"))
  (with-serial-kernel
    (let ((forest (synthetic-forest)))
      (ok (eq :dense (packed-classifier-kind (build-packed-classifier forest)))
          (format nil "a ~D-class forest selects dense" +synthetic-n-class+)))))

(deftest packed-classifier-csr-holds-only-the-non-zero-entries
  (with-serial-kernel
    (let* ((forest (synthetic-forest))
           (topology (build-packed-topology forest))
           (dense (build-packed-classifier forest :payload :dense))
           (csr (build-packed-classifier forest :payload :csr))
           (n-leaf (packed-topology-n-leaf topology))
           (n-class (packed-classifier-n-class dense))
           (expected 0))
      (dotimes (row n-leaf)
        (dotimes (k n-class)
          (unless (zerop (aref (packed-classifier-table dense) row k))
            (incf expected))))
      (ok (= expected (length (packed-classifier-probability csr)))
          (format nil "~D non-zero entries, CSR stores ~D"
                  expected (length (packed-classifier-probability csr)))))))

(deftest packed-batch-agrees-with-per-datum
  ;; Tree-major batching walks one tree over a whole tile before moving to the next, which
  ;; changes the order arrays are touched but not the arithmetic: each datum's accumulator
  ;; receives the same values in the same tree order.
  (with-serial-kernel
    (multiple-value-bind (datamatrix target) (synthetic-classification-test)
      (declare (ignore target))
      (let* ((forest (synthetic-forest))
             (n-rows (array-dimension datamatrix 0)))
        (dolist (payload '(:dense :csr))
          (let* ((classifier (build-packed-classifier forest :payload payload))
                 (acc (make-packed-accumulator classifier))
                 (bad 0))
            (dolist (tile (list 1 7 64 n-rows))
              (let ((accs (make-packed-accumulators classifier tile))
                    (out (make-array tile :element-type 'fixnum))
                    (i 0))
                (loop while (< i n-rows)
                      do (let ((end (min n-rows (+ i tile))))
                           (packed-predict-batch classifier datamatrix i end accs out)
                           (loop for j from i below end
                                 do (unless (= (aref out (- j i))
                                               (packed-predict classifier datamatrix j acc))
                                      (incf bad)))
                           (setf i end)))))
            (ok (zerop bad)
                (format nil "~A: ~D batch answers differ from per-datum" payload bad))))))))

(deftest packed-round-trips-through-a-file
  (with-serial-kernel
    (multiple-value-bind (datamatrix target) (synthetic-classification-test)
      (declare (ignore target))
      (let ((forest (synthetic-forest)))
        (dolist (payload '(:dense :csr))
          (uiop:with-temporary-file (:pathname path :type "packed")
            (let* ((original (build-packed-classifier forest :payload payload))
                   (acc (make-packed-accumulator original)))
              (packed-save original path)
              (let* ((restored (packed-load path))
                     (restored-acc (make-packed-accumulator restored))
                     (bad 0))
                (ok (eq (packed-classifier-kind restored) payload)
                    (format nil "~A survives the round trip" payload))
                (dotimes (i (array-dimension datamatrix 0))
                  (unless (= (packed-predict original datamatrix i acc)
                             (packed-predict restored datamatrix i restored-acc))
                    (incf bad)))
                (ok (zerop bad)
                    (format nil "~A: ~D predictions differ after reload" payload bad))))))))))

(deftest packed-load-rejects-a-file-it-cannot-trust
  (with-serial-kernel
    (let ((forest (synthetic-forest)))
      (uiop:with-temporary-file (:pathname path :type "packed")
        (packed-save (build-packed-classifier forest) path)
        ;; Corrupt the magic.
        (with-open-file (s path :direction :io :if-exists :overwrite
                                :element-type '(unsigned-byte 8))
          (file-position s 0)
          (write-byte 0 s))
        (ok (handler-case (progn (packed-load path) nil)
              (packed-load-error () t))
            "a bad magic number is rejected"))
      (uiop:with-temporary-file (:pathname path :type "packed")
        (packed-save (build-packed-classifier forest) path)
        ;; Corrupt the version, which lives at byte offset 8.
        (with-open-file (s path :direction :io :if-exists :overwrite
                                :element-type '(unsigned-byte 8))
          (file-position s 8)
          (write-byte 99 s))
        (ok (handler-case (progn (packed-load path) nil)
              (packed-load-error () t))
            "an unknown version is rejected"))
      (uiop:with-temporary-file (:pathname path :type "packed")
        (packed-save (build-packed-classifier forest) path)
        ;; Corrupt the byte-order probe, at offset 12.
        (with-open-file (s path :direction :io :if-exists :overwrite
                                :element-type '(unsigned-byte 8))
          (file-position s 12)
          (write-byte 99 s))
        (ok (handler-case (progn (packed-load path) nil)
              (packed-load-error () t))
            "a byte-order mismatch is rejected"))
      (uiop:with-temporary-file (:pathname path :type "packed")
        (packed-save (build-packed-classifier forest) path)
        ;; Truncate.
        (let ((bytes (with-open-file (s path :element-type '(unsigned-byte 8))
                       (file-length s))))
          (with-open-file (s path :direction :output :if-exists :supersede
                                  :element-type '(unsigned-byte 8))
            (dotimes (i (floor bytes 2)) (write-byte 0 s))))
        (ok (handler-case (progn (packed-load path) nil)
              (packed-load-error () t))
            "a truncated file is rejected")))))

(deftest packed-float-bits-round-trip
  ;; The fast paths exist per implementation; the portable fallback is the reference. They
  ;; must agree, or a model saved on one implementation would not load on another.
  (let ((values (list 0.0 1.0 -1.0 0.5 -0.5 3.14159 1.0e-8 1.0e8
                      most-positive-single-float least-positive-normalized-single-float)))
    (dolist (v values)
      (ok (= v (cl-random-forest/src/packed::bits-to-single-float
                (cl-random-forest/src/packed::single-float-to-bits v)))
          (format nil "~S survives the bit round trip" v))
      (ok (= (cl-random-forest/src/packed::single-float-to-bits v)
             (cl-random-forest/src/packed::%portable-single-float-to-bits v))
          (format nil "~S: fast path agrees with the portable one" v)))))
