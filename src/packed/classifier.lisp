(defpackage :cl-random-forest/src/packed/classifier
  (:use #:cl
        #:cl-random-forest/src/packed/topology)
  (:import-from #:cl-random-forest/src/random-forest
                #:forest-dtree-list
                #:forest-n-class
                #:dtree-root
                #:node-test-attribute
                #:node-left-node
                #:node-right-node
                #:node-class-distribution
                #:class-distribution-forest
                #:predict-forest
                #:argmax)
  (:export #:packed-classifier
           #:packed-classifier-p
           #:packed-classifier-topology
           #:packed-classifier-n-class
           #:packed-classifier-kind
           #:packed-classifier-table
           #:packed-classifier-offsets
           #:packed-classifier-class
           #:packed-classifier-probability
           #:%make-packed-classifier
           #:build-packed-classifier
           #:make-packed-accumulator
           #:make-packed-accumulators
           #:packed-predict
           #:packed-predict-batch
           #:packed-verify))

(in-package :cl-random-forest/src/packed/classifier)

(defstruct (packed-classifier (:constructor %make-packed-classifier))
  "A topology plus a leaf payload.

KIND is :DENSE, in which case TABLE is an n-leaf x n-class array and the CSR slots are
empty; or :CSR, in which case leaf L occupies [OFFSETS[L], OFFSETS[L+1]) of CLASS and
PROBABILITY and TABLE is empty. Omitting the zeros is exact, not approximate: the values
are non-negative, so leaving out a + 0.0 changes no sum."
  (topology (%make-packed-topology) :type packed-topology)
  (n-class 0 :type fixnum)
  (kind :dense :type (member :dense :csr))
  (table (make-array '(0 0) :element-type 'single-float)
         :type (simple-array single-float (* *)))
  (offsets (make-array 1 :element-type '(unsigned-byte 32))
           :type (simple-array (unsigned-byte 32) (*)))
  (class (make-array 0 :element-type '(unsigned-byte 16))
         :type (simple-array (unsigned-byte 16) (*)))
  (probability (make-array 0 :element-type 'single-float)
               :type (simple-array single-float (*))))

(defun leaf-distributions (forest topology)
  "A simple-vector of every leaf's distribution, indexed by the topology's leaf number.

NODE-CLASS-DISTRIBUTION hands back the dtree's shared scratch buffer, so each one is copied
before the next leaf overwrites it."
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
                                           (copy-seq (node-class-distribution node)))
                                     (incf local)))))
                 (walk (dtree-root dtree)))))
    out))

(defun choose-payload (n-class)
  "CSR once a dense row exceeds a cache line, dense otherwise.

The tempting rule is the compression ratio, and it picks wrong: a 10-class model compresses
2.4x and runs 0.82x as fast, because a 40-byte dense row already fits in a cache line and
CSR's offset loads and scattered writes cost more than the bytes saved."
  (if (> (* 4 n-class) 64) :csr :dense))

(defun build-packed-classifier (forest &key (payload :auto))
  "Flatten FOREST into a packed classifier. PAYLOAD is :AUTO, :DENSE or :CSR."
  (check-type payload (member :auto :dense :csr))
  (let* ((topology (build-packed-topology forest))
         (n-class (forest-n-class forest))
         (kind (if (eq payload :auto) (choose-payload n-class) payload))
         (n-leaf (packed-topology-n-leaf topology))
         (dists (leaf-distributions forest topology)))
    (ecase kind
      (:dense
       (let ((table (make-array (list (max n-leaf 1) n-class)
                                :element-type 'single-float :initial-element 0.0)))
         (dotimes (row n-leaf)
           (let ((d (svref dists row)))
             (dotimes (k n-class) (setf (aref table row k) (aref d k)))))
         (%make-packed-classifier :topology topology :n-class n-class
                                  :kind :dense :table table)))
      (:csr
       (unless (< n-class 65536)
         (error 'packed-build-error
                :detail (format nil "~D classes exceeds a (unsigned-byte 16) class index"
                                n-class)))
       (let ((nnz 0))
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
           (%make-packed-classifier :topology topology :n-class n-class :kind :csr
                                    :offsets offsets :class class
                                    :probability probability)))))))

(defun make-packed-accumulator (classifier)
  "A fresh accumulator for PACKED-PREDICT. One per thread, never shared."
  (make-array (packed-classifier-n-class classifier)
              :element-type 'single-float :initial-element 0.0))

(defun packed-predict (classifier datamatrix datum-index acc)
  "PREDICT-FOREST's answer for one datum. Writes only ACC.

ACC is normalised in place: on return it holds the class distribution (summed votes divided
by N-TREE), not the raw sums. PACKED-VERIFY relies on this to read the distribution straight
out of ACC. Contrast PACKED-PREDICT-BATCH, which leaves its accumulators unnormalised."
  (declare (optimize (speed 3) (safety 0))
           (type packed-classifier classifier)
           (type (simple-array single-float (* *)) datamatrix)
           (type (simple-array single-float (*)) acc)
           (type fixnum datum-index))
  (let* ((topology (packed-classifier-topology classifier))
         (n-class (packed-classifier-n-class classifier))
         (roots (packed-topology-roots topology))
         (n-tree (packed-topology-n-tree topology)))
    (declare (type (simple-array (signed-byte 32) (*)) roots)
             (type fixnum n-class n-tree))
    (dotimes (k n-class) (setf (aref acc k) 0.0))
    (ecase (packed-classifier-kind classifier)
      (:dense
       (let ((table (packed-classifier-table classifier)))
         (declare (type (simple-array single-float (* *)) table))
         (dotimes (tree n-tree)
           (let ((row (packed-leaf topology datamatrix datum-index (aref roots tree))))
             (declare (type fixnum row))
             (dotimes (k n-class) (incf (aref acc k) (aref table row k)))))))
      (:csr
       (let ((offsets (packed-classifier-offsets classifier))
             (class (packed-classifier-class classifier))
             (probability (packed-classifier-probability classifier)))
         (declare (type (simple-array (unsigned-byte 32) (*)) offsets)
                  (type (simple-array (unsigned-byte 16) (*)) class)
                  (type (simple-array single-float (*)) probability))
         (dotimes (tree n-tree)
           (let ((leaf (packed-leaf topology datamatrix datum-index (aref roots tree))))
             (declare (type fixnum leaf))
             (loop for i of-type fixnum
                     from (aref offsets leaf) below (aref offsets (1+ leaf))
                   do (incf (aref acc (aref class i)) (aref probability i))))))))
    (dotimes (k n-class) (setf (aref acc k) (/ (aref acc k) n-tree)))
    (argmax acc)))

(defun make-packed-accumulators (classifier tile)
  "Accumulators for PACKED-PREDICT-BATCH: one row per datum in the tile."
  (make-array (list tile (packed-classifier-n-class classifier))
              :element-type 'single-float :initial-element 0.0))

(defun packed-predict-batch (classifier datamatrix start end accs out)
  "Predict rows [START,END) a tree at a time, writing classes into OUT indexed from zero.

Walking one tree over the whole tile before moving to the next touches each tree's arrays
once per tile instead of once per datum. The cost is holding (END - START) x n-class
accumulators, which is why a tile of one loses and a tile of a few hundred wins.

ACCS must have at least (END - START) rows -- as MAKE-PACKED-ACCUMULATORS gives it when
called with a TILE no smaller than (END - START) -- and OUT must have length at least
(END - START); this runs at (safety 0), so either being too small is a silent out-of-bounds
write, not a signalled error.

Unlike PACKED-PREDICT, ACCS is left holding raw per-tree sums, not divided by N-TREE: the
division by N-TREE happens only in a local while picking each row's argmax for OUT, and
that quotient is never written back. A caller that wants ACCS's own rows to be class
distributions must divide each one by N-TREE itself."
  (declare (optimize (speed 3) (safety 0))
           (type packed-classifier classifier)
           (type (simple-array single-float (* *)) datamatrix accs)
           (type (simple-array fixnum (*)) out)
           (type fixnum start end))
  (let* ((topology (packed-classifier-topology classifier))
         (n-class (packed-classifier-n-class classifier))
         (roots (packed-topology-roots topology))
         (n-tree (packed-topology-n-tree topology))
         (tile (- end start)))
    (declare (type (simple-array (signed-byte 32) (*)) roots)
             (type fixnum n-class n-tree tile))
    (dotimes (r tile)
      (dotimes (k n-class) (setf (aref accs r k) 0.0)))
    (ecase (packed-classifier-kind classifier)
      (:dense
       (let ((table (packed-classifier-table classifier)))
         (declare (type (simple-array single-float (* *)) table))
         (dotimes (tree n-tree)
           (let ((root (aref roots tree)))
             (dotimes (r tile)
               (let ((row (packed-leaf topology datamatrix (+ start r) root)))
                 (declare (type fixnum row))
                 (dotimes (k n-class)
                   (incf (aref accs r k) (aref table row k)))))))))
      (:csr
       (let ((offsets (packed-classifier-offsets classifier))
             (class (packed-classifier-class classifier))
             (probability (packed-classifier-probability classifier)))
         (declare (type (simple-array (unsigned-byte 32) (*)) offsets)
                  (type (simple-array (unsigned-byte 16) (*)) class)
                  (type (simple-array single-float (*)) probability))
         (dotimes (tree n-tree)
           (let ((root (aref roots tree)))
             (dotimes (r tile)
               (let ((leaf (packed-leaf topology datamatrix (+ start r) root)))
                 (declare (type fixnum leaf))
                 (loop for i of-type fixnum
                         from (aref offsets leaf) below (aref offsets (1+ leaf))
                       do (incf (aref accs r (aref class i))
                                (aref probability i))))))))))
    (dotimes (r tile out)
      (let ((best most-negative-single-float)
            (best-k 0))
        (declare (type single-float best) (type fixnum best-k))
        (dotimes (k n-class)
          (let ((v (/ (aref accs r k) n-tree)))
            (when (> v best) (setf best v best-k k))))
        (setf (aref out r) best-k)))))

(defun packed-verify (classifier forest datamatrix)
  "Compare CLASSIFIER with FOREST over every row of DATAMATRIX.

Returns (values class-disagreements distribution-disagreements worst-absolute-difference).
All three should be zero: the distribution is checked element by element, not merely after
ARGMAX, and exact equality is achievable because the same floats are summed in the same
order."
  (let* ((n-class (packed-classifier-n-class classifier))
         (acc (make-packed-accumulator classifier))
         (class-bad 0) (dist-bad 0) (worst 0.0))
    (dotimes (i (array-dimension datamatrix 0) (values class-bad dist-bad worst))
      (let ((reference-class (predict-forest forest datamatrix i))
            (reference-dist (copy-seq (class-distribution-forest forest datamatrix i))))
        (unless (= reference-class (packed-predict classifier datamatrix i acc))
          (incf class-bad))
        (let ((row-bad nil))
          (dotimes (k n-class)
            (let ((d (abs (- (aref reference-dist k) (aref acc k)))))
              (when (> d 0.0) (setf row-bad t))
              (setf worst (max worst d))))
          (when row-bad (incf dist-bad)))))))
