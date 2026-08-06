(defpackage :cl-random-forest/src/packed/topology
  (:use #:cl)
  (:import-from #:cl-random-forest/src/random-forest
                #:forest-dtree-list
                #:forest-n-tree
                #:forest-datum-dim
                #:dtree-root
                #:node-test-attribute
                #:node-test-threshold
                #:node-left-node
                #:node-right-node
                #:node-sample-indices)
  (:export #:packed-topology
           #:packed-topology-p
           #:packed-topology-n-tree
           #:packed-topology-n-internal
           #:packed-topology-n-leaf
           #:packed-topology-feature
           #:packed-topology-threshold
           #:packed-topology-left
           #:packed-topology-right
           #:packed-topology-roots
           #:packed-topology-tree-leaf-offsets
           #:%make-packed-topology
           #:build-packed-topology
           #:packed-leaf
           #:packed-leaf-indices
           #:packed-build-error
           #:packed-build-error-detail))

(in-package :cl-random-forest/src/packed/topology)

(define-condition packed-build-error (error)
  ((detail :initarg :detail :reader packed-build-error-detail))
  (:report (lambda (c s)
             (format s "cannot pack this forest: ~A" (packed-build-error-detail c))))
  (:documentation
   "Signalled for anything the packed traversal could not survive.

The traversal runs at (safety 0) and trusts its declarations, so every assumption has to
hold before it starts. One of them is not obvious: a leaf that has lost its sample indices
would get a uniform class distribution from CLASS-DISTRIBUTION rather than an error, and
that would be frozen into the model silently."))

(defstruct (packed-topology (:constructor %make-packed-topology))
  "Every internal node of every tree of a forest, flattened into parallel arrays.

A leaf is encoded as a negative child index, `~leaf`, so a walk's continuation test is its
leaf test and a leaf costs no extra read. Leaf numbers are
TREE-LEAF-OFFSETS[tree] + the tree's own leaf index, which is by construction the index
Global Refinement uses."
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

(defun check-packable (forest)
  "Signal PACKED-BUILD-ERROR unless FOREST can be packed."
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
                             :detail (format nil "feature ~S outside [0,~D) at depth ~D"
                                             f dim depth))))
                  (unless (typep (node-test-threshold node) 'single-float)
                    (error 'packed-build-error
                           :detail (format nil "threshold ~S is not a single-float"
                                           (node-test-threshold node))))
                  (unless (and (node-left-node node) (node-right-node node))
                    (error 'packed-build-error
                           :detail "an internal node with only one child"))
                  (walk (node-left-node node) (1+ depth))
                  (walk (node-right-node node) (1+ depth)))
                 (t
                  ;; NIL is not the only way to have nothing to count. A split whose
                  ;; sampled attribute is constant over the node's rows gets
                  ;; threshold = min = max from MAKE-RANDOM-TEST, and with the >=
                  ;; convention every row goes left -- leaving the right child a leaf whose
                  ;; SAMPLE-INDICES is a real but zero-length array. CLASS-DISTRIBUTION
                  ;; divides by a zero sum either way and returns a uniform distribution
                  ;; without signalling. Measured: 3 of letter's 26936 leaves at
                  ;; :max-depth 20.
                  (let ((indices (node-sample-indices node)))
                    (unless (and indices (plusp (length indices)))
                      (error 'packed-build-error
                             :detail (format nil "a leaf at depth ~D has ~:[no sample ~
indices at all~;an empty sample-indices array~], so its class distribution would come out ~
uniform without any error being signalled -- if the forest has been pruned, rebuild it ~
with :remove-sample-indices? nil (issue #14)" depth indices))))))))
      (dolist (dtree dtrees)
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
  "Flatten FOREST's structure into arrays.

Two passes: count and validate, then fill exact-size arrays. Validating first matters
because the traversal runs at (safety 0)."
  (check-packable forest)
  (let* ((dtrees (forest-dtree-list forest))
         (n-tree (length dtrees))
         (n-internal 0)
         (n-leaf 0))
    (dolist (dtree dtrees)
      (multiple-value-bind (i l) (count-tree-nodes dtree)
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
                               (prog1 (lognot (+ leaf-base local-leaf))
                                 (incf local-leaf))))))
                   (setf (aref roots tree) (emit (dtree-root dtree))))
                 (incf leaf-base local-leaf)))
      (%make-packed-topology
       :n-tree n-tree :n-internal n-internal :n-leaf n-leaf
       :feature feature :threshold threshold :left left :right right
       :roots roots :tree-leaf-offsets offsets))))

(declaim (inline packed-leaf))
(defun packed-leaf (topology datamatrix datum-index root)
  "The leaf number a datum reaches from ROOT. This is the Global Refinement index."
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

(defun packed-leaf-indices (topology datamatrix datum-index out)
  "Fill OUT with the leaf each tree sends a datum to. OUT is (simple-array fixnum (n-tree))."
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
      (setf (aref out tree)
            (packed-leaf topology datamatrix datum-index (aref roots tree))))))
