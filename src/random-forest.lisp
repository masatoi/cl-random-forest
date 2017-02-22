(in-package :cl-user)
(defpackage cl-random-forest
  (:use :cl)
  (:nicknames :clrf))

(in-package :cl-random-forest)

;;; decision tree

(defstruct (dtree (:constructor %make-dtree)
                  (:print-object %print-dtree))
  n-class class-count-array datum-dim dataset
  root max-depth min-region-samples n-trial gain-test)

(defun %print-dtree (obj stream)
  (format stream "#S(DTREE :N-CLASS ~A :DATUM-DIM ~A :ROOT ~A)"
          (dtree-n-class obj)
          (dtree-datum-dim obj)
          (dtree-root obj)))

(defun make-dtree (n-class datum-dim dataset
                           &key (max-depth 5) (min-region-samples 1) (n-trial 10)
                             (gain-test #'entropy) sample-indices)
  (let ((dtree (%make-dtree
                :n-class n-class
                :class-count-array (make-array n-class :element-type 'double-float)
                :datum-dim datum-dim
                :dataset dataset
                :max-depth max-depth :min-region-samples min-region-samples
                :n-trial n-trial
                :gain-test gain-test)))
    (setf (dtree-root dtree) (make-root-node dtree :sample-indices sample-indices))
    (split-node! (dtree-root dtree))
    dtree))

(defstruct (node (:constructor %make-node)
                 (:print-object %print-node))
  sample-indices depth test-attribute test-threshold information-gain 
  left-node right-node dtree)

(defun %print-node (obj stream)
  (format stream "#S(NODE :TEST ~A :GAIN ~A)"
          (list (node-test-attribute obj) (node-test-threshold obj))
          (node-information-gain obj)))

(defun make-root-node (dtree &key sample-indices)
  (%make-node
   :information-gain 1d0
   :sample-indices (if sample-indices
                     sample-indices
                     (alexandria:iota (length (dtree-dataset dtree))))
   :depth 0
   :dtree dtree))

(defun make-node (sample-indices parent-node)
  (%make-node :sample-indices sample-indices
              :depth (1+ (node-depth parent-node))
              :dtree (node-dtree parent-node)))

(defun class-distribution (node)
  (let* ((dtree (node-dtree node))
         (n-class (dtree-n-class dtree))
         (class-count-array (dtree-class-count-array dtree)))
    (declare (optimize (speed 3) (safety 1))
             (type fixnum n-class)
             (type (simple-array double-float) class-count-array))
    ;; init
    (loop for i fixnum from 0 to (1- n-class) do
      (setf (aref class-count-array i) 0d0))
    ;; count
    (loop for i fixnum in (node-sample-indices node) do
      (incf (aref class-count-array (car (aref (dtree-dataset dtree) i))) 1d0))
    (let ((sum (loop for c double-float across class-count-array summing c)))
      (declare (type double-float sum))
      (loop for i fixnum from 0 to (1- n-class) do
        (if (= sum 0d0)
            (setf (aref class-count-array i) (/ 1d0 n-class))
            (setf (aref class-count-array i) (/ (aref class-count-array i) sum)))))
    (dtree-class-count-array dtree)))

(defun gini (node)
  (* -1d0
     (loop for pk double-float across (class-distribution node)
           summing
           (* pk (- 1d0 pk)))))

(defun entropy (node)
  (* -1d0
     (loop for pk double-float across (class-distribution node)
           summing
           (if (= pk 0d0)
               0d0
               (* pk (log pk))))))

(defun min/max (lst)
  (let ((min (car lst))
        (max (car lst)))
    (dolist (elem (cdr lst))
      (cond ((< max elem) (setf max elem))
            ((> min elem) (setf min elem))))
    (values min max)))

(defun random-uniform (start end)
  (+ (random (- end start)) start))

(defun make-random-test (node)
  (declare (optimize (speed 3) (safety 1)))
  (let* ((dtree (node-dtree node))
         (attribute (random (dtree-datum-dim dtree)))
         (indices (node-sample-indices node))
         (line (mapcar (lambda (index)
                         (aref (cdr (aref (dtree-dataset dtree) index)) attribute))
                       indices)))
    (multiple-value-bind (min max)
        (min/max line)
      (let ((threshold (if (= min max) min (random-uniform min max))))
        (values attribute threshold)))))

(defun run-test (attribute threshold x)
  (declare (optimize (speed 3) (safety 1))
           (type fixnum attribute)
           (type double-float threshold)
           (type (simple-array double-float) x))
  (>= (aref x attribute) threshold))

(defun split-list (pred lst)
  (let ((true-list nil)
        (false-list nil))
    (dolist (elem lst)
      (if (funcall pred elem)
          (push elem true-list)
          (push elem false-list)))
    (values true-list false-list)))

(defun set-best-children! (k node &optional (gain-test #'entropy))
  (declare (optimize (speed 3) (safety 1)))
  (let ((max-children-gain most-negative-double-float))
    (loop repeat k do
      (multiple-value-bind (attribute threshold)
          (make-random-test node)
        (let ((test-func (lambda (index)
                           (let ((x (cdr (aref (dtree-dataset (node-dtree node)) index))))
                             (run-test attribute threshold x)))))
          (multiple-value-bind (left-sample-indices right-sample-indices)
              (split-list test-func (node-sample-indices node))
            (let* ((left-node (make-node left-sample-indices node))
                   (left-gain (funcall gain-test left-node))
                   (right-node (make-node right-sample-indices node))
                   (right-gain (funcall gain-test right-node))
                   (parent-size (length (node-sample-indices node)))
                   (children-gain
                     (+ (* -1d0 (/ (length left-sample-indices) parent-size) left-gain)
                        (* -1d0 (/ (length right-sample-indices) parent-size) right-gain))))
              (when (< max-children-gain children-gain)
                (setf max-children-gain children-gain
                      (node-test-attribute node) attribute
                      (node-test-threshold node) threshold
                      (node-left-node node) left-node
                      (node-information-gain left-node) left-gain
                      (node-right-node node) right-node
                      (node-information-gain right-node) right-gain))))))))
  node)

(defun stop-split? (node)
  (or (= (node-information-gain node) 0d0)
      (<= (length (node-sample-indices node))
          (dtree-min-region-samples (node-dtree node)))
      (>= (node-depth node)
          (dtree-max-depth (node-dtree node)))))

(defun split-node! (node)
  (when (and node (not (stop-split? node)))
    (set-best-children! (dtree-n-trial (node-dtree node)) node)
    (split-node! (node-left-node node))
    (split-node! (node-right-node node))))

(defun traverse (fn node)
  (cond ((null node) nil)
        ((and (null (node-left-node node))
              (null (node-right-node node)))
         (funcall fn node))
        ((null (node-left-node node)) (list (funcall fn node) (traverse fn (node-right-node node))))
        ((null (node-right-node node)) (list (funcall fn node) (traverse fn (node-left-node node))))
        (t (list (funcall fn node)
                 (traverse fn (node-left-node node))
                 (traverse fn (node-right-node node))))))

(defun find-leaf (node x)
  (cond ((null node) nil)
        ((null (node-test-attribute node)) node)
        (t (node-left-node node)
           (node-right-node node)
           (if (run-test (node-test-attribute node) (node-test-threshold node) x)
               (find-leaf (node-left-node node) x)
               (find-leaf (node-right-node node) x)))))

(defun predict-dtree (dtree x)
  (let ((max 0d0)
        (max-class 0)
        (dist (class-distribution (find-leaf (dtree-root dtree) x))))
    (declare (optimize (speed 3) (safety 1))
             (type double-float max)
             (type fixnum max-class)
             (type (simple-array double-float) dist))
    (loop for i fixnum from 0 to (1- (dtree-n-class dtree)) do
      (when (> (aref dist i) max)
        (setf max (aref dist i)
              max-class i)))
    max-class))

(defun test-dtree (dtree dataset)
  (let ((counter 0))
    (loop for datum across dataset do
      (when (= (predict-dtree dtree (cdr datum)) (car datum))
        (incf counter)))
    (* 100d0 (/ counter (length dataset)))))

;;; forest

(defstruct (forest (:constructor %make-forest)
                   (:print-object %print-forest))
  n-tree bagging-ratio dataset dtree-list
  n-class class-count-array datum-dim max-depth min-region-samples n-trial gain-test)

(defun %print-forest (obj stream)
  (format stream "#S(FOREST :N-TREE ~A)"
          (forest-n-tree obj)))

(defun bootstrap-sample-indices (n dataset)
  (let ((len (length dataset)))
    (loop repeat n collect (random len))))

(defun make-forest (n-class datum-dim dataset
                    &key (n-tree 100) (bagging-ratio 0.1) (max-depth 5) (min-region-samples 1)
                      (n-trial 10) (gain-test #'entropy))
  (let ((forest (%make-forest
                 :n-tree n-tree
                 :bagging-ratio bagging-ratio
                 :dataset dataset
                 :n-class n-class
                 :class-count-array (make-array n-class :element-type 'double-float)
                 :datum-dim datum-dim
                 :max-depth max-depth
                 :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test)))
    (setf (forest-dtree-list forest)
          (loop repeat n-tree collect
            (make-dtree n-class datum-dim dataset
                 :max-depth max-depth
                 :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test
                 :sample-indices (bootstrap-sample-indices
                                  (floor (* (length dataset) bagging-ratio)) dataset))))
    forest))

(defun class-distribution-forest (forest x)
  (declare (optimize (speed 3) (safety 1))
           (type (simple-array double-float) x))
  ;; init forest-class-count-array
  (loop for i fixnum from 0 to (1- (forest-n-class forest)) do
    (setf (aref (forest-class-count-array forest) i) 0d0))
  ;; whole count
  (dolist (dtree (forest-dtree-list forest))
    (let ((dist (class-distribution (find-leaf (dtree-root dtree) x))))
      (declare (type (simple-array double-float) dist))
      (loop for i fixnum from 0 to (1- (forest-n-class forest)) do
        (incf (aref (forest-class-count-array forest) i) (aref dist i)))))
  (let ((n-tree (coerce (forest-n-tree forest) 'double-float)))
    (loop for i fixnum from 0 to (1- (forest-n-class forest)) do
      (setf (aref (forest-class-count-array forest) i)
            (/ (aref (forest-class-count-array forest) i) n-tree))))
  (forest-class-count-array forest))

(defun predict-forest (forest x)
  (let ((max 0d0)
        (max-class 0)
        (dist (class-distribution-forest forest x)))
    (declare (optimize (speed 3) (safety 1))
             (type double-float max)
             (type fixnum max-class)
             (type (simple-array double-float) dist))
    (loop for i from 0 to (1- (forest-n-class forest)) do
      (when (> (aref dist i) max)
        (setf max (aref dist i)
              max-class i)))
    max-class))

(defun test-forest (forest dataset)
  (let ((counter 0))
    (loop for datum across dataset do
      (when (= (predict-forest forest (cdr datum)) (car datum))
        (incf counter)))
    (* 100d0 (/ counter (length dataset)))))
