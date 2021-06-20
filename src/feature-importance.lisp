;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

(defun make-oob-sample-indices (total-size sample-indices)
  (let ((bitvec (make-array total-size :element-type 'bit :initial-element 0)))
    (loop for index across sample-indices do
      (setf (aref bitvec index) 1))
    (let* ((len (loop for i from 0 below total-size count (= (aref bitvec i) 0)))
           (oob-sample-indices (make-array len :element-type 'fixnum))
           (i-oob 0))
      (loop for i-bitvec from 0 below total-size do
        (when (= (aref bitvec i-bitvec) 0)
          (setf (aref oob-sample-indices i-oob) i-bitvec)
          (incf i-oob)))
      oob-sample-indices)))

(defun dtree-oob-sample-indices (dtree)
  (make-oob-sample-indices (length (dtree-target dtree))
                           (node-sample-indices (dtree-root dtree))))

(defun test-dtree-oob (dtree datamatrix target
                       &key quiet-p oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array single-float) datamatrix)
           (type (simple-array fixnum (*)) target))
  (let* ((n-correct 0)
         (oob-sample-indices (if (null oob-sample-indices)
                                 (dtree-oob-sample-indices dtree)
                                 oob-sample-indices))
         (len-oob (length oob-sample-indices)))
    (declare (type fixnum n-correct len-oob)
             (type (simple-array fixnum (*)) oob-sample-indices))
    (loop for i fixnum from 0 below len-oob do
      (let ((j (aref oob-sample-indices i)))
        (declare (type fixnum j))
        (when (= (predict-dtree dtree datamatrix j)
                 (aref target j))
          (incf n-correct))))
    (calc-accuracy n-correct len-oob :quiet-p quiet-p)))

(defun find-leaf-randomized (node datamatrix datum-index randomized-attribute oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum datum-index)
           (type (simple-array single-float) datamatrix)
           (type (simple-array fixnum) oob-sample-indices))
  (flet ((random-pick-oob-index ()
           (aref oob-sample-indices (random (length oob-sample-indices)))))
    (cond ((null node) nil)
          ((null (node-test-attribute node)) node)
          (t (let* ((attribute (node-test-attribute node))
                    (threshold (node-test-threshold node))
                    (datum (if (= attribute randomized-attribute)
                               (aref datamatrix (random-pick-oob-index) attribute)
                               (aref datamatrix datum-index attribute))))
               (declare (type fixnum attribute)
                        (type single-float threshold datum))
               (if (>= datum threshold)
                   (find-leaf-randomized (node-left-node node) datamatrix datum-index
                                         randomized-attribute oob-sample-indices)
                   (find-leaf-randomized (node-right-node node) datamatrix datum-index
                                         randomized-attribute oob-sample-indices)))))))

(defun predict-dtree-randomized (dtree datamatrix datum-index randomized-attribute oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array single-float) datamatrix)
           (type fixnum datum-index randomized-attribute)
           (type (simple-array fixnum) oob-sample-indices))
  (let ((max 0.0)
        (max-class 0)
        (dist (node-class-distribution
               (find-leaf-randomized (dtree-root dtree) datamatrix datum-index
                                     randomized-attribute oob-sample-indices)))
        (n-class (dtree-n-class dtree)))
    (declare (type single-float max)
             (type fixnum max-class n-class)
             (type (simple-array single-float) dist))
    (loop for i fixnum from 0 to (1- n-class) do
      (when (> (aref dist i) max)
        (setf max (aref dist i)
              max-class i)))
    max-class))

(defun test-dtree-oob-randomized (dtree datamatrix target randomized-attribute
                                  &key quiet-p oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array single-float) datamatrix)
           (type (simple-array fixnum (*)) target))
  (let* ((n-correct 0)
         (oob-sample-indices (if (null oob-sample-indices)
                                 (dtree-oob-sample-indices dtree)
                                 oob-sample-indices))
         (len-oob (length oob-sample-indices)))
    (declare (type fixnum n-correct len-oob)
             (type (simple-array fixnum (*)) oob-sample-indices))
    (loop for i fixnum from 0 below len-oob do
      (let ((j (aref oob-sample-indices i)))
        (declare (type fixnum j))
        (when (= (predict-dtree-randomized dtree datamatrix j randomized-attribute oob-sample-indices)
                 (aref target j))
          (incf n-correct))))
    (calc-accuracy n-correct len-oob :quiet-p quiet-p)))

(defun normalize-arr! (arr)
  (let ((sum (loop for elem across arr sum elem)))
    (loop for i from 0 below (length arr) do
      (setf (aref arr i) (/ (aref arr i) sum)))
    arr))

;;; Mean Decrease Accuracy

(defun dtree-feature-importance (dtree datamatrix target)
  (let* ((oob-sample-indices (dtree-oob-sample-indices dtree))
         (accuracy-oob (test-dtree-oob dtree datamatrix target
                                       :quiet-p t :oob-sample-indices oob-sample-indices))
         (result (make-array (dtree-datum-dim dtree) :initial-element 0.0)))
    (loop for i from 0 below (dtree-datum-dim dtree) do
      (setf (aref result i)
            (- accuracy-oob
               (test-dtree-oob-randomized dtree datamatrix target i :quiet-p t))))
    (normalize-arr! result)))

(defun test-rtree-oob (rtree datamatrix target &key quiet-p oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree rtree)
           (type (simple-array single-float) datamatrix target))
  (let* ((sum-square-error 0.0)
         (oob-sample-indices (if (null oob-sample-indices)
                                 (dtree-oob-sample-indices rtree)
                                 oob-sample-indices))
         (len-oob (length oob-sample-indices)))
    (declare (type single-float sum-square-error)
             (type fixnum len-oob)
             (type (simple-array fixnum) oob-sample-indices))
    (loop for i fixnum from 0 below len-oob do
      (let ((j (aref oob-sample-indices i)))
        (declare (type fixnum j))
        (incf sum-square-error (square (- (predict-rtree rtree datamatrix j)
                                          (aref target j))))))
    (setf sum-square-error (sqrt (/ sum-square-error len-oob)))
    (when (null quiet-p)
      (format t "RMSE: ~A~%" sum-square-error))
    sum-square-error))

(defun predict-rtree-randomized (rtree datamatrix datum-index randomized-attribute oob-sample-indices)
  (node-regression-mean
   (find-leaf-randomized (dtree-root rtree)
                         datamatrix
                         datum-index
                         randomized-attribute
                         oob-sample-indices)))

(defun test-rtree-oob-randomized (rtree datamatrix target randomized-attribute
                                  &key quiet-p oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree rtree)
           (type (simple-array single-float) datamatrix target)
           (type fixnum randomized-attribute))
  (let* ((sum-square-error 0.0)
         (oob-sample-indices (if (null oob-sample-indices)
                                 (dtree-oob-sample-indices rtree)
                                 oob-sample-indices))
         (len-oob (length oob-sample-indices)))
    (declare (type single-float sum-square-error)
             (type fixnum len-oob)
             (type (simple-array fixnum) oob-sample-indices))
    (loop for i fixnum from 0 below len-oob do
      (let ((j (aref oob-sample-indices i)))
        (declare (type fixnum j))
        (incf sum-square-error (square (- (predict-rtree-randomized rtree datamatrix j randomized-attribute oob-sample-indices)
                                          (aref target j))))))
    (setf sum-square-error (sqrt (/ sum-square-error len-oob)))
    (when (null quiet-p)
      (format t "RMSE: ~A~%" sum-square-error))
    sum-square-error))

(defun rtree-feature-importance (rtree datamatrix target)
  (let* ((oob-sample-indices (dtree-oob-sample-indices rtree))
         (rms-oob (test-rtree-oob rtree datamatrix target
                                  :quiet-p t :oob-sample-indices oob-sample-indices))
         (result (make-array (dtree-datum-dim rtree) :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 below (dtree-datum-dim rtree) do
      (setf (aref result i)
            (- (test-rtree-oob-randomized rtree datamatrix target i :quiet-p t)
               rms-oob)))
    (normalize-arr! result)))

(defun forest-feature-importance (forest datamatrix target)
  (let* ((len (forest-datum-dim forest))
         (result (make-array len :initial-element 0.0)))
    (dolist (importance-vec
             (mapcar/pmapcar (lambda (dtree)
                               (if (rtree? dtree)
                                   (rtree-feature-importance dtree datamatrix target)
                                   (dtree-feature-importance dtree datamatrix target)))
                             (forest-dtree-list forest)))
      (loop for i from 0 below len do
        (incf (aref result i)
              (aref importance-vec i))))
    (loop for i from 0 below len do
      (setf (aref result i) (/ (aref result i) (forest-n-tree forest))))
    result))

;;; Mean Decrease Information gain

(defun dtree-feature-importance-impurity (dtree)
  (let* ((dim (dtree-datum-dim dtree))
         (acc-arr (clol::make-vec dim 0.0))
         (cnt-arr (clol::make-vec dim 0.0)))

    ;; ignore root and leaf nodes
    (flet ((store-decrease-impurity (node)
             (let ((left (node-left-node node))
                   (right (node-right-node node))
                   (len (node-n-sample node))
                   (attr (node-test-attribute node)))
               (when (and attr (node-test-attribute left) (node-test-attribute right))
                 (incf (aref acc-arr attr)
                       (- (node-information-gain node)
                          (+ (* (/ (node-n-sample  left) len) (node-information-gain left))
                             (* (/ (node-n-sample right) len) (node-information-gain right)))))
                 (incf (aref cnt-arr attr) 1.0)))))
      (traverse #'store-decrease-impurity (node-left-node (dtree-root dtree)))
      (traverse #'store-decrease-impurity (node-right-node (dtree-root dtree))))
    
    (loop for i from 0 below dim do
      (when (> (aref cnt-arr i) 0.0)
        (setf (aref acc-arr i) (/ (aref acc-arr i) (aref cnt-arr i)))))
    
    (let ((min (loop for i from 0 below dim minimize (aref acc-arr i))))
      (loop for i from 0 below dim do
        (setf (aref acc-arr i) (- (aref acc-arr i) min))))
    (normalize-arr! acc-arr)))

(defun forest-feature-importance-impurity (forest)
  (let* ((len (forest-datum-dim forest))
         (result (make-array len :initial-element 0.0)))
    (dolist (importance-vec
             (mapcar/pmapcar #'dtree-feature-importance-impurity (forest-dtree-list forest)))
      (loop for i from 0 below len do
        (incf (aref result i)
              (aref importance-vec i))))
    (loop for i from 0 below len do
      (setf (aref result i) (/ (aref result i) (forest-n-tree forest))))
    result))
