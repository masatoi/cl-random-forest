(defpackage :cl-random-forest/src/random-forest
  (:use #:cl
        #:cl-random-forest/src/utils)
  (:export #:make-dtree
           #:predict-dtree
           #:test-dtree
           #:make-forest
           #:predict-forest
           #:test-forest
           #:forest-bagging-ratio
           #:forest-class-count-array
           #:forest-datamatrix
           #:forest-datum-dim
           #:forest-dtree-list
           #:forest-gain-test
           #:forest-index-offset
           #:forest-max-depth
           #:forest-min-region-samples
           #:forest-n-class
           #:forest-n-leaf
           #:forest-n-tree
           #:forest-n-trial
           #:forest-p
           #:forest-target
           #:make-rtree
           #:predict-rtree
           #:test-rtree
           #:make-regression-forest
           #:predict-regression-forest
           #:test-regression-forest
           #:make-refine-vector
           #:make-refine-learner
           #:predict-refine-learner
           #:make-refine-dataset
           #:train-refine-learner
           #:test-refine-learner
           #:train-refine-learner-process
           #:cross-validation-forest-with-refine-learner
           #:pruning!
           #:gini
           #:entropy))

(in-package :cl-random-forest/src/random-forest)

;;; decision tree

(defstruct (dtree (:constructor %make-dtree)
                  (:print-object %print-dtree))
  n-class class-count-array ; for classification
  datum-dim datamatrix target
  root max-depth min-region-samples n-trial gain-test remove-sample-indices? save-parent-node?
  tmp-arr1 tmp-index1 tmp-arr2 tmp-index2
  best-arr1 best-index1 best-arr2 best-index2
  max-leaf-index id)

(defun rtree? (dtree)
  (and (eq (type-of dtree) 'dtree)
       (null (dtree-n-class dtree))))

(defun %print-dtree (obj stream)
  (print-unreadable-object (obj stream :type t :identity t)
    (format stream ":N-CLASS ~A :DATUM-DIM ~A :ROOT ~A"
            (dtree-n-class obj)
            (dtree-datum-dim obj)
            (dtree-root obj))))

(defun make-dtree (n-class datamatrix target
                   &key (max-depth 5) (min-region-samples 1) (n-trial 10)
                        (gain-test #'entropy)
                        (remove-sample-indices? t)
                        (save-parent-node? nil)
                        sample-indices)
  (check-type n-class alexandria:positive-integer)
  (check-type datamatrix (simple-array single-float))
  (check-type target (simple-array fixnum))
  (check-type max-depth alexandria:positive-integer)
  (check-type min-region-samples alexandria:positive-integer)
  (check-type n-trial alexandria:positive-integer)
  (assert (member gain-test (list #'entropy #'gini)))

  (let* ((len (if sample-indices
                  (length sample-indices)
                  (array-dimension datamatrix 0)))
         (dtree (%make-dtree
                 :n-class n-class
                 :class-count-array (make-array n-class :element-type 'single-float)
                 :datum-dim (array-dimension datamatrix 1)
                 :datamatrix datamatrix
                 :target target
                 :max-depth max-depth :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test
                 :remove-sample-indices? remove-sample-indices?
                 :save-parent-node? save-parent-node?
                 :tmp-arr1 (make-array len :element-type 'fixnum :initial-element 0)
                 :tmp-index1 0
                 :tmp-arr2 (make-array len :element-type 'fixnum :initial-element 0)
                 :tmp-index2 0
                 :best-arr1 (make-array len :element-type 'fixnum :initial-element 0)
                 :best-index1 0
                 :best-arr2 (make-array len :element-type 'fixnum :initial-element 0)
                 :best-index2 0)))
    (setf (dtree-root dtree) (make-root-node dtree :sample-indices sample-indices))
    (split-node! (dtree-root dtree))
    (clean-dtree! dtree)
    dtree))

(defun make-rtree (datamatrix target
                   &key (max-depth 5) (min-region-samples 1) (n-trial 10)
                        (gain-test #'variance)
                        (remove-sample-indices? t)
                        (save-parent-node? nil)
                        sample-indices)
  (check-type datamatrix (simple-array single-float))
  (check-type target (simple-array single-float))
  (check-type max-depth alexandria:positive-integer)
  (check-type min-region-samples alexandria:positive-integer)
  (check-type n-trial alexandria:positive-integer)
  (assert (member gain-test (list #'variance)))

  (let* ((len (if sample-indices
                  (length sample-indices)
                  (array-dimension datamatrix 0)))
         (dtree (%make-dtree
                 :datum-dim (array-dimension datamatrix 1)
                 :datamatrix datamatrix
                 :target target
                 :max-depth max-depth :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test
                 :remove-sample-indices? remove-sample-indices?
                 :save-parent-node? save-parent-node?
                 :tmp-arr1 (make-array len :element-type 'fixnum :initial-element 0)
                 :tmp-index1 0
                 :tmp-arr2 (make-array len :element-type 'fixnum :initial-element 0)
                 :tmp-index2 0
                 :best-arr1 (make-array len :element-type 'fixnum :initial-element 0)
                 :best-index1 0
                 :best-arr2 (make-array len :element-type 'fixnum :initial-element 0)
                 :best-index2 0)))
    (setf (dtree-root dtree) (make-root-node dtree :sample-indices sample-indices))
    (split-node! (dtree-root dtree))
    (clean-dtree! dtree)
    dtree))

(defun clean-dtree! (dtree)
  (setf (dtree-datamatrix dtree)  nil
        (dtree-tmp-arr1 dtree)    nil
        (dtree-tmp-index1 dtree)  nil
        (dtree-tmp-arr2 dtree)    nil
        (dtree-tmp-index2 dtree)  nil
        (dtree-best-arr1 dtree)   nil
        (dtree-best-index1 dtree) nil
        (dtree-best-arr2 dtree)   nil
        (dtree-best-index2 dtree) nil)
  dtree)

(defstruct (node (:constructor %make-node)
                 (:print-object %print-node))
  sample-indices n-sample depth test-attribute test-threshold information-gain
  parent-node left-node right-node dtree leaf-index)

(defun %print-node (obj stream)
  (print-unreadable-object (obj stream :type t :identity t)
    (format stream ":TEST ~A :GAIN ~A"
            (list (node-test-attribute obj) (node-test-threshold obj))
            (node-information-gain obj))))

(defun make-root-node (dtree &key sample-indices)
  (let* ((len (array-dimension (dtree-datamatrix dtree) 0))
         (sample-indices
           (if sample-indices
               sample-indices
               (make-array len :element-type 'fixnum :initial-contents (alexandria:iota len)))))
    (%make-node
     :information-gain 1.0
     :sample-indices sample-indices
     :n-sample len
     :depth 0
     :dtree dtree)))

(defun make-node (sample-indices parent-node)
  (%make-node :sample-indices sample-indices
              :n-sample (length sample-indices)
              :depth (1+ (node-depth parent-node))
              :dtree (node-dtree parent-node)))

(defun class-distribution (sample-indices terminate-index dtree)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices)
           (type fixnum terminate-index)
           (type dtree dtree))
  (let ((n-class (dtree-n-class dtree))
        (class-count-array (dtree-class-count-array dtree))
        (target (dtree-target dtree)))
    (declare (type fixnum n-class)
             (type (simple-array fixnum) target)
             (type (simple-array single-float) class-count-array))
    ;; init
    (loop for i fixnum from 0 below n-class do
      (setf (aref class-count-array i) 0.0))
    ;; count
    (loop for i fixnum from 0 below terminate-index do
      (let* ((datum-index (aref sample-indices i))
             (class-label (aref target datum-index)))
        (incf (aref class-count-array class-label) 1.0)))
    ;; divide by sum
    (let ((sum (loop for c single-float across class-count-array summing c single-float)))
      (declare (type single-float sum))
      (loop for i fixnum from 0 below n-class do
        (if (= sum 0.0)
            (setf (aref class-count-array i) (/ 1.0 n-class))
            (setf (aref class-count-array i) (/ (aref class-count-array i) sum)))))
    class-count-array))

(defun node-class-distribution (node)
  (class-distribution (node-sample-indices node)
                      (length (node-sample-indices node))
                      (node-dtree node)))

(defun gini (sample-indices terminate-index dtree)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices)
           (type fixnum terminate-index)
           (type dtree dtree))
  (let ((dist (class-distribution sample-indices terminate-index dtree))
        (sum 0.0)
        (n-class (dtree-n-class dtree)))
    (declare (type (simple-array single-float) dist)
             (type fixnum n-class)
             (type single-float sum))
    (loop for i fixnum from 0 below n-class do
      (let ((pk (aref dist i)))
        (declare (type (single-float 0.0) pk))
        (setf sum (+ sum
                     (if (= pk 0.0)
                         0.0
                         (* pk pk))))))
    (* -1.0 sum)))

(defun entropy (sample-indices terminate-index dtree)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices)
           (type fixnum terminate-index)
           (type dtree dtree))
  (let ((dist (class-distribution sample-indices terminate-index dtree))
        (sum 0.0)
        (n-class (dtree-n-class dtree)))
    (declare (type (simple-array single-float) dist)
             (type fixnum n-class)
             (type single-float sum))
    (loop for i fixnum from 0 below n-class do
      (let ((pk (aref dist i)))
        (declare (type (single-float 0.0) pk))
        (setf sum (+ sum
                     (if (= pk 0.0)
                         0.0
                         (* pk (log pk)))))))
    (* -1.0 sum)))

(declaim (inline square))
(defun square (x)
  (* x x))

;; score function for regression

(defun variance (sample-indices terminate-index rtree)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices)
           (type fixnum terminate-index)
           (type dtree rtree))
  (if (zerop terminate-index)
      0.0
      (let ((len (* terminate-index 1.0))
            (target (dtree-target rtree))
            (sum 0.0))
        (declare (type (simple-array single-float) target)
                 (type single-float sum len))
        (let ((ave (progn
                     (loop for i fixnum from 0 below terminate-index do
                       (incf sum (aref target (aref sample-indices i))))
                     (/ sum len))))
          (declare (type single-float ave))
          (let ((sum-of-squares 0.0))
            (declare (type single-float sum-of-squares))
            (loop for i fixnum from 0 below terminate-index do
              (incf sum-of-squares
                    (square (- (aref target (aref sample-indices i))
                               ave))))
            sum-of-squares)))))

(defun region-min/max (sample-indices datamatrix attribute)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices)
           (type (simple-array single-float) datamatrix)
           (type fixnum attribute))
  (let ((min (aref datamatrix (aref sample-indices 0) attribute))
        (max (aref datamatrix (aref sample-indices 0) attribute)))
    (declare (type single-float min max))
    (loop for index fixnum across sample-indices do
      (let ((elem (aref datamatrix index attribute)))
        (declare (type single-float elem))
        (cond ((< max elem) (setf max elem))
              ((> min elem) (setf min elem)))))
    (values min max)))

(defun make-random-test (node)
  (let* ((dtree (node-dtree node))
         (datamatrix (dtree-datamatrix dtree))
         (attribute (random (the fixnum (dtree-datum-dim dtree))))
         (sample-indices (node-sample-indices node)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array fixnum) sample-indices)
             (type (simple-array single-float) datamatrix)
             (type fixnum attribute))
    (multiple-value-bind (min max)
        (region-min/max sample-indices datamatrix attribute)
      (declare (type single-float min max))
      (let ((threshold (if (= min max) min (random-uniform min max))))
        (declare (type single-float threshold))
        (values attribute threshold)))))

(defun split-sample-indices (sample-indices true-array false-array attribute threshold datamatrix)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices true-array false-array)
           (type fixnum attribute)
           (type single-float threshold)
           (type (simple-array single-float) datamatrix))
  (let ((true-len 0)
        (false-len 0))
    (declare (type fixnum true-len false-len))
    (loop for index fixnum across sample-indices do
      (cond ((>= (aref datamatrix index attribute) threshold)
             (setf (aref true-array true-len) index)
             (incf true-len))
            (t
             (setf (aref false-array false-len) index)
             (incf false-len))))
    (values true-len false-len)))

(defun copy-tmp->best! (dtree)
  (let ((tmp-arr1   (dtree-tmp-arr1 dtree))
        (tmp-index1 (dtree-tmp-index1 dtree))
        (tmp-arr2   (dtree-tmp-arr2 dtree))
        (tmp-index2 (dtree-tmp-index2 dtree))
        (best-arr1   (dtree-best-arr1 dtree))
        (best-arr2   (dtree-best-arr2 dtree)))
    (declare (optimize (speed 3) (safety 0))
             (type fixnum tmp-index1 tmp-index2)
             (type (simple-array fixnum) tmp-arr1 tmp-arr2 best-arr1 best-arr2))
    (loop for i fixnum from 0 below tmp-index1 do
      (setf (aref best-arr1 i) (aref tmp-arr1 i)))
    (loop for i fixnum from 0 below tmp-index2 do
      (setf (aref best-arr2 i) (aref tmp-arr2 i)))
    (setf (dtree-best-index1 dtree) tmp-index1
          (dtree-best-index2 dtree) tmp-index2)))

(defun make-partial-arr (arr len)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) arr)
           (type fixnum len))
  (let ((new-arr (make-array len :element-type 'fixnum)))
    (loop for i fixnum from 0 below len do
      (setf (aref new-arr i) (aref arr i)))
    new-arr))

(defun set-best-children! (node)
  (let* ((dtree (node-dtree node))
         (n-trial (dtree-n-trial dtree))
         (gain-test (dtree-gain-test dtree))
         (max-children-gain most-negative-single-float)
         (left-node (make-node nil node))
         (right-node (make-node nil node)))
    (setf (node-left-node node) left-node
          (node-right-node node) right-node)
    (dotimes (_ n-trial)
      (multiple-value-bind (attribute threshold)
          (make-random-test node)
        (multiple-value-bind (left-len right-len)
            (split-sample-indices (node-sample-indices node)
                                  (dtree-tmp-arr1 dtree) (dtree-tmp-arr2 dtree)
                                  attribute threshold (dtree-datamatrix dtree))
          (setf (dtree-tmp-index1 dtree) left-len
                (dtree-tmp-index2 dtree) right-len)
          (let* ((left-gain  (funcall gain-test (dtree-tmp-arr1 dtree) left-len  dtree))
                 (right-gain (funcall gain-test (dtree-tmp-arr2 dtree) right-len dtree))
                 (parent-size (length (node-sample-indices node)))
                 (children-gain
                   (if (not (rtree? dtree))
                       ;; classification
                       (+ (* -1.0 (/ left-len parent-size)  left-gain)
                          (* -1.0 (/ right-len parent-size) right-gain))
                       ;; regression
                       (+ (* -1.0 left-gain)
                          (* -1.0 right-gain)))))
            (when (< max-children-gain children-gain)
              (copy-tmp->best! dtree)
              (setf max-children-gain children-gain
                    (node-test-attribute node) attribute
                    (node-test-threshold node) threshold
                    (node-information-gain left-node) left-gain
                    (node-information-gain right-node) right-gain))))))
    (setf (node-sample-indices left-node)
          (make-partial-arr (dtree-best-arr1 dtree) (dtree-best-index1 dtree))
          (node-n-sample left-node) (length (node-sample-indices left-node))
          (node-sample-indices right-node)
          (make-partial-arr (dtree-best-arr2 dtree) (dtree-best-index2 dtree))
          (node-n-sample right-node) (length (node-sample-indices right-node)))
    (when (dtree-remove-sample-indices? dtree)
      (setf (node-sample-indices node) nil))
    (when (dtree-save-parent-node? dtree)
      (setf (node-parent-node left-node)  node
            (node-parent-node right-node) node))
    node))

(defun stop-split? (node)
  (or (= (node-information-gain node) 0.0)
      (<= (length (node-sample-indices node))
          (dtree-min-region-samples (node-dtree node)))
      (>= (node-depth node)
          (dtree-max-depth (node-dtree node)))))

(defun split-node! (node)
  (when (and node (not (stop-split? node)))
    (set-best-children! node)
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

(defun do-leaf (fn node)
  (cond ((null node) nil)
        ((and (null (node-left-node node))
              (null (node-right-node node)))
         (funcall fn node))
        ((null (node-left-node node)) (do-leaf fn (node-right-node node)))
        ((null (node-right-node node)) (do-leaf fn (node-left-node node)))
        (t (do-leaf fn (node-left-node node))
           (do-leaf fn (node-right-node node)))))

(defun find-leaf (node datamatrix datum-index)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum datum-index)
           (type (simple-array single-float) datamatrix))
  (cond ((null node) nil)
        ((null (node-test-attribute node)) node)
        (t (let ((attribute (node-test-attribute node))
                 (threshold (node-test-threshold node)))
             (declare (type fixnum attribute)
                      (type single-float threshold))
             (if (>= (aref datamatrix datum-index attribute) threshold)
                 (find-leaf (node-left-node node) datamatrix datum-index)
                 (find-leaf (node-right-node node) datamatrix datum-index))))))

;; decision-tree prediction

(defun argmax (arr)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array single-float) arr))
  (let ((max most-negative-single-float)
        (max-i 0))
    (declare (type single-float max)
             (type fixnum max-i))
    (loop for i fixnum from 0 below (length arr) do
      (when (> (aref arr i) max)
        (setf max (aref arr i)
              max-i i)))
    max-i))

(defun predict-dtree (dtree datamatrix datum-index)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array single-float) datamatrix)
           (type fixnum datum-index))
  (let ((dist (node-class-distribution (find-leaf (dtree-root dtree) datamatrix datum-index))))
    (declare (type (simple-array single-float) dist))
    (argmax dist)))

(defun calc-accuracy (n-correct len &key quiet-p)
  (let ((accuracy (* (/ n-correct len) 100.0)))
    (if (not quiet-p)
        (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" accuracy n-correct len))
    (values accuracy n-correct len)))

(defun test-dtree (dtree datamatrix target &key quiet-p)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array single-float) datamatrix)
           (type (simple-array fixnum (*)) target))
  (let ((n-correct 0)
        (len (length target)))
    (declare (type fixnum n-correct len))
    (loop for i fixnum from 0 below len do
      (when (= (predict-dtree dtree datamatrix i)
               (aref target i))
        (incf n-correct)))
    (calc-accuracy n-correct len :quiet-p quiet-p)))

;; regression-tree prediction

(defun node-regression-mean (node)
  (declare (optimize (speed 3) (safety 0))
           (type node node))
  (let ((rtree (node-dtree node)))
    (declare (type dtree rtree))
    (let ((target (dtree-target rtree))
          (pred 0.0)
          (sample-indices (node-sample-indices node)))
      (declare (type (simple-array single-float) target)
               (type single-float pred)
               (type (simple-array fixnum) sample-indices))
      (let ((len (* (length sample-indices) 1.0)))
        (declare (type single-float len))
        (if (zerop len)
            pred
            (progn
              (loop for i fixnum across sample-indices do
                (incf pred (aref target i)))
              (/ pred len)))))))

(defun predict-rtree (rtree datamatrix datum-index)
  (node-regression-mean (find-leaf (dtree-root rtree) datamatrix datum-index)))

(defun test-rtree (rtree datamatrix target &key quiet-p)
  (declare (optimize (speed 3) (safety 0))
           (type dtree rtree)
           (type (simple-array single-float) datamatrix target))
  (let ((sum-square-error 0.0)
        (n-data (array-dimension datamatrix 0)))
    (declare (type single-float sum-square-error)
             (type fixnum n-data))
    (loop for i fixnum from 0 below n-data do
      (incf sum-square-error (square (- (predict-rtree rtree datamatrix i)
                                        (aref target i)))))
    (setf sum-square-error (sqrt (/ sum-square-error n-data)))
    (when (null quiet-p)
      (format t "RMSE: ~A~%" sum-square-error))
    sum-square-error))

;;; forest

(defstruct (forest (:constructor %make-forest)
                   (:print-object %print-forest))
  n-tree bagging-ratio datamatrix target dtree-list
  ;; for classification
  n-class class-count-array
  datum-dim max-depth min-region-samples n-trial gain-test
  ;; for global refinement
  n-leaf index-offset)

(defun %print-forest (obj stream)
  (if (forest-n-class obj)
      (format stream "#S(FOREST :N-CLASS ~A :N-TREE ~A)"
              (forest-n-class obj)
              (forest-n-tree obj))
      (format stream "#S(REGRESSION-FOREST :N-TREE ~A)"
              (forest-n-tree obj))))

(defun bootstrap-sample-indices (n datamatrix)
  (let ((len (array-dimension datamatrix 0))
        (arr (make-array n :element-type 'fixnum :initial-element 0)))
    (loop for i from 0 below n do
      (setf (aref arr i) (random len)))
    arr))

(defun balanced-bootstrap-sample-indices (n n-class target)
  "Bagging classifiers induced over balanced bootstrap samples. Also known as Balanced Random Forest.

This implementation counts the number of examples in each class, which might be inefficient.

> In this work we provide such an explanation, and we conclude that
> in almost all imbalanced scenarios, practitioners should bag classifiers induced over
> balanced bootstrap samples.

Wallace, Byron C., et al. ``Class imbalance, redux.''
2011 IEEE 11th International Conference on Data Mining (ICDM), 2011."
  (let* ((len (array-dimension target 0))
         (counters/class (make-array n-class :element-type 'fixnum :initial-element 0))
         (indices/class (coerce (loop :repeat n-class
                                   :collect (make-array len :element-type 'fixnum :initial-element -1))
                                'vector))
         (arr (make-array n :element-type 'fixnum :initial-element 0)))
    ;; collect indices for each class
    (loop for i from 0 below len do
         (let* ((class (aref target i))
                (index/class (aref counters/class class)))
           (setf (aref (aref indices/class class) index/class) i)
           (setf (aref counters/class class) (1+ index/class))))
    ;; collect the balanced bootstrap indices 
    (loop
       for i from 0 below n
       with class = 0
       for counter = (aref counters/class class)
       do
         (when (< 0 counter) 
           (setf (aref arr i)
                 (aref (aref indices/class class)
                       (random counter))))
         (setf class (mod (1+ class) n-class)))
    arr))

(defun make-forest (n-class datamatrix target
                    &key (n-tree 100) (bagging-ratio 0.1) (max-depth 5) (min-region-samples 1)
                         (n-trial 10) (gain-test #'entropy)
                         (remove-sample-indices? t) (save-parent-node? nil) (balance nil))
  (let ((forest (%make-forest
                 :n-tree n-tree
                 :bagging-ratio bagging-ratio
                 :datamatrix datamatrix
                 :target target
                 :n-class n-class
                 :class-count-array (make-array n-class :element-type 'single-float)
                 :datum-dim (array-dimension datamatrix 1)
                 :max-depth max-depth
                 :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test
                 :index-offset (make-array n-tree :element-type 'fixnum :initial-element 0))))
    ;; make dtree list
    (push-ntimes n-tree (forest-dtree-list forest)
      (prog1
          (make-dtree n-class datamatrix target
                      :max-depth max-depth
                      :min-region-samples min-region-samples
                      :n-trial n-trial
                      :gain-test gain-test
                      :remove-sample-indices? remove-sample-indices?
                      :save-parent-node? save-parent-node?
                      :sample-indices (if balance
                                          (balanced-bootstrap-sample-indices
                                           (floor (* (array-dimension datamatrix 0) bagging-ratio))
                                           n-class
                                           target)
                                          (bootstrap-sample-indices
                                           (floor (* (array-dimension datamatrix 0) bagging-ratio))
                                           datamatrix)))
        (format t ".")
        (force-output)))
    (terpri)
    ;; set dtree-id
    (loop for dtree in (forest-dtree-list forest)
          for i from 0
          do (setf (dtree-id dtree) i))
    ;; set leaf-id
    (set-leaf-index-forest! forest)
    (setf (forest-n-leaf forest)
          (apply #'+ (mapcar #'dtree-max-leaf-index
                             (forest-dtree-list forest))))
    forest))

(defun make-regression-forest (datamatrix target
                    &key (n-tree 100) (bagging-ratio 0.1) (max-depth 5) (min-region-samples 1)
                      (n-trial 10) (gain-test #'variance)
                      (remove-sample-indices? t) (save-parent-node? nil))
  (let ((forest (%make-forest
                 :n-tree n-tree
                 :bagging-ratio bagging-ratio
                 :datamatrix datamatrix
                 :target target
                 :datum-dim (array-dimension datamatrix 1)
                 :max-depth max-depth
                 :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test
                 :index-offset (make-array n-tree :element-type 'fixnum :initial-element 0))))
    ;; make dtree list
    (push-ntimes n-tree (forest-dtree-list forest)
      (prog1
          (make-rtree datamatrix target
                      :max-depth max-depth
                      :min-region-samples min-region-samples
                      :n-trial n-trial
                      :gain-test gain-test
                      :remove-sample-indices? remove-sample-indices?
                      :save-parent-node? save-parent-node?
                      :sample-indices (bootstrap-sample-indices
                                       (floor (* (array-dimension datamatrix 0) bagging-ratio))
                                       datamatrix))
        (format t ".")
        (force-output)))
    (terpri)
    ;; set dtree-id
    (loop for dtree in (forest-dtree-list forest)
          for i from 0
          do (setf (dtree-id dtree) i))
    ;; set leaf-id
    (set-leaf-index-forest! forest)
    (setf (forest-n-leaf forest)
          (apply #'+ (mapcar #'dtree-max-leaf-index
                             (forest-dtree-list forest))))
    forest))

(defun class-distribution-forest (forest datamatrix datum-index)
  (let ((n-class (forest-n-class forest))
        (n-tree (forest-n-tree forest))
        (class-count-array (forest-class-count-array forest)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array single-float) datamatrix class-count-array)
             (type fixnum datum-index n-class n-tree))
    ;; init forest-class-count-array
    (loop for i fixnum from 0 below n-class do
      (setf (aref class-count-array i) 0.0))
    ;; whole count
    (dolist (dtree (forest-dtree-list forest))
      (let ((dist (node-class-distribution
                   (find-leaf (dtree-root dtree) datamatrix datum-index))))
        (declare (type (simple-array single-float) dist))
        (loop for i fixnum from 0 below n-class do
          (incf (aref class-count-array i) (aref dist i)))))
    ;; divide by n-tree
    (loop for i fixnum from 0 below n-class do
      (setf (aref class-count-array i)
            (/ (aref class-count-array i) n-tree)))
    class-count-array))

(defun predict-forest (forest datamatrix datum-index)
  (declare (optimize (speed 3) (safety 0))
           (type forest forest)
           (type (simple-array single-float) datamatrix)
           (type fixnum datum-index))
  (let ((dist (class-distribution-forest forest datamatrix datum-index)))
    (declare (type (simple-array single-float) dist))
    (argmax dist)))

(defun test-forest (forest datamatrix target &key quiet-p)
  (declare (optimize (speed 3) (safety 0))
           (type forest forest)
           (type (simple-array single-float) datamatrix)
           (type (simple-array fixnum) target))
  (let ((n-correct 0)
        (len (length target)))
    (declare (type fixnum n-correct len))
    (loop for i fixnum from 0 below len do
      (when (= (predict-forest forest datamatrix i)
               (aref target i))
        (incf n-correct)))
    (calc-accuracy n-correct len :quiet-p quiet-p)))

;; predict regression forest

(defun predict-regression-forest (forest datamatrix datum-index)
  (/ (loop for rtree in (forest-dtree-list forest)
             sum (predict-rtree rtree datamatrix datum-index))
       (forest-n-tree forest)))

(defun test-regression-forest (forest datamatrix target &key quiet-p)
  (let ((sum-square-error 0.0)
        (n-data (array-dimension datamatrix 0)))
    (loop for i fixnum from 0 below n-data do
      (incf sum-square-error (square (- (predict-regression-forest forest datamatrix i)
                                        (aref target i)))))
    (setf sum-square-error (sqrt (/ sum-square-error n-data)))
    (when (null quiet-p)
      (format t "RMSE: ~A~%" sum-square-error))
    sum-square-error))

;;; Global refinement (Classification)

(defun make-refine-vector (forest datamatrix datum-index)
  (let ((index-offset (forest-index-offset forest))
        (n-tree (forest-n-tree forest)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array single-float) datamatrix)
             (type (simple-array fixnum) index-offset)
             (type fixnum datum-index n-tree))
    (let ((leaf-index-list
            (mapcar/pmapcar
             (lambda (dtree)
               (let ((node (find-leaf (dtree-root dtree) datamatrix datum-index)))
                 (node-leaf-index node)))
             (forest-dtree-list forest))))
      (let ((sv-index (make-array (forest-n-tree forest) :element-type 'fixnum))
            (sv-val (make-array (forest-n-tree forest)
                                :element-type 'single-float :initial-element 1.0)))
        (declare (type (simple-array fixnum) sv-index)
                 (type (simple-array single-float) sv-val))
        (loop for i fixnum from 0 below n-tree
              for index fixnum in leaf-index-list
              do (setf (aref sv-index i) (+ index (aref index-offset i))))
        (clol.vector:make-sparse-vector sv-index sv-val)))))

(defun set-leaf-index! (dtree)
  (setf (dtree-max-leaf-index dtree) 0)
  (do-leaf (lambda (node)
             (setf (node-leaf-index node) (dtree-max-leaf-index dtree))
             (incf (dtree-max-leaf-index dtree)))
    (dtree-root dtree)))

(defun set-leaf-index-forest! (forest)
  (dolist (dtree (forest-dtree-list forest))
    (set-leaf-index! dtree))
  (let ((sum 0)
        (offset (forest-index-offset forest)))
    (setf (aref offset 0) 0)
    (loop for dtree in (forest-dtree-list forest)
          for i from 1 below (forest-n-tree forest)
          do (setf sum (+ sum (dtree-max-leaf-index dtree)))
             (setf (aref offset i) sum))))

(defun make-refine-learner (forest &optional (gamma 10.0))
  (let ((n-class (forest-n-class forest))
        (input-dim (loop for n-leaves in (mapcar #'dtree-max-leaf-index (forest-dtree-list forest))
                         sum n-leaves)))
    (if (> n-class 2)
        (clol:make-one-vs-rest input-dim n-class 'sparse-arow gamma)
        (clol:make-sparse-arow input-dim gamma))))

(defun predict-refine-learner (forest refine-learner datamatrix datum-index)
  (let ((sv (make-refine-vector forest datamatrix datum-index)))
    (etypecase refine-learner
      (cl-online-learning::sparse-arow
       (if (> (clol:sparse-arow-predict refine-learner sv) 0.0) 1 0))
      (cl-online-learning::one-vs-rest
       (clol:one-vs-rest-predict refine-learner sv)))))

;; Make vector of leaf-index vectors
(defun make-refine-dataset (forest datamatrix)
  (let ((index-offset (forest-index-offset forest))
        (len (array-dimension datamatrix 0))
        (n-tree (forest-n-tree forest)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array single-float) datamatrix)
             (type (simple-array fixnum) index-offset)
             (type fixnum len n-tree))
    (let ((refine-dataset (make-array len)))
      (loop for i from 0 below len do
        (setf (aref refine-dataset i) (make-array n-tree :element-type 'fixnum)))
      (mapc/pmapc
       (lambda (dtree)
         (let* ((tree-id (dtree-id dtree))
                (offset (aref index-offset tree-id)))
           (declare (type fixnum tree-id offset))
           (loop for i fixnum from 0 below len do
             (let ((leaf-index (node-leaf-index (find-leaf (dtree-root dtree) datamatrix i)))
                   (refine-datum (svref refine-dataset i)))
               (declare (type fixnum leaf-index)
                        (type (simple-array fixnum) refine-datum))
               (setf (aref refine-datum tree-id) (+ leaf-index offset)))))
         (format t ".")
         (force-output))
       (forest-dtree-list forest))
      (terpri)
      refine-dataset)))

(defun train-refine-learner-binary (refine-learner refine-dataset target)
  (let* ((n-tree (length (svref refine-dataset 0)))
         (sv-index (make-array n-tree :element-type 'fixnum :initial-element 0))
         (sv-val (make-array n-tree :element-type 'single-float :initial-element 1.0))
         (sv (clol.vector:make-sparse-vector sv-index sv-val)))
    (loop for i from 0 below (length refine-dataset) do
      (setf (clol.vector:sparse-vector-index-vector sv) (svref refine-dataset i))
      (clol:sparse-arow-update refine-learner sv
                               (if (= (aref target i) 0) -1.0 1.0)))))

(defun test-refine-learner-binary (refine-learner refine-dataset target &key quiet-p)
  (let* ((len (length refine-dataset))
         (n-tree (length (svref refine-dataset 0)))
         (sv-index (make-array n-tree :element-type 'fixnum :initial-element 0))
         (sv-val (make-array n-tree :element-type 'single-float :initial-element 1.0))
         (sv (clol.vector:make-sparse-vector sv-index sv-val))
         (n-correct 0))
    (loop for i from 0 below (length refine-dataset) do
      (setf (clol.vector:sparse-vector-index-vector sv) (svref refine-dataset i))
      (when (= (aref target i)
               (if (> (clol:sparse-arow-predict refine-learner sv) 0.0) 1 0))
        (incf n-correct)))
    (calc-accuracy n-correct len :quiet-p quiet-p)))
  
;; Parallel multiclass classifiers

(defun train-refine-learner-multiclass (refine-learner refine-dataset target)
  (let* ((len (length refine-dataset))
         (n-tree (length (svref refine-dataset 0)))
         (n-class (clol::one-vs-rest-n-class refine-learner))
         (learners (clol::one-vs-rest-learners-vector refine-learner))
         (sv-index (make-array n-tree :element-type 'fixnum :initial-element 0))
         (sv-val (make-array n-tree :element-type 'single-float :initial-element 1.0))
         (sv-vec (make-array n-class)))
    (loop for i fixnum from 0 below n-class do
      (setf (svref sv-vec i)
            (clol.vector:make-sparse-vector sv-index sv-val)))
    (dotimes/pdotimes (class-id n-class)
      (loop for datum-id fixnum from 0 below len do
        (setf (clol.vector:sparse-vector-index-vector (svref sv-vec class-id))
              (svref refine-dataset datum-id))
        (clol:sparse-arow-update (svref learners class-id)
                                 (svref sv-vec class-id)
                                 (if (= (aref target datum-id) class-id) 1.0 -1.0))))))

;; dataset: simple vector of refine-dataset
(defun set-activation-matrix!
    (activation-matrix refine-learner n-class sv-vec refine-dataset cycle end-of-mini-batch)
  (dotimes/pdotimes (class-id n-class) 
    (let ((learner (svref (clol::one-vs-rest-learners-vector refine-learner) class-id))
          (input (svref sv-vec class-id)))
      (loop for i from 0 below end-of-mini-batch do
        (setf (clol.vector:sparse-vector-index-vector input)
              (svref refine-dataset (+ (* (array-dimension activation-matrix 0) cycle) i)))
        (setf (aref activation-matrix i class-id)
              (funcall (clol::one-vs-rest-learner-activate refine-learner)
                       input
                       (funcall (clol::one-vs-rest-learner-weight refine-learner) learner)
                       (funcall (clol::one-vs-rest-learner-bias refine-learner) learner)))))))

(defun maximize-activation/count (activation-matrix end-of-mini-batch target cycle)
  (let ((n-correct 0))
    (loop for i from 0 below end-of-mini-batch do
      (let ((max-activation most-negative-single-float)
            (max-class 0))
        (loop for j from 0 below (array-dimension activation-matrix 1) do
          (when (> (aref activation-matrix i j) max-activation)
            (setf max-activation (aref activation-matrix i j)
                  max-class j)))
        (when (= max-class (aref target (+ (* (array-dimension activation-matrix 0) cycle) i)))
          (incf n-correct))))
    n-correct))

(defun test-refine-learner-multiclass (refine-learner refine-dataset target &key quiet-p (mini-batch-size 1000))
  (let* ((len (length refine-dataset))
         (n-tree (length (svref refine-dataset 0)))
         (n-class (clol::one-vs-rest-n-class refine-learner))
         (activation-matrix (make-array (list mini-batch-size n-class) :element-type 'single-float :initial-element 0.0))
         (sv-index (make-array n-tree :element-type 'fixnum :initial-element 0))
         (sv-val (make-array n-tree :element-type 'single-float :initial-element 1.0))
         (sv-vec (make-array n-class))
         (n-correct 0))
    ;; init sv-vec
    (loop for i fixnum from 0 below n-class do
      (setf (svref sv-vec i)
            (clol.vector:make-sparse-vector sv-index sv-val)))
    (multiple-value-bind (max-cycle end-of-mini-batch)
        (floor len mini-batch-size)
      (when (> max-cycle 0)
        (loop for cycle from 0 below max-cycle do
          (set-activation-matrix! activation-matrix refine-learner n-class sv-vec
                                 refine-dataset cycle mini-batch-size)
          (incf n-correct
                (maximize-activation/count activation-matrix mini-batch-size target cycle))))
      ;; remain
      (set-activation-matrix! activation-matrix refine-learner n-class sv-vec
                             refine-dataset max-cycle end-of-mini-batch)
      (incf n-correct
            (maximize-activation/count activation-matrix end-of-mini-batch target max-cycle))

      (calc-accuracy n-correct len :quiet-p quiet-p))))

(defun train-refine-learner (refine-learner refine-dataset target)
  (etypecase refine-learner
    (cl-online-learning::sparse-arow
     (train-refine-learner-binary refine-learner refine-dataset target))
    (cl-online-learning::one-vs-rest
     (train-refine-learner-multiclass refine-learner refine-dataset target))))

(defun test-refine-learner (refine-learner refine-dataset target
                            &key quiet-p (mini-batch-size 1000))
  (etypecase refine-learner
    (cl-online-learning::sparse-arow
     (test-refine-learner-binary refine-learner refine-dataset target :quiet-p quiet-p))
    (cl-online-learning::one-vs-rest
     (test-refine-learner-multiclass refine-learner refine-dataset target
                                     :quiet-p quiet-p :mini-batch-size mini-batch-size))))

;; Training process with detection of convergence

(defun train-refine-learner-process-inner
    (refine-learner train-dataset train-target test-dataset test-target &key (max-epoch 100))
  (let ((tmp-learner nil)
        (max-accuracy 0.0))
    (loop repeat max-epoch do
      (setf tmp-learner
            (etypecase refine-learner
              (cl-online-learning::sparse-arow (clol::copy-sparse-arow refine-learner))
              (cl-online-learning::one-vs-rest (clol::copy-one-vs-rest refine-learner))))
      (train-refine-learner refine-learner train-dataset train-target)
      (let ((accuracy (test-refine-learner refine-learner test-dataset test-target :quiet-p t)))
        (format t "Accuracy: ~A~%" accuracy)
        (if (> accuracy max-accuracy)
            (setf max-accuracy accuracy)
            (return))))
    (values tmp-learner max-accuracy)))

(defmacro train-refine-learner-process
    (refine-learner train-dataset train-target test-dataset test-target &key (max-epoch 100))
  `(setf ,refine-learner
         (train-refine-learner-process-inner
          ,refine-learner ,train-dataset ,train-target ,test-dataset ,test-target
          :max-epoch ,max-epoch)))

(defun cross-validation-forest-with-refine-learner
    (n-fold n-class datamatrix target
     &key (n-tree 100) (bagging-ratio 0.1) (max-depth 5) (min-region-samples 1)
       (n-trial 10) (gain-test #'entropy) (remove-sample-indices? t) (gamma 10.0))
  (let* ((datum-dim (array-dimension datamatrix 1))
         (accuracy-sum 0)
         (total-size (array-dimension datamatrix 0))
         (test-size (floor (/ total-size n-fold)))
         (train-size (- total-size test-size))
         (test-datamatrix (make-array (list test-size datum-dim)
                                       :element-type 'single-float :initial-element 0.0))
         (test-target (make-array test-size :element-type 'fixnum :initial-element 0))
         (train-datamatrix (make-array (list train-size datum-dim)
                                       :element-type 'single-float :initial-element 0.0))
         (train-target (make-array train-size :element-type 'fixnum :initial-element 0)))
    (loop for n from 0 below n-fold do
      ;; Init train/test datamatrix/target
      (loop for i from 0 below (* n test-size) do
        (setf (aref train-target i) (aref target i))
        (loop for j from 0 below datum-dim do
          (setf (aref train-datamatrix i j) (aref datamatrix i j))))
      (loop for i from (* n test-size) below (* (1+ n) test-size) do
        (setf (aref test-target (- i (* n test-size))) (aref target i))
        (loop for j from 0 below datum-dim do
          (setf (aref test-datamatrix (- i (* n test-size)) j) (aref datamatrix i j))))
      (loop for i from (* (1+ n) test-size) below total-size do
        (setf (aref train-target (- i test-size)) (aref target i))
        (loop for j from 0 below datum-dim do
          (setf (aref train-datamatrix (- i test-size) j) (aref datamatrix i j))))
      ;; Build a random-forest and make a learner and datasets for Global refinement
      (let* ((forest (make-forest n-class train-datamatrix train-target
                                  :n-tree n-tree :bagging-ratio bagging-ratio :max-depth max-depth
                                  :min-region-samples min-region-samples :n-trial n-trial
                                  :gain-test gain-test :remove-sample-indices? remove-sample-indices?))
             (refine-learner (make-refine-learner forest gamma))
             (refine-train-dataset (make-refine-dataset forest train-datamatrix))
             (refine-test-dataset  (make-refine-dataset forest test-datamatrix)))
        (train-refine-learner-process refine-learner
                                      refine-train-dataset train-target
                                      refine-test-dataset  test-target)
        (incf accuracy-sum
              (test-refine-learner refine-learner refine-test-dataset test-target)))
      ;; cleanup
      #+sbcl (sb-ext:gc :full t))
    (format t "Average accuracy: ~f%~%" (/ accuracy-sum n-fold))))

;;; Global refinement (Regression)

(defun make-regression-refine-learner (forest &optional (gamma 0.99))
  (assert (null (forest-n-class forest)))
  (let ((input-dim (loop for n-leaves in (mapcar #'dtree-max-leaf-index (forest-dtree-list forest))
                         sum n-leaves)))
    (clol:make-sparse-rls input-dim gamma)))

(defun make-regression-refine-dataset (forest datamatrix)
  (let ((index-offset (forest-index-offset forest))
        (len (array-dimension datamatrix 0))
        (n-tree (forest-n-tree forest)))
    ;; (declare (optimize (speed 3) (safety 0))
    ;;          (type (simple-array single-float) datamatrix)
    ;;          (type (simple-array fixnum) index-offset)
    ;;          (type fixnum len n-tree))
    (let ((refine-dataset (make-array len)))
      (loop for i from 0 below len do
        (setf (aref refine-dataset i)
              (cons (make-array n-tree :element-type 'fixnum)
                    (make-array n-tree :element-type 'single-float))))
      (mapc ;mapc/pmapc
       (lambda (dtree)
         (let* ((tree-id (dtree-id dtree))
                (offset (aref index-offset tree-id)))
           ;; (declare (type fixnum tree-id offset))
           (loop for i fixnum from 0 below len do
             (let* ((leaf (find-leaf (dtree-root dtree) datamatrix i))
                    (leaf-index (node-leaf-index leaf))
                    (leaf-mean (node-regression-mean leaf))
                    (refine-datum-index (car (svref refine-dataset i)))
                    (refine-datum-value (cdr (svref refine-dataset i))))
               ;; (declare (type fixnum leaf-index)
               ;;          (type (simple-array fixnum) refine-datum))
               (setf (aref refine-datum-index tree-id) (+ leaf-index offset))
               (setf (aref refine-datum-value tree-id) leaf-mean)))))
       (forest-dtree-list forest))
      refine-dataset)))

(defun make-regression-refine-vector (forest datamatrix datum-index)
  (let ((index-offset (forest-index-offset forest))
        (n-tree (forest-n-tree forest)))
    ;; (declare (optimize (speed 3) (safety 0))
    ;;          (type (simple-array single-float) datamatrix)
    ;;          (type (simple-array fixnum) index-offset)
    ;;          (type fixnum datum-index n-tree))
    (let ((leaf-index-val-pair-list
            (mapcar ;mapcar/pmapcar
             (lambda (dtree)
               (let ((node (find-leaf (dtree-root dtree) datamatrix datum-index)))
                 (cons (node-leaf-index node)
                       (node-regression-mean node))))
             (forest-dtree-list forest))))
      (let ((sv-index (make-array (forest-n-tree forest) :element-type 'fixnum))
            (sv-val (make-array (forest-n-tree forest) :element-type 'single-float)))
        ;; (declare (type (simple-array fixnum) sv-index)
        ;;          (type (simple-array single-float) sv-val))
        (loop for i fixnum from 0 below n-tree
              for index-val-pair in leaf-index-val-pair-list
              do (setf (aref sv-index i) (+ (car index-val-pair) (aref index-offset i))
                       (aref sv-val i) (cdr index-val-pair)))
        (clol.vector:make-sparse-vector sv-index sv-val)))))

(defun predict-regression-refine-learner (forest refine-learner datamatrix datum-index)
  (let ((sv (make-regression-refine-vector forest datamatrix datum-index)))
    (clol:sparse-rls-predict refine-learner sv)))

(defun train-regression-refine-learner (refine-learner refine-dataset target)
  (let* ((n-tree (length (car (svref refine-dataset 0))))
         (sv-index (make-array n-tree :element-type 'fixnum :initial-element 0))
         (sv-val (make-array n-tree :element-type 'single-float))
         (sv (clol.vector:make-sparse-vector sv-index sv-val)))
    (loop for i from 0 below (length refine-dataset) do
      (setf (clol.vector:sparse-vector-index-vector sv) (car (svref refine-dataset i))
            (clol.vector:sparse-vector-value-vector sv) (cdr (svref refine-dataset i)))
      (clol:sparse-rls-update refine-learner sv (aref target i)))))

(defun test-regression-refine-learner (refine-learner refine-dataset target &key quiet-p)
  (flet ((sq (x) (* x x)))
    (let* ((len (length refine-dataset))
           (n-tree (length (car (svref refine-dataset 0))))
           (sv-index (make-array n-tree :element-type 'fixnum :initial-element 0))
           (sv-val (make-array n-tree :element-type 'single-float))
           (sv (clol.vector:make-sparse-vector sv-index sv-val))
           (sum-square-error 0.0))
      (loop for i from 0 below (length refine-dataset) do
        (setf (clol.vector:sparse-vector-index-vector sv) (car (svref refine-dataset i))
              (clol.vector:sparse-vector-value-vector sv) (cdr (svref refine-dataset i)))
        (incf sum-square-error
              (sq (- (clol:sparse-rls-predict refine-learner sv)
                     (aref target i)))))
      (let ((rmse (sqrt (/ sum-square-error len))))
        (if (not quiet-p)
            (format t "RMSE: ~A~%" rmse))
        rmse))))



;;; Global pruning

(defun make-l2-norm-multiclass (learner)
  (let* ((dim (clol::one-vs-rest-input-dimension learner))
         (arr (make-array dim :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 below (clol::one-vs-rest-n-class learner) do
      (let* ((sub-learner (svref (clol::one-vs-rest-learners-vector learner) i))
             (weight-vec (funcall (clol::one-vs-rest-learner-weight learner) sub-learner)))
        (loop for j from 0 below dim do
          (setf (aref arr j) (+ (aref arr j) (square (aref weight-vec j)))))))
    arr))

(defun make-l2-norm-binary (learner)
  (let* ((dim (clol::sparse-arow-input-dimension learner))
         (arr (make-array dim :element-type 'single-float :initial-element 0.0))
         (weight-vec (clol::sparse-arow-weight learner)))
    (loop for i from 0 below dim do
      (setf (aref arr i) (square (aref weight-vec i))))
    arr))

(defun make-l2-norm (learner)
  (etypecase learner
    (cl-online-learning::one-vs-rest (make-l2-norm-multiclass learner))
    (cl-online-learning::sparse-arow (make-l2-norm-binary learner))))

;; Find leaf-parent

(defun leaf? (node)
  (node-leaf-index node))

(defun collect-leaf-parent (forest)
  (let ((parent-list nil))
    (labels ((push-parent-node (node)
               (cond ((null node) nil)
                     ((and (leaf? (node-left-node node)) (leaf? (node-right-node node)))
                      (push node parent-list))
                     ((leaf? (node-left-node node))  (push-parent-node (node-right-node node)))
                     ((leaf? (node-right-node node)) (push-parent-node (node-left-node node)))
                     (t (push-parent-node (node-left-node node))
                        (push-parent-node (node-right-node node))))))
      (dolist (dtree (forest-dtree-list forest))
        (push-parent-node (dtree-root dtree)))
      parent-list)))

;; Sort by L2 norm of children

(defun children-l2-norm (node l2-norm-arr forest)
  (let* ((left-leaf-index (node-leaf-index (node-left-node node)))
         (right-leaf-index (node-leaf-index (node-right-node node)))
         (dtree-index (dtree-id (node-dtree node)))
         (offset (aref (forest-index-offset forest) dtree-index))
         (left-leaf-norm (aref l2-norm-arr (+ left-leaf-index offset)))
         (right-leaf-norm (aref l2-norm-arr (+ right-leaf-index offset))))
    (+ left-leaf-norm right-leaf-norm)))

(defun collect-leaf-parent-sorted (forest learner)
  (let ((l2-norm-arr (make-l2-norm learner))
        (leaf-parent (collect-leaf-parent forest)))
    (sort leaf-parent
          (lambda (node1 node2)
            (< (children-l2-norm node1 l2-norm-arr forest)
               (children-l2-norm node2 l2-norm-arr forest))))))

;; Delete non-significant nodes

(defun delete-children! (node)
  (setf (node-test-attribute node) nil
        (node-test-threshold node) nil
        (node-left-node node) nil
        (node-right-node node) nil)
  node)

(defun pruning! (forest learner &optional (pruning-rate 0.1) (min-depth 1))
  (let* ((leaf-parents (collect-leaf-parent-sorted forest learner))
         (pruning-size (floor (* (length leaf-parents) pruning-rate))))
    (loop for i from 0 below pruning-size
          for node in leaf-parents
          do (when (>= (node-depth node) min-depth)
               (delete-children! node)))
    (set-leaf-index-forest! forest)))
