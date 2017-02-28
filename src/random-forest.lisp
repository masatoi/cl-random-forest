(in-package :cl-user)
(defpackage cl-random-forest
  (:use :cl)
  (:nicknames :clrf))

(in-package :cl-random-forest)

;;; decision tree

(defstruct (dtree (:constructor %make-dtree)
                  (:print-object %print-dtree))
  n-class class-count-array datum-dim datamatrix target
  root max-depth min-region-samples n-trial gain-test
  tmp-arr1 tmp-index1 tmp-arr2 tmp-index2
  best-arr1 best-index1 best-arr2 best-index2)

(defun %print-dtree (obj stream)
  (format stream "#S(DTREE :N-CLASS ~A :DATUM-DIM ~A :ROOT ~A)"
          (dtree-n-class obj)
          (dtree-datum-dim obj)
          (dtree-root obj)))

(defun make-dtree (n-class datum-dim datamatrix target
                   &key (max-depth 5) (min-region-samples 1) (n-trial 10)
                     (gain-test #'entropy) sample-indices)
  (let* ((len (array-dimension datamatrix 0))
         (dtree (%make-dtree
                 :n-class n-class
                 :class-count-array (make-array n-class :element-type 'double-float)
                 :datum-dim datum-dim
                 :datamatrix datamatrix
                 :target target
                 :max-depth max-depth :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test
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
                     (let ((len (array-dimension (dtree-datamatrix dtree) 0)))
                       (make-array len :element-type 'fixnum :initial-contents (alexandria:iota len))))
   :depth 0
   :dtree dtree))

(defun make-node (sample-indices parent-node)
  (%make-node :sample-indices sample-indices
              :depth (1+ (node-depth parent-node))
              :dtree (node-dtree parent-node)))

(defun class-distribution (sample-indices terminate-index dtree)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices)
           (type fixnum terminate-index))
  (let ((n-class (dtree-n-class dtree))
        (class-count-array (dtree-class-count-array dtree))
        (target (dtree-target dtree)))
    (declare (type fixnum n-class)
             (type (simple-array fixnum) target)
             (type (simple-array double-float) class-count-array))
    ;; init
    (loop for i fixnum from 0 to (1- n-class) do
      (setf (aref class-count-array i) 0d0))
    ;; count
    (loop for i fixnum from 0 to (1- terminate-index) do
      (let* ((datum-index (aref sample-indices i))
             (class-label (aref target datum-index)))
        (incf (aref class-count-array class-label) 1d0)))
    ;; divide by sum
    (let ((sum (loop for c double-float across class-count-array summing c double-float)))
      (loop for i fixnum from 0 to (1- n-class) do
        (if (= sum 0d0)
            (setf (aref class-count-array i) (/ 1d0 n-class))
            (setf (aref class-count-array i) (/ (aref class-count-array i) sum)))))
    class-count-array))

(defun node-class-distribution (node)
  (class-distribution (node-sample-indices node)
                      (length (node-sample-indices node))
                      (node-dtree node)))

;; (defun gini (node)
;;   (* -1d0
;;      (loop for pk double-float across (class-distribution node)
;;            summing
;;            (* pk (- 1d0 pk)))))

(defun entropy (sample-indices terminate-index dtree)
  (let ((dist (class-distribution sample-indices terminate-index dtree))
        (sum 0d0)
        (n-class (dtree-n-class dtree)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array fixnum) sample-indices)
             (type (simple-array double-float) dist)
             (type fixnum terminate-index n-class)
             (type double-float sum))
    (loop for i fixnum from 0 to (1- n-class) do
      (let ((pk (aref dist i)))
        (declare (type (double-float 0d0) pk))
        (setf sum (+ sum
                     (if (= pk 0d0)
                         0d0
                         (* pk (log pk)))))))
    (* -1d0 sum)))

(defun node-entropy (node)
  (entropy (node-sample-indices node)
           (length (node-sample-indices node))
           (node-dtree node)))

(defun min/max (lst)
  (let ((min (car lst))
        (max (car lst)))
    (dolist (elem (cdr lst))
      (cond ((< max elem) (setf max elem))
            ((> min elem) (setf min elem))))
    (values min max)))

(defun region-min/max (sample-indices datamatrix attribute)
  (let ((min (aref datamatrix (aref sample-indices 0) attribute))
        (max (aref datamatrix (aref sample-indices 0) attribute)))
    (declare (optimize (speed 3) (safety 0))
             (type double-float min max)
             (type (simple-array fixnum) sample-indices)
             (type (simple-array double-float) datamatrix)
             (type fixnum attribute))
    (loop for index fixnum across sample-indices do
      (let ((elem (aref datamatrix index attribute)))
        (declare (type double-float elem))
        (cond ((< max elem) (setf max elem))
              ((> min elem) (setf min elem)))))
    (values min max)))

(defun random-uniform (start end)
  (+ (random (- end start)) start))

(defun make-random-test (node)
  (let* ((dtree (node-dtree node))
         (datamatrix (dtree-datamatrix dtree))
         (attribute (random (the fixnum (dtree-datum-dim dtree))))
         (sample-indices (node-sample-indices node)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array fixnum) sample-indices)
             (type (simple-array double-float) datamatrix)
             (type fixnum attribute))
    (multiple-value-bind (min max)
        (region-min/max sample-indices datamatrix attribute)
      (declare (type double-float min max))
      (let ((threshold (if (= min max) min (random-uniform min max))))
        (declare (type double-float threshold))
        (values attribute threshold)))))

;; ;; Pick 2 points from sample-indices, then random sampling between them.
;; ;; faster, but less accuracy
;; (defun make-random-test (node)
;;   (let* ((dtree (node-dtree node))
;;          (datamatrix (dtree-datamatrix dtree))
;;          (attribute (random (the fixnum (dtree-datum-dim dtree))))
;;          (sample-indices (node-sample-indices node)))
;;     (declare (optimize (speed 3) (safety 0))
;;              (type (simple-array fixnum) sample-indices)
;;              (type (simple-array double-float) datamatrix)
;;              (type fixnum attribute))
;;     (let ((v1 (aref datamatrix (random (the fixnum (length sample-indices))) attribute))
;;           (v2 (aref datamatrix (random (the fixnum (length sample-indices))) attribute)))
;;       (declare (type double-float v1 v2))
;;       (let ((threshold (cond ((= v1 v2) v1)
;;                              ((<= v1 v2) (random-uniform v1 v2))
;;                              (t (random-uniform v2 v1)))))
;;         (declare (type double-float threshold))
;;         (values attribute threshold)))))

(defun split-sample-indices (sample-indices true-array false-array attribute threshold datamatrix)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) sample-indices true-array false-array)
           (type fixnum attribute)
           (type double-float threshold)
           (type (simple-array double-float) datamatrix))
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
    (loop for i fixnum from 0 to (1- tmp-index1) do
      (setf (aref best-arr1 i) (aref tmp-arr1 i)))
    (loop for i fixnum from 0 to (1- tmp-index2) do
      (setf (aref best-arr2 i) (aref tmp-arr2 i)))
    (setf (dtree-best-index1 dtree) tmp-index1
          (dtree-best-index2 dtree) tmp-index2)))

(defun make-partial-arr (arr len)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array fixnum) arr)
           (type fixnum len))
  (let ((new-arr (make-array len :element-type 'fixnum)))
    (loop for i fixnum from 0 to (1- len) do
      (setf (aref new-arr i) (aref arr i)))
    new-arr))

(defun set-best-children! (n-trial node &optional (gain-test #'entropy))
  ;;(declare (optimize (speed 3) (safety 0)))
  (let ((dtree (node-dtree node))
        (max-children-gain most-negative-double-float)
        (left-node (make-node nil node))
        (right-node (make-node nil node)))
    (setf (node-left-node node) left-node
          (node-right-node node) right-node)
    (loop repeat n-trial do
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
                   (+ (* -1d0 (/ left-len parent-size) left-gain)
                      (* -1d0 (/ right-len parent-size) right-gain))))
            (when (< max-children-gain children-gain)
              (copy-tmp->best! dtree)
              (setf max-children-gain children-gain
                    (node-test-attribute node) attribute
                    (node-test-threshold node) threshold
                    (node-information-gain left-node) left-gain
                    (node-information-gain right-node) right-gain))))))
    (setf (node-sample-indices left-node)
          (make-partial-arr (dtree-best-arr1 dtree) (dtree-best-index1 dtree))
          (node-sample-indices right-node)
          (make-partial-arr (dtree-best-arr2 dtree) (dtree-best-index2 dtree)))
    node))

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

(defun find-leaf (node datamatrix datum-index)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum datum-index)
           (type (simple-array double-float) datamatrix))
  (cond ((null node) nil)
        ((null (node-test-attribute node)) node)
        (t (let ((attribute (node-test-attribute node))
                 (threshold (node-test-threshold node)))
             (declare (type fixnum attribute)
                      (type double-float threshold))
             (if (>= (aref datamatrix datum-index attribute) threshold)
                 (find-leaf (node-left-node node) datamatrix datum-index)
                 (find-leaf (node-right-node node) datamatrix datum-index))))))

(defun predict-dtree (dtree datamatrix datum-index)
  (let ((max 0d0)
        (max-class 0)
        (dist (node-class-distribution (find-leaf (dtree-root dtree) datamatrix datum-index)))
        (n-class (dtree-n-class dtree)))
    (declare (optimize (speed 3) (safety 0))
             (type double-float max)
             (type fixnum max-class n-class)
             (type (simple-array double-float) dist))
    (loop for i fixnum from 0 to (1- n-class) do
      (when (> (aref dist i) max)
        (setf max (aref dist i)
              max-class i)))
    max-class))

(defun test-dtree (dtree datamatrix target)
  (let ((counter 0)
        (len (array-dimension datamatrix 0)))
    (loop for i fixnum from 0 to (1- len) do
      (when (= (predict-dtree dtree datamatrix i)
               (aref target i))
        (incf counter)))
    (* 100d0 (/ counter len))))

;;; forest

(defstruct (forest (:constructor %make-forest)
                   (:print-object %print-forest))
  n-tree bagging-ratio datamatrix target dtree-list
  n-class class-count-array datum-dim max-depth min-region-samples n-trial gain-test)

(defun %print-forest (obj stream)
  (format stream "#S(FOREST :N-TREE ~A)"
          (forest-n-tree obj)))

(defun bootstrap-sample-indices (n datamatrix)
  (let ((len (array-dimension datamatrix 0))
        (arr (make-array n :element-type 'fixnum :initial-element 0)))
    (loop for i from 0 to (1- n) do
      (setf (aref arr i) (random len)))
    arr))

(defun make-forest (n-class datum-dim datamatrix target
                    &key (n-tree 100) (bagging-ratio 0.1) (max-depth 5) (min-region-samples 1)
                      (n-trial 10) (gain-test #'entropy))
  (let ((forest (%make-forest
                 :n-tree n-tree
                 :bagging-ratio bagging-ratio
                 :datamatrix datamatrix
                 :target target
                 :n-class n-class
                 :class-count-array (make-array n-class :element-type 'double-float)
                 :datum-dim datum-dim
                 :max-depth max-depth
                 :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test)))
    (setf (forest-dtree-list forest)
          (loop repeat n-tree collect
            (make-dtree n-class datum-dim datamatrix target
                 :max-depth max-depth
                 :min-region-samples min-region-samples
                 :n-trial n-trial
                 :gain-test gain-test
                 :sample-indices (bootstrap-sample-indices
                                  (floor (* (array-dimension datamatrix 0) bagging-ratio))
                                  datamatrix))))
    forest))

(defun class-distribution-forest (forest datamatrix datum-index)
  (let ((n-class (forest-n-class forest))
        (n-tree (forest-n-tree forest))
        (class-count-array (forest-class-count-array forest)))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array double-float) datamatrix class-count-array)
             (type fixnum datum-index n-class n-tree))
    ;; init forest-class-count-array
    (loop for i fixnum from 0 to (1- n-class) do
      (setf (aref class-count-array i) 0d0))
    ;; whole count
    (dolist (dtree (forest-dtree-list forest))
      (let ((dist (node-class-distribution
                   (find-leaf (dtree-root dtree) datamatrix datum-index))))
        (declare (type (simple-array double-float) dist))
        (loop for i fixnum from 0 to (1- n-class) do
          (incf (aref class-count-array i) (aref dist i)))))
    ;;
    (loop for i fixnum from 0 to (1- n-class) do
      (setf (aref class-count-array i)
            (/ (aref class-count-array i) n-tree)))
    class-count-array))

(defun predict-forest (forest datamatrix datum-index)
  (let ((max 0d0)
        (max-class 0)
        (dist (class-distribution-forest forest datamatrix datum-index))
        (n-class (forest-n-class forest)))
    (declare (optimize (speed 3) (safety 0))
             (type double-float max)
             (type fixnum max-class n-class)
             (type (simple-array double-float) dist))
    (loop for i from 0 to (1- n-class) do
      (when (> (aref dist i) max)
        (setf max (aref dist i)
              max-class i)))
    max-class))

(defun test-forest (forest datamatrix target)
  (let ((counter 0)
        (len (array-dimension datamatrix 0)))
    (loop for i fixnum from 0 to (1- len) do
      (when (= (predict-forest forest datamatrix i)
               (aref target i))
        (incf counter)))
    (* 100d0 (/ counter len))))
