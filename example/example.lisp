;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; Small dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter *n-class* 4)
(defparameter *n-dim* 2)

(defparameter *dataset*
  '((0 . (-1d0 -2d0))
    (0 . (-2d0 -1d0))
    (1 . (1d0 -2d0))
    (1 . (3d0 -1.5d0))
    (2 . (-2d0 2d0))
    (2 . (-3d0 1d0))
    (2 . (-2d0 1d0))
    (3 . (3d0 2d0))
    (3 . (2d0 2d0))
    (3 . (1d0 2d0))
    (3 . (1d0 1d0))))

;; (ql:quickload :clgplot)
;; (clgp:plots '((-2d0 -1d0)
;;               (-2d0 -1.5d0)
;;               (2d0 1d0 1d0)
;;               (2d0 2d0 2d0 1d0))
;;             :x-seqs '((-1d0 -2d0)
;;                       (1d0 3d0)
;;                       (-2d0 -3d0 -2d0)
;;                       (3d0 2d0 1d0 1d0))
;;             :style 'points
;;             :x-range '(-3.5 3.5)
;;             :y-range '(-3.5 3.5))

(defparameter *target*
  (make-array (length *dataset*) :element-type 'fixnum :initial-contents (mapcar #'car *dataset*)))

(defparameter *datamatrix* (make-array (list (length *dataset*) *n-dim*) :element-type 'double-float))
;; set datamatrix
(loop for i from 0 below (length *dataset*)
      for elem in *dataset*
      do
   (loop for j from 0 to 1 do
     (setf (aref *datamatrix* i j)
           (nth j (cdr elem)))))

;; make decision tree
(defparameter *dtree* (make-dtree *n-class* *n-dim* *datamatrix* *target*))

;; test decision tree
(test-dtree *dtree* *datamatrix* *target*)

;; display tree information
(traverse #'node-information-gain (dtree-root *dtree*))
(traverse #'node-sample-indices (dtree-root *dtree*))

(set-leaf-index! *dtree*)
(do-leaf (lambda (node)
           (format t "~%leaf-index: ~A, sample-indices: ~A"
                   (node-leaf-index node)
                   (node-sample-indices node)))
  (dtree-root *dtree*))
(traverse #'node-leaf-index (dtree-root *dtree*))

;; make random forest
(defparameter *forest*
  (make-forest *n-class* *n-dim* *datamatrix* *target* :n-tree 3 :bagging-ratio 1.0))

;; test random forest
(test-forest *forest* *datamatrix* *target*)

;; make refine learner
(defparameter *forest-learner* (make-refine-learner *forest*))

(defparameter *forest-refine-dataset* (make-refine-dataset *forest* *datamatrix* *target*))
(clol:train *forest-learner* *forest-refine-dataset*)
(clol:test *forest-learner* *forest-refine-dataset*)

(defparameter *forest-leaf-indices-vector*
  (make-leaf-indices-vector *forest* *datamatrix*))

(train-refine-learner-fast *forest-learner* *forest-leaf-indices-vector* *target*)
(test-refine-learner-fast *forest-learner* *forest-leaf-indices-vector* *target*)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; parallelizaion of test

(defparameter mini-batch-size 4)
(defparameter n-class 4)
(defparameter n-tree 3)

(defparameter sv-vec (make-array n-class))
(let ((sv-index (make-array n-tree :element-type 'fixnum :initial-element 0))
      (sv-val (make-array n-tree :element-type 'double-float :initial-element 1d0)))
  (loop for i fixnum from 0 to (1- n-class) do
    (setf (svref sv-vec i)
          (clol.vector:make-sparse-vector sv-index sv-val))))

(defparameter activation-matrix
  (make-array (list mini-batch-size n-class) :element-type 'double-float :initial-element 0d0))

(set-activation-matrix activation-matrix *forest-learner* n-class sv-vec *forest-leaf-indices-vector* 0 mini-batch-size)

(set-activation-matrix activation-matrix *forest-learner* n-class sv-vec *forest-leaf-indices-vector* 2 3)

(set-activation-matrix activation-matrix *forest-learner* n-class sv-vec *forest-leaf-indices-vector* 0 11)

(floor 11 40)
2
3


(defparameter n-correct 0)
(maximize-activation/count activation-matrix mini-batch-size *target* 0)


(test-refine-learner-fast *forest-learner* *forest-leaf-indices-vector* *target*)
(test-refine-learner-parallel *forest-learner* *forest-leaf-indices-vector* *target* :mini-batch-size 4)

;; Enable parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
(train-refine-learner-parallel *forest-learner* *forest-leaf-indices-vector* *target*)
(test-refine-learner-fast *forest-learner* *forest-leaf-indices-vector* *target*)

;; Global pruning
(pruning! *forest* *forest-learner* 0.1)
(setf *forest-leaf-indices-vector* (make-leaf-indices-vector *forest* *datamatrix*))
(setf *forest-learner* (make-refine-learner *forest*))
(train-refine-learner-parallel *forest-learner* *forest-leaf-indices-vector* *target*)
(test-refine-learner-fast *forest-learner* *forest-leaf-indices-vector* *target*)

;; 枝刈りの前後で森の中の決定木の深さを比較する
(defun collect-leaf-depth (dtree)
  (let ((lst nil))
    (do-leaf (lambda (node)
               (push (node-depth node) lst))
      (dtree-root dtree))
    (nreverse lst)))

(mapcar #'collect-leaf-depth (forest-dtree-list *forest*))

(traverse #'node-information-gain (dtree-root (forest-dtree-list *forest*))
(traverse #'node-sample-indices (dtree-root *dtree*))

;; 初期状態
;; '((2 2 2 2) (2 2 2 2) (2 2 2 2) (2 2 2 2) (1 2 2) (1 2 2) (2 2 1) (2 3 3 1)
;;   (2 2 2 2) (2 2 2 2))

;; 枝刈り実行
(pruning! *forest* *forest-learner* 0.1)
(setf *forest-leaf-indices-vector* (make-leaf-indices-vector *forest* *datamatrix*))
(setf *forest-learner* (make-refine-learner *forest*))
(train-refine-learner-parallel *forest-learner* *forest-leaf-indices-vector* *target*)
(test-refine-learner-fast *forest-learner* *forest-leaf-indices-vector* *target*)
(mapcar #'collect-leaf-depth (forest-dtree-list *forest*))

;; '((2 2 2 2) (2 2 2 2) (2 2 2 2) (2 2 2 2) (1 2 2) (1 1) (2 2 1) (2 3 3 1)
;;   (2 2 2 2) (2 2 2 2))

;; '((2 2 2 2) (2 2 2 2) (2 2 2 2) (2 2 2 2) (1 2 2) (1 1) (2 2 1) (2 2 1)
;;   (2 2 2 2) (2 2 2 2))

;; '((2 2 2 2) (2 2 2 2) (2 2 2 2) (2 2 2 2) (1 2 2) (1 1) (2 2 1) (2 2 1)
;;   (2 2 2 2) (2 2 1))

;; '((2 2 2 2) (2 2 2 2) (2 2 2 2) (2 2 1) (1 2 2) (1 1) (2 2 1) (2 2 1) (2 2 2 2)
;;   (2 2 1))

;;; MNIST ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(defparameter mnist-dim 780)

(let ((mnist-train (clol.utils:read-data "/home/wiz/tmp/mnist.scale" mnist-dim :multiclass-p t))
      (mnist-test (clol.utils:read-data "/home/wiz/tmp/mnist.scale.t" mnist-dim :multiclass-p t)))

  ;; Add 1 to labels in order to form class-labels begin from 0
  (dolist (datum mnist-train) (incf (car datum)))
  (dolist (datum mnist-test)  (incf (car datum)))

  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target mnist-train)
    (defparameter mnist-datamatrix datamat)
    (defparameter mnist-target target))
  
  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target mnist-test)
    (defparameter mnist-datamatrix-test datamat)
    (defparameter mnist-target-test target)))
  
(time (defparameter mnist-dtree (make-dtree 10 780 mnist-datamatrix mnist-target :max-depth 10 :n-trial 27)))

;; (write-to-r-format-from-clol-dataset mnist-train "/home/wiz/datasets/mnist-for-R")

;; Evaluation took:
;;   2.329 seconds of real time
;;   2.332534 seconds of total run time (2.332534 user, 0.000000 system)
;;   [ Run times consist of 0.027 seconds GC time, and 2.306 seconds non-GC time. ]
;;   100.17% CPU
;;   7,902,772,046 processor cycles
;;   833,405,296 bytes consed

;; Evaluation took:
;;   1.913 seconds of real time
;;   1.913793 seconds of total run time (1.889773 user, 0.024020 system)
;;   [ Run times consist of 0.022 seconds GC time, and 1.892 seconds non-GC time. ]
;;   100.05% CPU
;;   6,488,816,835 processor cycles
;;   834,031,648 bytes consed

;; split-list => split-arr
;; Evaluation took:
;;   2.162 seconds of real time
;;   2.170621 seconds of total run time (2.170621 user, 0.000000 system)
;;   100.42% CPU
;;   7,333,558,776 processor cycles
;;   568,714,880 bytes consed

;; with datamatrix and target
;; Evaluation took:
;;   1.427 seconds of real time
;;   1.428196 seconds of total run time (1.421912 user, 0.006284 system)
;;   [ Run times consist of 0.011 seconds GC time, and 1.418 seconds non-GC time. ]
;;   100.07% CPU
;;   4,841,932,737 processor cycles
;;   566,070,512 bytes consed

;; split-arr => split-sample-indices
;; Evaluation took:
;;   1.157 seconds of real time
;;   1.159157 seconds of total run time (1.159157 user, 0.000000 system)
;;   [ Run times consist of 0.008 seconds GC time, and 1.152 seconds non-GC time. ]
;;   100.17% CPU
;;   3,925,551,873 processor cycles
;;   533,164,544 bytes consed

;; make-random-test exploit fixed array
;; Evaluation took:
;;   0.587 seconds of real time
;;   0.586155 seconds of total run time (0.578620 user, 0.007535 system)
;;   99.83% CPU
;;   1,988,323,468 processor cycles
;;   16,330,528 bytes consed

;; Pick 2-points in make-random-test (less accuracy)
;; Evaluation took:
;;   0.486 seconds of real time
;;   0.486908 seconds of total run time (0.486908 user, 0.000000 system)
;;   100.21% CPU
;;   1,648,894,569 processor cycles
;;   14,178,160 bytes consed

;; safety 0
;; Evaluation took:
;;   0.273 seconds of real time
;;   0.273069 seconds of total run time (0.269585 user, 0.003484 system)
;;   [ Run times consist of 0.010 seconds GC time, and 0.264 seconds non-GC time. ]
;;   100.00% CPU
;;   925,629,119 processor cycles
;;   16,496,000 bytes consed

;; Pick 2-points + safety 0
;; Evaluation took:
;;   0.215 seconds of real time
;;   0.215951 seconds of total run time (0.215951 user, 0.000000 system)
;;   100.47% CPU
;;   731,896,506 processor cycles
;;   13,927,600 bytes consed

(require 'sb-sprof)
(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (defparameter mnist-dtree (make-dtree 10 780 mnist-datamatrix mnist-target :max-depth 5 :n-trial 270)))

(time (test-dtree mnist-dtree mnist-datamatrix mnist-target))
(time (test-dtree mnist-dtree mnist-datamatrix-test mnist-target-test))
;; => 75.89d0

(time
 (loop repeat 10 do
   (print (test-dtree (make-dtree 10 780 mnist-datamatrix mnist-target :max-depth 10 :n-trial 27)
                      mnist-datamatrix mnist-target))))

;; numbers of datapoints each node has
(traverse (lambda (node) (length (node-sample-indices node)))
          (dtree-root mnist-dtree))

;; prediction
(ql:quickload :clgplot)
(clgp:plot (node-class-distribution
            (find-leaf (dtree-root mnist-dtree) mnist-datamatrix-test 0))
           :style 'impulse)
(predict-dtree mnist-dtree mnist-datamatrix-test 0)

;;; forest

;; 2.19 sec
(time (defparameter mnist-forest
        (make-forest 10 780 mnist-datamatrix mnist-target
                     :n-tree 500 :bagging-ratio 0.1
                     :max-depth 10 :n-trial 10 :min-region-samples 5)))

(time (print (test-forest mnist-forest mnist-datamatrix mnist-target)))
(time (print (test-forest mnist-forest mnist-datamatrix-test mnist-target-test)))

;; 18.952 seconds
(time (defparameter mnist-refine-dataset
        (make-refine-dataset mnist-forest mnist-datamatrix mnist-target)))

;; 3.286 seconds
(time (defparameter mnist-refine-test
        (make-refine-dataset mnist-forest mnist-datamatrix-test mnist-target-test)))

(defparameter mnist-refine-learner (make-refine-learner mnist-forest 1.0d0))
(defparameter mnist-refine-learner (make-refine-learner-scw mnist-forest 0.999d0 0.01d0))

;; 23.396 seconds
(time
 (loop repeat 10 do
   (clol:train mnist-refine-learner mnist-refine-dataset)
   (clol:test  mnist-refine-learner mnist-refine-test)))

(clol:test  mnist-refine-learner mnist-refine-dataset)
(predict-refine-learner mnist-forest mnist-refine-learner mnist-datamatrix-test 0)

;; Evaluation took:
;;   22.525 seconds of real time
;;   22.539637 seconds of total run time (22.458771 user, 0.080866 system)
;;   [ Run times consist of 0.557 seconds GC time, and 21.983 seconds non-GC time. ]
;;   100.07% CPU
;;   76,407,726,956 processor cycles
;;   11,103,020,544 bytes consed

;; Evaluation took:
;;   3.439 seconds of real time
;;   3.438304 seconds of total run time (3.376220 user, 0.062084 system)
;;   [ Run times consist of 0.074 seconds GC time, and 3.365 seconds non-GC time. ]
;;   99.97% CPU
;;   11,667,948,885 processor cycles
;;   731,683,488 bytes consed

;;; Parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
;; (setf lparallel:*kernel* nil)

(time (defparameter mnist-forest
        (make-forest 10 780 mnist-datamatrix mnist-target
                     :n-tree 100 :bagging-ratio 0.1
                     :max-depth 10 :n-trial 27)))

;; Evaluation took:
;;   1.095 seconds of real time
;;   4.126016 seconds of total run time (4.036967 user, 0.089049 system)
;;   [ Run times consist of 0.064 seconds GC time, and 4.063 seconds non-GC time. ]
;;   376.80% CPU
;;   3,714,638,160 processor cycles
;;   720,102,160 bytes consed

;; prediction
(clgp:plot (class-distribution-forest mnist-forest mnist-datamatrix-test 0) :style 'impulse)
(predict-forest mnist-forest mnist-datamatrix-test 0)
(time (print (test-forest mnist-forest mnist-datamatrix mnist-target)))
(time (print (test-forest mnist-forest mnist-datamatrix-test mnist-target-test)))

(time (defparameter mnist-forest
        (make-forest 10 780 mnist-datamatrix mnist-target
                     :n-tree 500 :bagging-ratio 1.0
                     :max-depth 10 :n-trial 27)))

;; train
;; 53.616 seconds

;; predict
;; train 99.885d0   58.639 seconds
;; test  96.65d0    9.791 seconds

(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (time (print (test-forest mnist-forest mnist-datamatrix mnist-target))))

;; => 93.43d0
;; Evaluation took:
;;   1.912 seconds of real time
;;   1.914804 seconds of total run time (1.914804 user, 0.000000 system)
;;   [ Run times consist of 0.017 seconds GC time, and 1.898 seconds non-GC time. ]
;;   100.16% CPU
;;   6,486,260,337 processor cycles
;;   803,371,520 bytes consed

;; Evaluation took:
;;   0.744 seconds of real time
;;   0.744223 seconds of total run time (0.744223 user, 0.000000 system)
;;   100.00% CPU
;;   2,524,311,970 processor cycles
;;   72,800 bytes consed

(mapcar (lambda (dtree)
          (test-dtree dtree mnist-datamatrix-test mnist-target-test))
        (forest-dtree-list mnist-forest))

;; (66.99000000000001d0 70.59d0 69.96d0 71.12d0 70.08d0 69.89d0 69.51d0 66.9d0
;;  69.57d0 68.49d0 70.16d0 69.1d0 69.6d0 67.7d0 70.23d0 70.98d0 66.88d0 69.16d0
;;  71.21d0 68.38d0 69.06d0 69.89999999999999d0 69.87d0 69.82000000000001d0
;;  69.84d0 70.17999999999999d0 70.34d0 66.95d0 70.83d0 68.22d0 69.23d0 68.16d0
;;  68.63d0 71.04d0 68.19d0 67.86d0 68.04d0 68.56d0 69.21000000000001d0 67.97d0
;;  69.04d0 69.39d0 67.85d0 69.57d0 69.78999999999999d0 70.17999999999999d0
;;  67.58d0 68.89d0 67.89d0 71.57d0 69.35d0 68.02d0 68.41000000000001d0 68.2d0
;;  71.14d0 70.88d0 70.42d0 70.73d0 68.73d0 67.57d0 70.58d0 69.67999999999999d0
;;  70.71d0 70.57d0 70.98d0 69.12d0 69.25d0 67.95d0 70.38d0 70.58d0 70.44d0
;;  67.92d0 69.53d0 70.59d0 68.02d0 68.83d0 70.08d0 69.28999999999999d0 71.81d0
;;  70.8d0 69.67999999999999d0 69.35d0 71.95d0 68.19d0 69.42d0 68.60000000000001d0
;;  68.96d0 67.73d0 70.75d0 70.14d0 68.16d0 69.58d0 70.41d0 70.03d0
;;  68.08999999999999d0 66.79d0 72.23d0 68.05d0 68.05d0 72.14d0)

;;;;;;;;;;;;; mushrooms

;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms

(defparameter *mushrooms-dim* 112)
(defparameter *mushrooms-train* (clol.utils:read-data "/home/wiz/datasets/mushrooms-train" *mushrooms-dim*))
(defparameter *mushrooms-test* (clol.utils:read-data "/home/wiz/datasets/mushrooms-test" *mushrooms-dim*))

(dolist (datum *mushrooms-train*)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(dolist (datum *mushrooms-test*)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target *mushrooms-train*)
  (defparameter mushrooms-datamatrix datamat)
  (defparameter mushrooms-target target))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target *mushrooms-test*)
  (defparameter mushrooms-datamatrix-test datamat)
  (defparameter mushrooms-target-test target))

(write-to-r-format-from-clol-dataset *mushrooms-train* "/home/wiz/datasets/mushrooms-for-R")
(write-to-r-format-from-clol-dataset *mushrooms-test* "/home/wiz/datasets/mushrooms-for-R.t")

;; decision tree

(defparameter mushrooms-dtree
  (make-dtree 2 *mushrooms-dim* mushrooms-datamatrix mushrooms-target :max-depth 20))
(traverse #'node-information-gain (dtree-root mushrooms-dtree))
(test-dtree mushrooms-dtree mushrooms-datamatrix-test mushrooms-target-test)
;; => 98.21092278719398d0

(time
 (defparameter mushrooms-forest
   (make-forest 2 *mushrooms-dim* mushrooms-datamatrix mushrooms-target
                :n-tree 500 :bagging-ratio 1.0 :min-region-samples 1 :n-trial 10 :max-depth 10)))

(time
 (defparameter mushrooms-forest
   (make-forest 2 *mushrooms-dim* mushrooms-datamatrix mushrooms-target
                :n-tree 100 :bagging-ratio 1.0 :min-region-samples 1 :n-trial 10 :max-depth 10)))

;; Evaluation took:
;;   1.901 seconds of real time
;;   1.902504 seconds of total run time (1.866545 user, 0.035959 system)
;;   [ Run times consist of 0.084 seconds GC time, and 1.819 seconds non-GC time. ]
;;   100.11% CPU
;;   6,448,467,269 processor cycles
;;   1,171,116,816 bytes consed

;; Evaluation took:
;;   0.447 seconds of real time
;;   0.447074 seconds of total run time (0.378886 user, 0.068188 system)
;;   [ Run times consist of 0.157 seconds GC time, and 0.291 seconds non-GC time. ]
;;   100.00% CPU
;;   1,515,547,477 processor cycles
;;   137,457,680 bytes consed

;; Evaluation took:
;;   0.018 seconds of real time
;;   0.059598 seconds of total run time (0.059598 user, 0.000000 system)
;;   333.33% CPU
;;   60,858,259 processor cycles
;;   28,202,128 bytes consed

(test-forest mushrooms-forest mushrooms-datamatrix mushrooms-target)
(test-forest mushrooms-forest mushrooms-datamatrix-test mushrooms-target-test)

(time
 (defparameter mushrooms-refine-dataset
   (make-refine-dataset mushrooms-forest mushrooms-datamatrix mushrooms-target)))

(time
 (defparameter mushrooms-refine-test
   (make-refine-dataset mushrooms-forest mushrooms-datamatrix-test mushrooms-target-test)))

(defparameter mushrooms-refine-learner (make-refine-learner mushrooms-forest 1.0d0))

(time
 (loop repeat 10 do
   (clol:train mushrooms-refine-learner mushrooms-refine-dataset)
   (clol:test  mushrooms-refine-learner mushrooms-refine-test)))

;; pruning
(length (collect-leaf-parent mushrooms-forest)) ; => 1691
(pruning! mushrooms-forest mushrooms-refine-learner 0.1)
(length (collect-leaf-parent mushrooms-forest)) ; => 1614

;; Re-learning refine learner
(defparameter mushrooms-refine-dataset
  (make-refine-dataset mushrooms-forest mushrooms-datamatrix mushrooms-target))
(defparameter mushrooms-refine-test
  (make-refine-dataset mushrooms-forest mushrooms-datamatrix-test mushrooms-target-test))
(defparameter mushrooms-refine-learner (make-refine-learner mushrooms-forest))
(loop repeat 5 do
  (clol:train mushrooms-refine-learner mushrooms-refine-dataset)
  (clol:test  mushrooms-refine-learner mushrooms-refine-test))

;;;;;;;;;;;; covtype.binary

(defparameter covtype-dim 54)
(defparameter covtype-train (clol.utils:read-data "/home/wiz/datasets/covtype.libsvm.binary.scale" covtype-dim))

(dolist (datum covtype-train)
  (if (> (car datum) 1d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target covtype-train)
  (defparameter covtype-datamatrix datamat)
  (defparameter covtype-target target))

;; (setf lparallel:*kernel* (lparallel:make-kernel 4))

(time
 (defparameter covtype-forest
   (make-forest 2 covtype-dim covtype-datamatrix covtype-target
                :n-tree 500 :bagging-ratio 0.1 :n-trial 10 :max-depth 5)))

(time (print (test-forest covtype-forest covtype-datamatrix covtype-target)))
(time (print (predict-forest covtype-forest covtype-datamatrix 0)))

;; heap exhaust
(defparameter covtype-refine-dataset (make-refine-dataset covtype-forest covtype-datamatrix covtype-target))
(defparameter covtype-refine-test (make-refine-dataset covtype-forest covtype-datamatrix-test covtype-target-test))
(defparameter covtype-refine-learner (make-refine-learner covtype-forest 1.0d0))
(loop repeat 10 do
  (clol:train covtype-refine-learner covtype-refine-dataset)
  (clol:test  covtype-refine-learner covtype-refine-test))

;;;;;;;;;;;;;;;;;;;;

;;; a1a

(defparameter a1a-dim 123)
(defparameter a1a-train (clol.utils:read-data "/home/wiz/datasets/a1a" a1a-dim))
(defparameter a1a-test (clol.utils:read-data "/home/wiz/datasets/a1a.t" a1a-dim))

(dolist (datum a1a-train)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(dolist (datum a1a-test)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target a1a-train)
  (defparameter a1a-datamatrix datamat)
  (defparameter a1a-target target))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target a1a-test)
  (defparameter a1a-datamatrix-test datamat)
  (defparameter a1a-target-test target))

;; dtree
(defparameter a1a-dtree (make-dtree 2 a1a-dim a1a-datamatrix a1a-target :max-depth 20))
(test-dtree a1a-dtree a1a-datamatrix-test a1a-target-test)

;; random-forest
(defparameter a1a-forest
  (make-forest 2 a1a-dim a1a-datamatrix a1a-target
               :n-tree 500 :bagging-ratio 1.0 :min-region-samples 5 :n-trial 10 :max-depth 5))
(test-forest a1a-forest a1a-datamatrix a1a-target)
(test-forest a1a-forest a1a-datamatrix-test a1a-target-test)

(defparameter a1a-refine-dataset (make-refine-dataset a1a-forest a1a-datamatrix a1a-target))
(defparameter a1a-refine-test (make-refine-dataset a1a-forest a1a-datamatrix-test a1a-target-test))
(defparameter a1a-refine-learner (make-refine-learner a1a-forest 1.0d0))
(loop repeat 10 do
  (clol:train a1a-refine-learner a1a-refine-dataset)
  (clol:test  a1a-refine-learner a1a-refine-test))

;;; a9a

(defparameter a9a-dim 123)
(defparameter a9a-train (clol.utils:read-data "/home/wiz/datasets/a9a" a9a-dim))
(defparameter a9a-test (clol.utils:read-data "/home/wiz/datasets/a9a.t" a9a-dim))

(dolist (datum a9a-train)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(dolist (datum a9a-test)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target a9a-train)
  (defparameter a9a-datamatrix datamat)
  (defparameter a9a-target target))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target a9a-test)
  (defparameter a9a-datamatrix-test datamat)
  (defparameter a9a-target-test target))

;; dtree
(defparameter a9a-dtree (make-dtree 2 a9a-dim a9a-datamatrix a9a-target :max-depth 20))
(test-dtree a9a-dtree a9a-datamatrix-test a9a-target-test)

;; random-forest
(defparameter a9a-forest
  (make-forest 2 a9a-dim a9a-datamatrix a9a-target
               :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 5))
(test-forest a9a-forest a9a-datamatrix a9a-target)
(test-forest a9a-forest a9a-datamatrix-test a9a-target-test)

(defparameter a9a-refine-dataset (make-refine-dataset a9a-forest a9a-datamatrix a9a-target))
(defparameter a9a-refine-test (make-refine-dataset a9a-forest a9a-datamatrix-test a9a-target-test))
(defparameter a9a-refine-learner (make-refine-learner a9a-forest 1.0d0))
(loop repeat 10 do
  (clol:train a9a-refine-learner a9a-refine-dataset)
  (clol:test  a9a-refine-learner a9a-refine-test))

;;; 

;;;;; letter

(defparameter letter-dim 16)
(defparameter letter-n-class 26)
(defparameter letter-train (clol.utils:read-data "/home/wiz/datasets/letter.scale" letter-dim
                                                 :multiclass-p t))
(defparameter letter-test (clol.utils:read-data "/home/wiz/datasets/letter.scale.t" letter-dim
                                                :multiclass-p t))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target letter-train)
  (defparameter letter-datamatrix datamat)
  (defparameter letter-target target))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix/target letter-test)
  (defparameter letter-datamatrix-test datamat)
  (defparameter letter-target-test target))

;; dtree
(defparameter letter-dtree
  (make-dtree letter-n-class letter-dim letter-datamatrix letter-target :max-depth 10))
(test-dtree letter-dtree letter-datamatrix-test letter-target-test)

;; random-forest
(defparameter letter-forest
  (make-forest letter-n-class letter-dim letter-datamatrix letter-target
               :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10))
(test-forest letter-forest letter-datamatrix-test letter-target-test)
;; max-depth=5: 73.22% / max-depth=10: 89.03% / max-depth=15: 91.4%

(defparameter letter-forest-tall
  (make-forest letter-n-class letter-dim letter-datamatrix letter-target
               :n-tree 100 :bagging-ratio 1.0 :min-region-samples 5 :n-trial 10 :max-depth 100))
(test-forest letter-forest-tall letter-datamatrix-test letter-target-test) ; 96.82%

(defparameter letter-refine-dataset
  (make-refine-dataset letter-forest letter-datamatrix letter-target))

(defparameter letter-refine-test
  (make-refine-dataset letter-forest letter-datamatrix-test letter-target-test))

(defparameter letter-refine-learner (make-refine-learner letter-forest 1.0d0))

(loop repeat 10 do
  (clol:train letter-refine-learner letter-refine-dataset)
  (clol:test  letter-refine-learner letter-refine-test))
;; max-depth=5: 95.880005% / max-depth=10: 97.259995% / max-depth=15: 97.34%

;;;;; covtype

(defparameter covtype-dim 54)
(defparameter covtype-n-class 7)

(let ((covtype-train (clol.utils:read-data "/home/wiz/datasets/covtype.scale" covtype-dim :multiclass-p t))
      (covtype-test (clol.utils:read-data "/home/wiz/datasets/covtype.scale.t" covtype-dim :multiclass-p t)))
  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target covtype-train)
    (defparameter covtype-datamatrix datamat)
    (defparameter covtype-target target))

  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target covtype-test)
    (defparameter covtype-datamatrix-test datamat)
    (defparameter covtype-target-test target)))

;; dtree
(defparameter covtype-dtree
  (make-dtree covtype-n-class covtype-dim covtype-datamatrix covtype-target :max-depth 10))
(test-dtree covtype-dtree covtype-datamatrix-test covtype-target-test)

;; random-forest
(time
 (defparameter covtype-forest
   (make-forest covtype-n-class covtype-dim covtype-datamatrix covtype-target
                :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 20 :max-depth 15)))
(test-forest covtype-forest covtype-datamatrix-test covtype-target-test)
;; max-depth=5,n-tree=100: 2.861 seconds

(defparameter covtype-forest-tall
  (make-forest covtype-n-class covtype-dim covtype-datamatrix covtype-target
               :n-tree 100 :bagging-ratio 1.0 :min-region-samples 5 :n-trial 10 :max-depth 100))
(test-forest covtype-forest-tall covtype-datamatrix-test covtype-target-test) ; 96.82%

(defparameter covtype-refine-dataset
  (make-refine-dataset covtype-forest covtype-datamatrix covtype-target))
;; 1.787 seconds

(defparameter covtype-refine-test
  (make-refine-dataset covtype-forest covtype-datamatrix-test covtype-target-test))

(defparameter covtype-refine-learner (make-refine-learner covtype-forest 1.0d0))

(loop repeat 10 do
  (clol:train covtype-refine-learner covtype-refine-dataset)
  (clol:test  covtype-refine-learner covtype-refine-test))
;; max-depth=5: 95.880005% / max-depth=10: 97.259995% / max-depth=15: 97.34%

;; In case of without making dataset
(loop repeat 5 do
  (time (train-refine-learner covtype-forest covtype-refine-learner covtype-datamatrix covtype-target))
  (time (test-refine-learner covtype-forest covtype-refine-learner covtype-datamatrix-test covtype-target-test)))
;; 72.57374%

;;  4.495 seconds of real time
(time
 (defparameter covtype-leaf-index-matrix
   (make-leaf-index-matrix covtype-forest covtype-datamatrix)))

;; 4.221 seconds
(time
 (defparameter covtype-leaf-indices-vector
   (make-leaf-indices-vector covtype-forest covtype-datamatrix)))

(time
 (defparameter covtype-leaf-indices-vector-test
   (make-leaf-indices-vector covtype-forest covtype-datamatrix-test)))

(loop repeat 10 do
  (train-refine-learner-fast covtype-refine-learner covtype-leaf-indices-vector covtype-target)
  (format t "train: ")
  (test-refine-learner-fast covtype-refine-learner covtype-leaf-indices-vector covtype-target)
  (format t "test: ")
  (test-refine-learner-fast covtype-refine-learner covtype-leaf-indices-vector-test covtype-target-test))


;; usps
;; Char74k
;; covtype
