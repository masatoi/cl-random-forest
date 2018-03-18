;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; covtype
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#covtype

;; Shuffle and split
;; $ shuf -o covtype.scale.shuf covtype.scale
;; $ split -348607 covtype.scale.shuf covtype.scale.shuf

(setf lparallel:*kernel* (lparallel:make-kernel 4))
(defparameter covtype-dim 54)
(defparameter covtype-n-class 7)

(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/covtype.scale.train" covtype-dim)
  (defparameter covtype-datamatrix datamat)
  (defparameter covtype-target target))

(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/covtype.scale.test" covtype-dim)
  (defparameter covtype-datamatrix-test datamat)
  (defparameter covtype-target-test target))

;; dtree
(defparameter covtype-dtree
  (make-dtree covtype-n-class covtype-datamatrix covtype-target :max-depth 10))
(test-dtree covtype-dtree covtype-datamatrix-test covtype-target-test)

;; random-forest
(defparameter covtype-forest
  (make-forest covtype-n-class covtype-datamatrix covtype-target
               :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 20 :max-depth 15))
(test-forest covtype-forest covtype-datamatrix-test covtype-target-test)
;; max-depth=5,n-tree=100: 2.861 seconds
;; max-depth=10,n-tree=500: 14.975 seconds
;; max-depth=15,n-tree=500,n-trial 20: 31.005 seconds

(time
 (defparameter covtype-forest-tall
   (make-forest covtype-n-class covtype-datamatrix covtype-target
                :n-tree 100 :bagging-ratio 1.0 :min-region-samples 5 :n-trial 10 :max-depth 25)))

(time
 (defparameter covtype-refine-dataset
   (make-refine-dataset covtype-forest covtype-datamatrix)))
;; max-depth=10,n-tree=500: 4.923 seconds
;; max-depth=15,n-tree=500: 7.669 seconds

(time
 (defparameter covtype-refine-test
   (make-refine-dataset covtype-forest covtype-datamatrix-test)))
;; max-depth=10,n-tree=500: 2.797 seconds
;; max-depth=15,n-tree=500: 5.422 seconds

(defparameter covtype-refine-learner (make-refine-learner covtype-forest))

(time
 (train-refine-learner-process covtype-refine-learner
                               covtype-refine-dataset covtype-target
                               covtype-refine-test    covtype-target-test))

(test-refine-learner covtype-refine-learner covtype-refine-test covtype-target-test)

(time
 (loop repeat 10 do
   (train-refine-learner covtype-refine-learner covtype-refine-dataset covtype-target)
   (test-refine-learner  covtype-refine-learner covtype-refine-test covtype-target-test)))
;; max-depth=15,n-tree=500: 75.847 seconds
;; max-depth=10: 92.87623% / max-depth=15: 96.00137%

;; Pruning
(loop repeat 10 do
  (setf covtype-refine-dataset nil
        covtype-refine-test nil)
  (sb-ext:gc :full t)
  (room)
  (format t "~%Making covtype-refine-dataset~%")
  (setf covtype-refine-dataset (make-refine-dataset covtype-forest covtype-datamatrix))
  (format t "Making covtype-refine-test~%")
  (setf covtype-refine-test (make-refine-dataset covtype-forest covtype-datamatrix-test))
  (format t "Re-learning~%")
  (setf covtype-refine-learner (make-refine-learner covtype-forest))
  (loop repeat 10 do
    (train-refine-learner covtype-refine-learner covtype-refine-dataset covtype-target)
    (test-refine-learner  covtype-refine-learner covtype-refine-test covtype-target-test))
  (format t "Pruning. leaf-size: ~A" (length (collect-leaf-parent covtype-forest)))
  (pruning! covtype-forest covtype-refine-learner 0.5)
  (format t " -> ~A ~%" (length (collect-leaf-parent covtype-forest))))
