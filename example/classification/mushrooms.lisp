;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; mushrooms ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms

(defparameter mushrooms-dim 112)
(defparameter mushrooms-n-class 2)

;; Read training data
(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/mushrooms-train" mushrooms-dim)
  (defparameter mushrooms-datamatrix datamat)
  (defparameter mushrooms-target target))

;; Change from binary data to 2-class data
(loop for i from 0 below (length mushrooms-target) do
  (if (> (aref mushrooms-target i) 0)
      (setf (aref mushrooms-target i) 0)
      (setf (aref mushrooms-target i) 1)))

;; Read test data
(multiple-value-bind (datamat target)
    (read-data "/home/wiz/datasets/mushrooms-test" mushrooms-dim)
  (defparameter mushrooms-datamatrix-test datamat)
  (defparameter mushrooms-target-test target))

(loop for i from 0 below (length mushrooms-target-test) do
  (if (> (aref mushrooms-target-test i) 0)
      (setf (aref mushrooms-target-test i) 0)
      (setf (aref mushrooms-target-test i) 1)))

;; Make decision tree

(defparameter mushrooms-dtree
  (make-dtree mushrooms-n-class mushrooms-datamatrix mushrooms-target
              :max-depth 15 :min-region-samples 5 :n-trial 10))

(test-dtree mushrooms-dtree mushrooms-datamatrix mushrooms-target)
(test-dtree mushrooms-dtree mushrooms-datamatrix-test mushrooms-target-test)
;; => 98.21092278719398d0

;; Make forest
(defparameter mushrooms-forest
  (make-forest mushrooms-n-class mushrooms-datamatrix mushrooms-target
               :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10))

(test-forest mushrooms-forest mushrooms-datamatrix mushrooms-target) ; => 99.98
(test-forest mushrooms-forest mushrooms-datamatrix-test mushrooms-target-test) ; => 96.7

(defparameter mushrooms-forest-tall
  (make-forest mushrooms-n-class mushrooms-datamatrix mushrooms-target
               :n-tree 100 :bagging-ratio 1.0 :min-region-samples 1 :n-trial 10 :max-depth 15))

(test-forest mushrooms-forest-tall mushrooms-datamatrix mushrooms-target) ; => 100.0
(test-forest mushrooms-forest-tall mushrooms-datamatrix-test mushrooms-target-test) ; => 93.03201

;; Global refinement

(defparameter mushrooms-refine-dataset
  (make-refine-dataset mushrooms-forest mushrooms-datamatrix))

(defparameter mushrooms-refine-test
  (make-refine-dataset mushrooms-forest mushrooms-datamatrix-test))

(defparameter mushrooms-refine-learner (make-refine-learner mushrooms-forest))

(train-refine-learner-process mushrooms-refine-learner
                              mushrooms-refine-dataset mushrooms-target
                              mushrooms-refine-test mushrooms-target-test)

;; Test & Prediction

(test-refine-learner mushrooms-refine-learner mushrooms-refine-dataset mushrooms-target) ; => 100.0
(test-refine-learner mushrooms-refine-learner mushrooms-refine-test mushrooms-target-test) ; => 94.3

(predict-refine-learner mushrooms-forest mushrooms-refine-learner mushrooms-datamatrix-test 0)

;; Global Pruning

(loop repeat 10 do
  (setf mushrooms-refine-dataset nil
        mushrooms-refine-test nil)
  (sb-ext:gc :full t)
  (format t "~%Making mushrooms-refine-dataset~%")
  (setf mushrooms-refine-dataset (make-refine-dataset mushrooms-forest mushrooms-datamatrix))
  (format t "Making mushrooms-refine-test~%")
  (setf mushrooms-refine-test (make-refine-dataset mushrooms-forest mushrooms-datamatrix-test))
  (format t "Re-learning~%")
  (setf mushrooms-refine-learner (make-refine-learner mushrooms-forest))
  (train-refine-learner-process mushrooms-refine-learner
                                mushrooms-refine-dataset mushrooms-target
                                mushrooms-refine-test mushrooms-target-test)
  (test-refine-learner mushrooms-refine-learner mushrooms-refine-test mushrooms-target-test)
  (format t "Pruning. leaf-size: ~A" (length (collect-leaf-parent mushrooms-forest)))
  (pruning! mushrooms-forest mushrooms-refine-learner 0.5)
  (format t " -> ~A ~%" (length (collect-leaf-parent mushrooms-forest))))
