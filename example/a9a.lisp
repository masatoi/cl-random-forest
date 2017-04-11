;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; a9a ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter a9a-dim 123)
(defparameter a9a-n-class 2)

(let ((a9a-train (clol.utils:read-data "/home/wiz/datasets/a9a" a9a-dim))
      (a9a-test (clol.utils:read-data "/home/wiz/datasets/a9a.t" a9a-dim)))

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
    (defparameter a9a-target-test target)))

;; dtree
(defparameter a9a-dtree (make-dtree a9a-n-class a9a-dim a9a-datamatrix a9a-target :max-depth 20))
(test-dtree a9a-dtree a9a-datamatrix-test a9a-target-test)

;; random-forest
(defparameter a9a-forest
  (make-forest a9a-n-class a9a-dim a9a-datamatrix a9a-target
               :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10))
(test-forest a9a-forest a9a-datamatrix a9a-target)
(test-forest a9a-forest a9a-datamatrix-test a9a-target-test)

(defparameter a9a-refine-dataset (make-refine-dataset a9a-forest a9a-datamatrix))
(defparameter a9a-refine-test (make-refine-dataset a9a-forest a9a-datamatrix-test))
(defparameter a9a-refine-learner (make-refine-learner a9a-forest))

(train-refine-learner-process a9a-refine-learner a9a-refine-dataset a9a-target
                              a9a-refine-test a9a-target-test)

(test-refine-learner a9a-refine-learner a9a-refine-test a9a-target-test)
