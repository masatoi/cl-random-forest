;; -*- coding:utf-8; mode:lisp -*-

(defpackage :cl-random-forest/example/regression/abalone
  (:use #:cl
        #:cl-random-forest)
  (:import-from #:cl-random-forest/src/utils
                #:read-data-regression))

(in-package :cl-random-forest/example/regression/abalone)

;;; abalone
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#abalone

;; Shuffle and split
;; $ shuf -o abalone_scale.shuf abalone_scale
;; $ split -2506 abalone_scale.shuf abalone_scale.shuf
;; $ mv abalone_scale.shufaa abalone.train
;; $ mv abalone_scale.shufab abalone.test

(defparameter dim 8)

(multiple-value-bind (datamat targetmat)
    (read-data-regression "/home/wiz/datasets/regression/abalone.train" dim)
  (defparameter datamatrix datamat)
  (defparameter target targetmat))

(multiple-value-bind (datamat targetmat)
    (read-data-regression "/home/wiz/datasets/regression/abalone.test" dim)
  (defparameter test-datamatrix datamat)
  (defparameter test-target targetmat))

(defparameter rtree (make-rtree datamatrix target :max-depth 10))
(test-rtree rtree test-datamatrix test-target)
;; RMSE: 2.6315442182534965d0

(defparameter rforest
  (make-regression-forest datamatrix target
                          :n-tree 100 :max-depth 15 :bagging-ratio 0.6 :n-trial 10))

(test-regression-forest rforest test-datamatrix test-target)
;; RMSE: 2.208641788206514d0

;;; Global refinement

(defparameter rforest-500-tree
  (make-regression-forest datamatrix target
                          :n-tree 500 :max-depth 5 :bagging-ratio 0.1 :n-trial 10))

(test-regression-forest rforest-500-tree test-datamatrix test-target)
;; RMSE: 2.3859577

(defparameter refine-learner (make-regression-refine-learner rforest-500-tree 0.9999))

(defparameter refine-dataset (make-regression-refine-dataset rforest-500-tree datamatrix))
(defparameter refine-testset (make-regression-refine-dataset rforest-500-tree test-datamatrix))

(loop repeat 100 do
  (train-regression-refine-learner refine-learner refine-dataset target)
  (test-regression-refine-learner refine-learner refine-testset test-target))
