;; -*- coding:utf-8; mode:lisp -*-

(defpackage :cl-random-forest/example/regression/cpusmall
  (:use #:cl
        #:cl-random-forest)
  (:import-from #:cl-random-forest/src/utils
                #:read-data-regression))

(in-package :cl-random-forest/example/regression/cpusmall)

;;; cpusmall
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cpusmall

;; Shuffle and split
;; $ shuf -o cpusmall_scale.shuf cpusmall_scale
;; $ split -4915 cpusmall_scale.shuf cpusmall_scale.shuf
;; $ mv cpusmall_scale.shufaa cpusmall.train
;; $ mv cpusmall_scale.shufab cpusmall.test

(defparameter dim 12)

(multiple-value-bind (datamat targetmat)
    (read-data-regression "/home/wiz/datasets/regression/cpusmall.train" dim)
  (defparameter datamatrix datamat)
  (defparameter target targetmat))

(multiple-value-bind (datamat targetmat)
    (read-data-regression "/home/wiz/datasets/regression/cpusmall.test" dim)
  (defparameter test-datamatrix datamat)
  (defparameter test-target targetmat))

(defparameter rtree (make-rtree datamatrix target :max-depth 10))
(test-rtree rtree test-datamatrix test-target)
;; RMSE: 3.846092375763221d0

;; (require :sb-sprof)
;; (sb-sprof:with-profiling (:max-samples 1000
;;                           :report :flat
;;                           :show-progress t)
;;   (defparameter rforest
;;     (make-regression-forest datamatrix target
;;                             :n-tree 100 :max-depth 15 :bagging-ratio 0.6 :n-trial 10)))

;; 0.794 seconds
(defparameter rforest
  (make-regression-forest datamatrix target
                          :n-tree 100 :max-depth 15 :min-region-samples 5
                          :bagging-ratio 1.0 :n-trial 4))

(test-regression-forest rforest datamatrix target)
(test-regression-forest rforest test-datamatrix test-target)
;; RMSE: 2.921160545860163d0

;;; Global refinement

(defparameter rforest-500
  (make-regression-forest datamatrix target
                          :n-tree 500 :max-depth 5 :bagging-ratio 0.1 :n-trial 10))

(test-regression-forest rforest-500 test-datamatrix test-target)

(defparameter refine-learner (make-regression-refine-learner rforest-500 1.0))
(defparameter refine-dataset (make-regression-refine-dataset rforest-500 datamatrix))
(defparameter refine-testset (make-regression-refine-dataset rforest-500 test-datamatrix))

(loop repeat 100 do
  (train-regression-refine-learner refine-learner refine-dataset target)
  (test-regression-refine-learner refine-learner refine-testset test-target))
