;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; cpusmall
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cpusmall

;; Shuffle and split
;; $ shuf -o cpusmall_scale.shuf cpusmall_scale
;; $ split -4915 cpusmall_scale.shuf cpusmall_scale.shuf
;; $ mv cpusmall_scale.shufaa cpusmall.train
;; $ mv cpusmall_scale.shufab cpusmall.test

(multiple-value-bind (datamat targetmat)
    (clrf.utils:clol-dataset->datamatrix/target-regression
     (clol.utils:read-data "/home/wiz/datasets/regression/cpusmall.train" 12))
  (defparameter datamatrix datamat)
  (defparameter target targetmat))

(multiple-value-bind (datamat targetmat)
    (clrf.utils:clol-dataset->datamatrix/target-regression
     (clol.utils:read-data "/home/wiz/datasets/regression/cpusmall.test" 12))
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
                          :n-tree 100 :max-depth 15 :bagging-ratio 0.6 :n-trial 10))

(test-regression-forest rforest test-datamatrix test-target)
;; RMSE: 2.921160545860163d0
