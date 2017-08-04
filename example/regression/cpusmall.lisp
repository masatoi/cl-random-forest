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
  (defparameter cpusmall-datamatrix datamat)
  (defparameter cpusmall-target targetmat))

(multiple-value-bind (datamat targetmat)
    (clrf.utils:clol-dataset->datamatrix/target-regression
     (clol.utils:read-data "/home/wiz/datasets/regression/cpusmall.test" 12))
  (defparameter cpusmall-datamatrix-test datamat)
  (defparameter cpusmall-target-test targetmat))

(defparameter cpusmall-rtree (make-rtree cpusmall-datamatrix cpusmall-target :max-depth 10))
(test-rtree cpusmall-rtree cpusmall-datamatrix-test cpusmall-target-test)
;; RMSE: 3.846092375763221d0

;; (require :sb-sprof)
;; (sb-sprof:with-profiling (:max-samples 1000
;;                           :report :flat
;;                           :show-progress t)
;;   (defparameter rforest
;;     (make-regression-forest datamatrix target
;;                             :n-tree 100 :max-depth 15 :bagging-ratio 0.6 :n-trial 10)))

;; 0.794 seconds
(defparameter cpusmall-rforest
  (make-regression-forest cpusmall-datamatrix cpusmall-target
                          :n-tree 100 :max-depth 15 :bagging-ratio 0.6 :n-trial 10))

(test-regression-forest cpusmall-rforest cpusmall-datamatrix-test cpusmall-target-test)
;; RMSE: 2.921160545860163d0
