;; -*- coding:utf-8; mode:lisp -*-

(ql:quickload :clgplot)

(in-package :clrf)

;; Sampling from Gaussian distribution (Box–Muller's method)
(defun random-normal (&key (mean 0d0) (sd 1d0))
  (let ((alpha (random 1.0d0))
	(beta  (random 1.0d0)))
    (+ (* sd
	  (sqrt (* -2 (log alpha)))
	  (sin (* 2 pi beta)))
       mean)))

;;; sine curve  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter *n* 100)

(defparameter *datamatrix*
  (let ((arr (make-array (list *n* 1) :element-type 'double-float)))
    (loop for i from 0 below *n* do
      (setf (aref arr i 0) (random-uniform (- pi) pi)))
    arr))

(defparameter *target*
  (let ((arr (make-array *n* :element-type 'double-float)))
    (loop for i from 0 below *n* do
      (setf (aref arr i) (+ (sin (aref *datamatrix* i 0))
                            (random-normal :sd 0.1d0))))
    arr))

(defparameter *test*
  (let ((arr (make-array (list *n* 1) :element-type 'double-float)))
    (loop for i from 0 below *n*
          for x from (- pi) to pi by (/ (* 2 pi) *n*)
          do (setf (aref arr i 0) x))
    arr))

(defparameter *test-target*
  (let ((arr (make-array *n* :element-type 'double-float)))
    (loop for i from 0 below *n* do
      (setf (aref arr i) (sin (aref *test* i 0))))
    arr))

;; Plot training-data
(defun slice (arr)
  (loop for i from 0 below (array-dimension arr 0) collect (aref arr i 0)))

(clgp:plots (list *test-target* *target*)
            :x-seqs (list (slice *test*)
                          (slice *datamatrix*))
            :style '(lines points))

;; make decision tree
(defparameter *rtree*
  (make-rtree *datamatrix* *target* :max-depth 5 :min-region-samples 2 :n-trial 10))

;; test by training data
(test-rtree *rtree* *datamatrix* *target*)
;; test by testing data
(test-rtree *rtree* *test* *test-target*)

(predict-rtree *rtree* *test* 0)

;; display tree information
(traverse #'node-information-gain (dtree-root *rtree*))
(traverse #'node-sample-indices (dtree-root *rtree*))

(do-leaf (lambda (node)
           (print (node-sample-indices node)))
  (dtree-root *rtree*))

;; make random forest
(defparameter *rforest*
  (make-regression-forest *datamatrix* *target*
                          :n-tree 100 :bagging-ratio 0.6
                          :max-depth 5 :min-region-samples 2 :n-trial 10))

(let ((x-sample-lst (slice *datamatrix*))
      (x-lst (slice *test*)))
  (clgp:plots
   (list *target*
         *test-target*
         (loop for i from 0 below *n* collect (predict-rtree *rtree* *test* i))
         (loop for i from 0 below *n* collect (predict-regression-forest *rforest* *test* i)))
   :x-seqs (list x-sample-lst x-lst x-lst x-lst)
   :style '(points lines lines lines)
   :title-list '("training-data" "true" "predict(dtree)" "predict(forest)")
   ;; :output "/home/wiz/Dropbox/tmp/regression-forest.png"
   :x-range '(-3.3 3.3)
   :y-range '(-1.5 2.0)))

;; test by training data
(test-regression-forest *rforest* *datamatrix* *target*)
;; test by testing data
(test-regression-forest *rforest* *test* *test-target*)
