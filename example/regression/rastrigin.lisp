;; -*- coding:utf-8; mode:lisp -*-

(ql:quickload :clgplot)

(defpackage :cl-random-forest/example/regression/rastrigin
  (:use #:cl
        #:cl-random-forest))

(in-package :cl-random-forest/example/regression/rastrigin)

;; Rastrigin function
;; https://en.wikipedia.org/wiki/Rastrigin_function

(defun rastrigin (x-list)
  (let ((n (length x-list))
        (pi-float (coerce pi 'single-float)))
    (+ (* 10 n)
       (loop for xi in x-list
             summing (- (* xi xi) (* 10 (cos (* 2 pi-float xi))))))))

(clgp:splot-list (lambda (x y)
                   (rastrigin (list x y)))
                 (clgp:seq -5.12 5.12 0.16)
                 (clgp:seq -5.12 5.12 0.16)
                 :map t)

(defparameter *datamatrix*
  (let ((arr (make-array '(10000 2) :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 below (array-dimension arr 0) do
      (loop for j from 0 below (array-dimension arr 1) do
        (setf (aref arr i j)
              (- (random 10.24) 5.12))))
    arr))

(defparameter *target*
  (let ((arr (make-array 10000 :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 below (array-dimension arr 0) do
      (setf (aref arr i) (rastrigin (list (aref *datamatrix* i 0)
                                          (aref *datamatrix* i 1)))))
    arr))

(defparameter *test-datamatrix*
  (let* ((n (* 64 64)) ; separate to 64x64 cells (by 0.16)
         (arr (make-array (list n 2) :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16) 5.04))
              (y (- (* j 0.16) 5.04)))
          (setf (aref arr (+ (* i 64) j) 0) x
                (aref arr (+ (* i 64) j) 1) y))))
    arr))

(defparameter *test-target*
  (let* ((n (* 64 64)) ; separate to 64x64 cells (by 0.16)
         (arr (make-array n :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16) 5.04))
              (y (- (* j 0.16) 5.04)))
          (setf (aref arr (+ (* i 64) j)) (rastrigin (list x y))))))
    arr))

;; make decision tree
(defparameter *rtree* (make-rtree *datamatrix* *target* :max-depth 15 :min-region-samples 1 :n-trial 25))

;; test by training data
(test-rtree *rtree* *datamatrix* *target*)
;; test by testing data
(test-rtree *rtree* *test-datamatrix* *test-target*)

(predict-rtree *rtree* *test-datamatrix* 0)

;; make random forest
;; 3.343 seconds (1 core), 1.227 seconds (4 core)
(defparameter *rforest*
  (make-regression-forest *datamatrix* *target* :n-tree 100 :bagging-ratio 0.6 :max-depth 15 :min-region-samples 1  :n-trial 25))

;; test random forest
(test-regression-forest *rforest* *datamatrix* *target*)
(test-regression-forest *rforest* *test-datamatrix* *test-target*)

;; plot prediction

(defparameter *predict-matrix-rtree*
  ;; separate to 64x64 cells (by 0.16)
  (let ((arr (make-array (list 64 64) :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16) 5.04))
              (y (- (* j 0.16) 5.04)))
          (setf (aref arr i j)
                (predict-rtree
                 *rtree*
                 (make-array '(1 2) :element-type 'single-float
                                    :initial-contents (list (list x y))) 0)))))
    arr))

(clgp:splot-matrix *predict-matrix-rtree*)

(defparameter *predict-matrix-forest*
  ;; separate to 64x64 cells (by 0.16)
  (let ((arr (make-array (list 64 64) :element-type 'single-float :initial-element 0.0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16) 5.04))
              (y (- (* j 0.16) 5.04)))
          (setf (aref arr i j)
                (predict-regression-forest
                 *rforest*
                 (make-array '(1 2) :element-type 'single-float
                                    :initial-contents (list (list x y))) 0)))))
    arr))

(clgp:splot-matrix *predict-matrix-forest*)
