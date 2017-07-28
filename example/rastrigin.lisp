;; -*- coding:utf-8; mode:lisp -*-

;; $ ros install masatoi/clgplot
(ql:quickload :clgplot)

(in-package :clrf)

(defun rastrigin (x-list)
  (let ((n (length x-list)))
    (+ (* 10 n)
       (loop for xi in x-list summing
	 (- (* xi xi) (* 10 (cos (* 2 pi xi))))))))

(clgp:splot-list (lambda (x y)
                   (rastrigin (list x y)))
                 (clgp:seq -5.12 5.12 0.08)
                 (clgp:seq -5.12 5.12 0.08)
                 :map t)

(defparameter *datamatrix*
  (let ((arr (make-array '(10000 2) :element-type 'double-float :initial-element 0d0)))
    (loop for i from 0 below (array-dimension arr 0) do
      (loop for j from 0 below (array-dimension arr 1) do
        (setf (aref arr i j)
              (- (random 10.24d0) 5.12d0))))
    arr))

(defparameter *target*
  (let ((arr (make-array '(10000 1) :element-type 'double-float :initial-element 0d0)))
    (loop for i from 0 below (array-dimension arr 0) do
      (setf (aref arr i 0) (rastrigin (list (aref *datamatrix* i 0)
                                            (aref *datamatrix* i 1)))))
    arr))

(defparameter *test-datamatrix*
  (let* ((n (* 64 64)) ; separate to 64x64 cells (by 0.16)
         (arr (make-array (list n 2) :element-type 'double-float :initial-element 0d0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16d0) 5.04d0))
              (y (- (* j 0.16d0) 5.04d0)))
          (setf (aref arr (+ (* i 64) j) 0) x
                (aref arr (+ (* i 64) j) 1) y))))
    arr))

(defparameter *test-target*
  (let* ((n (* 64 64)) ; separate to 64x64 cells (by 0.16)
         (arr (make-array (list n 1) :element-type 'double-float :initial-element 0d0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16d0) 5.04d0))
              (y (- (* j 0.16d0) 5.04d0)))
          (setf (aref arr (+ (* i 64) j) 0)
                (rastrigin (list x y))))))
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

(defparameter *predict-matrix*
  ;; separate to 64x64 cells (by 0.16)
  (let ((arr (make-array (list 64 64) :element-type 'double-float :initial-element 0d0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16d0) 5.04d0))
              (y (- (* j 0.16d0) 5.04d0)))
          (setf (aref arr i j)
                (aref
                 (predict-rtree
                  *rtree*
                  (make-array '(1 2) :element-type 'double-float
                                     :initial-contents (list (list x y))) 0) 0)))))
    arr))

(defparameter *predict-matrix-forest*
  ;; separate to 64x64 cells (by 0.16)
  (let ((arr (make-array (list 64 64) :element-type 'double-float :initial-element 0d0)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16d0) 5.04d0))
              (y (- (* j 0.16d0) 5.04d0)))
          (setf (aref arr i j)
                (aref
                 (predict-regression-forest
                  *rforest*
                  (make-array '(1 2) :element-type 'double-float
                                     :initial-contents (list (list x y))) 0) 0)))))
    arr))

(clgp:splot-matrix *predict-matrix*)
(clgp:splot-matrix *predict-matrix-forest*)
