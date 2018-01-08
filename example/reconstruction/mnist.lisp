;; -*- coding:utf-8; mode:lisp -*-

;; set dynamic-space-size >= 2500

(in-package :cl-random-forest)

;;; Load Dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; MNIST data
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(defparameter mnist-dim 784)
(defparameter mnist-n-class 10)

(let ((mnist-train (clol.utils:read-data "/home/wiz/datasets/mnist.scale" mnist-dim :multiclass-p t)))
  ;; Add 1 to labels in order to form class-labels beginning from 0
  (dolist (datum mnist-train) (incf (car datum)))
  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target mnist-train)
    (defparameter mnist-datamatrix datamat)
    (defparameter mnist-target target)))


;; Enable parallelization
(setf lparallel:*kernel* (lparallel:make-kernel 4))

;; Note that SAVE-PARENT-NODE? keyword option is true
(defparameter mnist-forest-tall
  (make-forest mnist-n-class mnist-datamatrix mnist-target
               :n-tree 500 :bagging-ratio 0.1 :max-depth 15 :n-trial 1 :min-region-samples 5
               :save-parent-node? t))

;; Execute reconstruction
(defparameter *reconstruction*
  (reconstruction-forest mnist-forest-tall mnist-datamatrix 0))

;; Plot
(ql:quickload :clgplot)

(defparameter *original-image*
  (let ((arr (make-array '(28 28))))
    (loop for i from 0 below 28 do
      (loop for j from 0 below 28 do
        (setf (aref arr i j)
              (aref mnist-datamatrix 0 (+ (* i 28) j)))))
    arr))

(defparameter *reconstruction-image*
  (let ((arr (make-array '(28 28))))
    (loop for i from 0 below 28 do
      (loop for j from 0 below 28 do
        (setf (aref arr i j)
              (aref *reconstruction* (+ (* i 28) j)))))
    arr))

(clgp:splot-matrix *original-image*)
(clgp:splot-matrix *reconstruction-image*)
