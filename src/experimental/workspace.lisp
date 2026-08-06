;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-random-forest)

;;; Small dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter *n-class* 4)

(defparameter *target*
  (make-array 11 :element-type 'fixnum
                 :initial-contents '(0 0 1 1 2 2 2 3 3 3 3)))

(defparameter *datamatrix*
  (make-array '(11 2) 
              :element-type 'double-float
              :initial-contents '((-1.0d0 -2.0d0)
                                  (-2.0d0 -1.0d0)
                                  (1.0d0 -2.0d0)
                                  (3.0d0 -1.5d0)
                                  (-2.0d0 2.0d0)
                                  (-3.0d0 1.0d0)
                                  (-2.0d0 1.0d0)
                                  (3.0d0 2.0d0)
                                  (2.0d0 2.0d0)
                                  (1.0d0 2.0d0)
                                  (1.0d0 1.0d0))))

;;; Decision tree ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
;; make decision tree
(defparameter *dtree*
  (make-dtree *n-class* *datamatrix* *target*
              :max-depth 5 :min-region-samples 1 :n-trial 10))

;; prediction
(predict-dtree *dtree* *datamatrix* 0)
