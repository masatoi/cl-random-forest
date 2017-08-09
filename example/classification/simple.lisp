;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

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

;; test decision tree
(test-dtree *dtree* *datamatrix* *target*)

;; display tree information
(traverse #'node-information-gain (dtree-root *dtree*))
(traverse #'node-sample-indices (dtree-root *dtree*))

;;; Random forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; make random forest
(defparameter *forest*
  (make-forest *n-class* *datamatrix* *target*
               :n-tree 10 :bagging-ratio 1.0
               :max-depth 5 :min-region-samples 1 :n-trial 10))

;; predict
(predict-forest *forest* *datamatrix* 0)

;; test random forest
(test-forest *forest* *datamatrix* *target*)

;;; Global refinement ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; make refine learner
(defparameter *forest-learner* (make-refine-learner *forest*))
(defparameter *forest-refine-dataset* (make-refine-dataset *forest* *datamatrix*))

(train-refine-learner *forest-learner* *forest-refine-dataset* *target*)
(test-refine-learner  *forest-learner* *forest-refine-dataset* *target*)

;; Enable parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
(train-refine-learner *forest-learner* *forest-refine-dataset* *target*)
(test-refine-learner  *forest-learner* *forest-refine-dataset* *target*)

;;; Global pruning ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Pruning *forest*
(pruning! *forest* *forest-learner* 0.1)

;; Re-learning of refine-learner
(setf *forest-refine-dataset* (make-refine-dataset *forest* *datamatrix*))
(setf *forest-learner* (make-refine-learner *forest*))
(train-refine-learner *forest-learner* *forest-refine-dataset* *target*)
(test-refine-learner  *forest-learner* *forest-refine-dataset* *target*)

;; cross-validation
(defparameter *n-fold* 3)
(cross-validation-forest-with-refine-learner *n-fold* *n-class* *datamatrix* *target*)
