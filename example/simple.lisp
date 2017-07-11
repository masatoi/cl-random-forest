;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; Small dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter *n-class* 4)
(defparameter *n-dim* 2)

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
  
;; make decision tree
(defparameter *dtree* (make-dtree *n-class* *n-dim* *datamatrix* *target*))

;; test decision tree
(test-dtree *dtree* *datamatrix* *target*)

;; display tree information
(traverse #'node-information-gain (dtree-root *dtree*))
(traverse #'node-sample-indices (dtree-root *dtree*))

;; make random forest
(defparameter *forest*
  (make-forest *n-class* *n-dim* *datamatrix* *target* :n-tree 10 :bagging-ratio 1.0))

;; test random forest
(test-forest *forest* *datamatrix* *target*)

;; make refine learner
(defparameter *forest-learner* (make-refine-learner *forest*))
(defparameter *forest-refine-dataset* (make-refine-dataset *forest* *datamatrix*))

(train-refine-learner *forest-learner* *forest-refine-dataset* *target*)
(test-refine-learner  *forest-learner* *forest-refine-dataset* *target*)

;; Enable parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
(train-refine-learner *forest-learner* *forest-refine-dataset* *target*)
(test-refine-learner  *forest-learner* *forest-refine-dataset* *target*)

;; Global pruning

;; Pruning *forest*
(pruning! *forest* *forest-learner* 0.1)

;; Re-learning of refine-learner
(setf *forest-refine-dataset* (make-refine-dataset *forest* *datamatrix*))
(setf *forest-learner* (make-refine-learner *forest*))
(train-refine-learner *forest-learner* *forest-refine-dataset* *target*)
(test-refine-learner  *forest-learner* *forest-refine-dataset* *target*)

;; cross-validation
(defparameter *n-fold* 3)
(cross-validation-forest-with-refine-learner *n-fold* *n-class* *n-dim* *datamatrix* *target*)
