;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; Small dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter *n-class* 4)
(defparameter *n-dim* 2)

(defparameter *dataset*
  '((0 . (-1d0 -2d0))
    (0 . (-2d0 -1d0))
    (1 . (1d0 -2d0))
    (1 . (3d0 -1.5d0))
    (2 . (-2d0 2d0))
    (2 . (-3d0 1d0))
    (2 . (-2d0 1d0))
    (3 . (3d0 2d0))
    (3 . (2d0 2d0))
    (3 . (1d0 2d0))
    (3 . (1d0 1d0))))

(defparameter *target*
  (make-array (length *dataset*) :element-type 'fixnum :initial-contents (mapcar #'car *dataset*)))

(defparameter *datamatrix*
  (let ((arr (make-array (list (length *dataset*) *n-dim*) :element-type 'double-float)))
    ;; set datamatrix
    (loop for i from 0 below (length *dataset*)
          for elem in *dataset* do
            (loop for j from 0 to 1 do
              (setf (aref arr i j)
                    (nth j (cdr elem)))))
    arr))
  
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
