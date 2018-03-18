;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-random-forest)

(defun reconstruction-backward (node input-range-array)
  (let ((parent-node (node-parent-node node)))
    (if (null parent-node)
        input-range-array
        (let ((attribute (node-test-attribute parent-node))
              (threshold (node-test-threshold parent-node)))
          (if (eq (node-left-node parent-node) node)
              (when (>= threshold (aref input-range-array 0 attribute)) ; left-node
                (setf (aref input-range-array 0 attribute) threshold))
              (when (< threshold (aref input-range-array 1 attribute)) ; right-node
                (setf (aref input-range-array 1 attribute) threshold)))
          (reconstruction-backward parent-node input-range-array)))))

(defun reconstruction-dtree (leaf-node input-range-array)
  (if (null (node-parent-node leaf-node))
      (print leaf-node)
      (reconstruction-backward leaf-node input-range-array)))

(defun reconstruction-forest (forest datamatrix datum-index)
  (let* ((dim (forest-datum-dim forest))
         (input-range-array (make-array (list 2 dim) :element-type 'double-float))
         (result (make-array dim :element-type 'double-float)))

    ;; initialize input-range-array
    (loop for i from 0 below dim do
      (setf (aref input-range-array 0 i) most-negative-double-float
            (aref input-range-array 1 i) most-positive-double-float))

    ;; set input-range-array for each dtree
    (dolist (dtree (forest-dtree-list forest))
      (reconstruction-dtree
       (find-leaf (dtree-root dtree) datamatrix datum-index)
       input-range-array))

    ;; When only either upper-bound or lower-bound is bounded, set zero.
    (loop for i from 0 below dim do
      (when (= (aref input-range-array 0 i) most-negative-double-float)
        (setf (aref input-range-array 0 i) 0d0))
      (when (= (aref input-range-array 1 i) most-positive-double-float)
        (setf (aref input-range-array 1 i) 0d0)))

    (loop for i from 0 below dim do
      (setf (aref result i) (/ (+ (aref input-range-array 0 i)
                                  (aref input-range-array 1 i))
                               2d0)))
    result))

(defun encode-datum (forest datamatrix datum-index)
  (clol::sparse-vector-index-vector
   (make-refine-vector forest datamatrix datum-index)))

(defun make-leaf-node-vector (forest)
  (let* ((len (+ (aref (forest-index-offset forest) (1- (forest-n-tree forest)))
                 (dtree-max-leaf-index (car (last (forest-dtree-list forest))))))
         (leaf-node-vector (make-array len))
         (i 0))
    (dolist (dtree (forest-dtree-list forest))
      (do-leaf (lambda (node)
                 (setf (aref leaf-node-vector i) node)
                 (incf i))
        (dtree-root dtree)))
    leaf-node-vector))

(defun decode-datum (forest leaf-index-vector &optional leaf-node-vector)
  (let* ((dim (forest-datum-dim forest))
         (leaf-node-vector (if leaf-node-vector
                               leaf-node-vector
                               (make-leaf-node-vector forest)))
         (input-range-array (make-array (list 2 dim) :element-type 'double-float))
         (result (make-array dim :element-type 'double-float)))

    ;; initialize input-range-array
    (loop for i from 0 below dim do
      (setf (aref input-range-array 0 i) most-negative-double-float
            (aref input-range-array 1 i) most-positive-double-float))

    ;; set input-range-array for each dtree
    (loop for leaf-index across leaf-index-vector do
      (reconstruction-dtree (aref leaf-node-vector leaf-index) input-range-array))

    ;; When only either upper-bound or lower-bound is bounded, set zero.
    (loop for i from 0 below dim do
      (when (= (aref input-range-array 0 i) most-negative-double-float)
        (setf (aref input-range-array 0 i) 0d0))
      (when (= (aref input-range-array 1 i) most-positive-double-float)
        (setf (aref input-range-array 1 i) 0d0)))

    (loop for i from 0 below dim do
      (setf (aref result i) (/ (+ (aref input-range-array 0 i)
                                  (aref input-range-array 1 i))
                               2d0)))
    result))
