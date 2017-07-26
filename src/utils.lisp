;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-random-forest.utils
  (:use :cl)
  (:nicknames :clrf.utils)
  (:export :dotimes/pdotimes :mapcar/pmapcar :mapc/pmapc :push-ntimes
           :clol-dataset->datamatrix/target
           :clol-dataset->datamatrix/target-regression
           :write-to-r-format-from-clol-dataset))

(in-package :cl-random-forest.utils)

;;; parallelizaion utils

(defmacro dotimes/pdotimes ((var n) &body body)
  `(if lparallel:*kernel*
       (lparallel:pdotimes (,var ,n) ,@body)
       (dotimes (,var ,n) ,@body)))

(defmacro mapcar/pmapcar (fn &rest lsts)
  `(if lparallel:*kernel*
       (lparallel:pmapcar ,fn ,@lsts)
       (mapcar ,fn ,@lsts)))

(defmacro mapc/pmapc (fn &rest lsts)
  `(if lparallel:*kernel*
       (lparallel:pmapc ,fn ,@lsts)
       (mapc ,fn ,@lsts)))

(defmacro push-ntimes (n lst &body body)
  (let ((var (gensym)))
    `(if lparallel:*kernel*
         (lparallel:pdotimes (,var ,n)
           (progn
             ,var
             (push (progn ,@body) ,lst)))
         (dotimes (,var ,n)
           (progn
             ,var
             (push (progn ,@body) ,lst))))))

;;; read

(defun clol-dataset->datamatrix/target (dataset)
  (let* ((len (length dataset))
         (data-dimension (length (cdar dataset)))
         (target (make-array len :element-type 'fixnum))
         (datamatrix (make-array (list len data-dimension) :element-type 'double-float)))
    (loop for i from 0 to (1- len)
          for datum in dataset
          do (setf (aref target i) (car datum))
             (loop for j from 0 to (1- data-dimension) do
               (setf (aref datamatrix i j) (aref (cdr datum) j))))
    (values datamatrix target)))

(defun clol-dataset->datamatrix/target-regression (dataset)
  (let* ((len (length dataset))
         (data-dimension (length (cdar dataset)))
         (target (make-array (list len 1) :element-type 'double-float))
         (datamatrix (make-array (list len data-dimension) :element-type 'double-float)))
    (loop for i from 0 to (1- len)
          for datum in dataset
          do (setf (aref target i 0) (car datum))
             (loop for j from 0 to (1- data-dimension) do
               (setf (aref datamatrix i j) (aref (cdr datum) j))))
    (values datamatrix target)))

;;; write for R

(defun format-datum (datum stream)
  (format stream "class~A" (car datum))
  (loop for elem across (cdr datum) do
    (format stream "~t~f" elem))
  (format stream "~%"))

(defun write-to-r-format-from-clol-dataset (dataset file)
  (with-open-file (f file :direction :output :if-exists :supersede)
    (dolist (datum dataset)
      (format-datum datum f))))
