(defpackage :cl-random-forest/src/utils
  (:use :cl)
  (:nicknames :cl-random-forest.utils :clrf.utils)
  (:export #:dotimes/pdotimes
           #:mapcar/pmapcar
           #:mapc/pmapc
           #:push-ntimes
           #:clol-dataset->datamatrix/target
           #:clol-dataset->datamatrix/target-regression
           #:read-data
           #:read-data-regression
           #:write-to-r-format-from-clol-dataset))

(in-package :cl-random-forest/src/utils)

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
  (alexandria:with-gensyms (var vec)
    (alexandria:once-only (n)
      `(let ((,vec (make-array ,n)))
         (if lparallel:*kernel*
             (lparallel:pdotimes (,var ,n)
               (setf (aref ,vec ,var) (progn ,@body)))
             (dotimes (,var ,n)
               (setf (aref ,vec ,var) (progn ,@body))))
         (setf ,lst (coerce ,vec 'list))))))

;;; Read from cl-online-learning dataset

(defun clol-dataset->datamatrix/target (dataset)
  (let* ((len (length dataset))
         (data-dimension (length (cdar dataset)))
         (target (make-array len :element-type 'fixnum))
         (datamatrix (make-array (list len data-dimension) :element-type 'single-float)))
    (loop for i from 0 below len
          for datum in dataset
          do (setf (aref target i) (car datum))
             (loop for j from 0 below data-dimension do
               (setf (aref datamatrix i j) (aref (cdr datum) j))))
    (values datamatrix target)))

(defun clol-dataset->datamatrix/target-regression (dataset)
  (let* ((len (length dataset))
         (data-dimension (length (cdar dataset)))
         (target (make-array len :element-type 'single-float))
         (datamatrix (make-array (list len data-dimension) :element-type 'single-float)))
    (loop for i from 0 below len
          for datum in dataset
          do (setf (aref target i) (car datum))
             (loop for j from 0 below data-dimension do
               (setf (aref datamatrix i j) (aref (cdr datum) j))))
    (values datamatrix target)))

(defmacro do-index-value-list ((index value list) &body body)
  (let ((iter (gensym))
        (inner-list (gensym)))
    `(labels ((,iter (,inner-list)
                (when ,inner-list
                  (let ((,index (car ,inner-list))
                        (,value (cadr ,inner-list)))
                    ,@body)
                  (,iter (cddr ,inner-list)))))
       (,iter ,list))))

;;; Read from libsvm format dataset

(defun read-data (data-path data-dimension)
  (multiple-value-bind (data-list dim)
      (svmformat:parse-file data-path)
    (let* ((dim (if data-dimension data-dimension dim))
           (len (length data-list))
           (target (make-array len :element-type 'fixnum :initial-element 0))
           (datamatrix (make-array (list len dim)
                                   :element-type 'single-float
                                   :initial-element 0.0)))
      (loop for i from 0
            for datum in data-list
            do
               (setf (aref target i) (1- (car datum)))
               (do-index-value-list (j v (cdr datum))
                 (setf (aref datamatrix i (1- j)) (coerce v 'single-float))))
      (values datamatrix target))))

(defun read-data-regression (data-path data-dimension)
  (multiple-value-bind (data-list dim)
      (svmformat:parse-file data-path)
    (let* ((dim (if data-dimension data-dimension dim))
           (len (length data-list))
           (target (make-array len :element-type 'single-float :initial-element 0.0))
           (datamatrix (make-array (list len dim)
                                   :element-type 'single-float
                                   :initial-element 0.0)))
      (loop for i from 0
            for datum in data-list
            do
               (setf (aref target i) (coerce (car datum) 'single-float))
               (do-index-value-list (j v (cdr datum))
                 (setf (aref datamatrix i (1- j)) (coerce v 'single-float))))
      (values datamatrix target))))

;;; Write for R format

(defun format-datum (datum stream)
  (format stream "class~A" (car datum))
  (loop for elem across (cdr datum) do
    (format stream "~t~f" elem))
  (format stream "~%"))

(defun write-to-r-format-from-clol-dataset (dataset file)
  (with-open-file (f file :direction :output :if-exists :supersede)
    (dolist (datum dataset)
      (format-datum datum f))))
