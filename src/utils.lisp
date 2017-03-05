(in-package :cl-random-forest)

;;; read

(defun clol-dataset->datamatrix/target (dataset)
  (let* ((len (length dataset))
         (data-dimension (length (cdar dataset)))
         (target (make-array len :element-type 'fixnum))
         (datamatrix (make-array (list len data-dimension) :element-type 'double-float)))
    (loop for i from 0 to (1- len)
          for datum in dataset
          do
             (setf (aref target i) (car datum))
             (loop for j from 0 to (1- data-dimension) do
               (setf (aref datamatrix i j) (aref (cdr datum) j))))
    (values datamatrix target)))

;;; write

(defun format-datum (datum stream)
  (format stream "class~A" (car datum))
  (loop for elem across (cdr datum) do
    (format stream "~t~f" elem))
  (format stream "~%"))

(defun write-to-r-format-from-clol-dataset (dataset file)
  (with-open-file (f file :direction :output :if-exists :supersede)
    (dolist (datum dataset)
      (format-datum datum f))))
