(in-package :cl-user)

(defpackage cl-random-forest-test/fixture
  (:use :cl)
  (:import-from :cl-random-forest/src/utils
                :clol-dataset->datamatrix/target)
  (:export :*dataset-dir*
           :+a9a-dim+
           :+letter-dim+
           :+letter-n-class+
           :fetch-a9a
           :fetch-letter
           :a9a-train
           :a9a-test
           :letter-train
           :letter-test
           :approximately-equal
           :n-times-average
           :with-serial-kernel
           :with-parallel-kernel))
(in-package :cl-random-forest-test/fixture)

;;;; Dataset location

(defparameter *dataset-dir*
  (merge-pathnames #P"dataset/" (asdf:system-source-directory :cl-random-forest))
  "Directory the LIBSVM-format test datasets are downloaded into.")

(ensure-directories-exist *dataset-dir*)

(defconstant +a9a-dim+ 123)
(defconstant +letter-dim+ 16)
(defconstant +letter-n-class+ 26)

;;;; Pathname helpers for wget

(defun cat (&rest args)
  (apply #'concatenate 'string args))

(defun format-directory (p)
  (assert (eq (car (pathname-directory p)) :absolute))
  (reduce (lambda (a b) (cat a b "/"))
          (cons "/" (cdr (pathname-directory p)))))

(defun format-filename (p)
  (if (pathname-type p)
      (format nil "~A.~A" (pathname-name p) (pathname-type p))
      (format nil "~A"    (pathname-name p))))

(defun format-pathname (p)
  (let ((filename (format-filename p)))
    (if filename
        (cat (format-directory p) (format-filename p))
        (format-directory p))))

;;;; Dataset download

(defun fetch-file (url pathname)
  "Download URL to PATHNAME with wget unless PATHNAME already exists."
  (unless (uiop:file-exists-p pathname)
    (uiop:run-program (list "wget" url "-O" (format-pathname pathname)))))

(defun fetch-a9a ()
  "Download the a9a dataset into *DATASET-DIR* unless it is already there."
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
              (merge-pathnames "a9a" *dataset-dir*))
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t"
              (merge-pathnames "a9a.t" *dataset-dir*)))

(defun fetch-letter ()
  "Download the letter dataset into *DATASET-DIR* unless it is already there."
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale"
              (merge-pathnames "letter.scale" *dataset-dir*))
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t"
              (merge-pathnames "letter.scale.t" *dataset-dir*)))

;;;; Lazily loaded datasets
;;;
;;; Each accessor returns (values datamatrix target) and caches the result, so a
;;; single test can be run on its own and still get the data it needs.

(defvar *a9a-train-cache* nil)
(defvar *a9a-test-cache* nil)
(defvar *letter-train-cache* nil)
(defvar *letter-test-cache* nil)

(defun load-a9a (filename)
  "Read FILENAME from *DATASET-DIR* as a9a and return (values datamatrix target).
The a9a labels are +1/-1; they are remapped to class ids 0/1."
  (let ((data (clol.utils:read-data (merge-pathnames filename *dataset-dir*) +a9a-dim+)))
    (dolist (datum data)
      (setf (car datum) (if (> (car datum) 0d0) 0 1)))
    (clol-dataset->datamatrix/target data)))

(defun load-letter (filename)
  "Read FILENAME from *DATASET-DIR* as letter and return (values datamatrix target)."
  (let ((data (clol.utils:read-data (merge-pathnames filename *dataset-dir*)
                                    +letter-dim+ :multiclass-p t)))
    (clol-dataset->datamatrix/target data)))

(defun a9a-train ()
  "Return (values datamatrix target) for the a9a training set, loading it on first call."
  (unless *a9a-train-cache*
    (fetch-a9a)
    (multiple-value-bind (datamatrix target) (load-a9a "a9a")
      (setf *a9a-train-cache* (cons datamatrix target))))
  (values (car *a9a-train-cache*) (cdr *a9a-train-cache*)))

(defun a9a-test ()
  "Return (values datamatrix target) for the a9a test set, loading it on first call."
  (unless *a9a-test-cache*
    (fetch-a9a)
    (multiple-value-bind (datamatrix target) (load-a9a "a9a.t")
      (setf *a9a-test-cache* (cons datamatrix target))))
  (values (car *a9a-test-cache*) (cdr *a9a-test-cache*)))

(defun letter-train ()
  "Return (values datamatrix target) for the letter training set, loading it on first call."
  (unless *letter-train-cache*
    (fetch-letter)
    (multiple-value-bind (datamatrix target) (load-letter "letter.scale")
      (setf *letter-train-cache* (cons datamatrix target))))
  (values (car *letter-train-cache*) (cdr *letter-train-cache*)))

(defun letter-test ()
  "Return (values datamatrix target) for the letter test set, loading it on first call."
  (unless *letter-test-cache*
    (fetch-letter)
    (multiple-value-bind (datamatrix target) (load-letter "letter.scale.t")
      (setf *letter-test-cache* (cons datamatrix target))))
  (values (car *letter-test-cache*) (cdr *letter-test-cache*)))

;;;; Assertion helpers

(defun approximately-equal (x y &optional (delta 1d0))
  "Return true when X and Y are within DELTA. X may be a double-float, vector or list."
  (flet ((andf (x y) (and x y))
         (close? (x y) (< (abs (- x y)) delta)))
    (etypecase x
      (double-float (close? x y))
      (vector (reduce #'andf (map 'vector #'close? x y)))
      (list (reduce #'andf (mapcar #'close? x y))))))

(defmacro n-times-average (n-times &body body)
  "Evaluate BODY N-TIMES and return the mean of its values as a double-float."
  `(coerce (/ (loop repeat ,n-times
                    sum (progn ,@body))
              ,n-times)
           'double-float))

;;;; lparallel kernel scoping
;;;
;;; The old test file setf'd lparallel:*kernel* at toplevel and never called
;;; END-KERNEL, leaking a 4-worker kernel per parallel test and leaving the
;;; kernel set when the file finished. These macros bind it instead, so each
;;; test is independent of the order the suites run in.

(defmacro with-serial-kernel (&body body)
  "Evaluate BODY with lparallel:*kernel* bound to NIL, forcing serial execution."
  `(let ((lparallel:*kernel* nil))
     ,@body))

(defmacro with-parallel-kernel ((&optional (n 4)) &body body)
  "Evaluate BODY with lparallel:*kernel* bound to a fresh N-worker kernel, then shut it down."
  `(let ((lparallel:*kernel* (lparallel:make-kernel ,n)))
     (unwind-protect (progn ,@body)
       (lparallel:end-kernel :wait t))))
