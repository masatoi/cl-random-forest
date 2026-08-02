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
           :with-parallel-kernel
           :+synthetic-n-class+
           :synthetic-regression-train
           :synthetic-regression-test
           :synthetic-classification-train
           :synthetic-classification-test
           :target-stddev))
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

;;;; Synthetic datasets
;;;
;;; These need no network, so a suite built on them runs offline and in seconds.
;;; Each generator seeds its own LCG rather than sharing one: a suite has to see
;;; the same data whether it runs on its own or inside the full run, so the
;;; sequence must not depend on what else was generated first.

(defconstant +synthetic-n-class+ 4)

(defun make-lcg (seed)
  "Return a closure yielding a deterministic pseudo-random single-float in [0,1)."
  (let ((state seed))
    (lambda ()
      (setf state (mod (+ (* 1103515245 state) 12345) 2147483648))
      (/ (float state 1.0) 2147483648.0))))

(defun lcg-noise (rnd)
  "Roughly normal noise in [-3,3), summed from three uniform draws of RND."
  (* 2.0 (- (+ (funcall rnd) (funcall rnd) (funcall rnd)) 1.5)))

(defun generate-regression (seed n-datum n-dim)
  "Return (values datamatrix target) for y = 3*x0 - 2*x1 + x2^2 + 0.5*x3 + small noise.
Features are uniform over [-1,1)."
  (let ((rnd (make-lcg seed))
        (datamatrix (make-array (list n-datum n-dim) :element-type 'single-float))
        (target (make-array n-datum :element-type 'single-float)))
    (dotimes (i n-datum)
      (dotimes (j n-dim)
        (setf (aref datamatrix i j) (float (- (* 2.0 (funcall rnd)) 1.0) 1.0)))
      (setf (aref target i)
            (float (+ (* 3.0 (aref datamatrix i 0))
                      (* -2.0 (aref datamatrix i 1))
                      (expt (aref datamatrix i 2) 2)
                      (* 0.5 (aref datamatrix i 3))
                      (* 0.1 (lcg-noise rnd)))
                   1.0)))
    (values datamatrix target)))

(defun generate-classification (seed n-datum n-dim n-class signal)
  "Return (values datamatrix target). Class C gets SIGNAL added to every feature J
where (mod J N-CLASS) equals C; every other feature is noise alone."
  (let ((rnd (make-lcg seed))
        (datamatrix (make-array (list n-datum n-dim) :element-type 'single-float))
        (target (make-array n-datum :element-type 'fixnum)))
    (dotimes (i n-datum)
      (let ((class (mod i n-class)))
        (setf (aref target i) class)
        (dotimes (j n-dim)
          (setf (aref datamatrix i j)
                (+ (lcg-noise rnd) (if (= (mod j n-class) class) signal 0.0))))))
    (values datamatrix target)))

(defvar *synthetic-regression-train-cache* nil)
(defvar *synthetic-regression-test-cache* nil)
(defvar *synthetic-classification-train-cache* nil)
(defvar *synthetic-classification-test-cache* nil)

(defun synthetic-regression-train ()
  "Return (values datamatrix target) for the synthetic regression training set."
  (unless *synthetic-regression-train-cache*
    (multiple-value-bind (datamatrix target) (generate-regression 42 1000 6)
      (setf *synthetic-regression-train-cache* (cons datamatrix target))))
  (values (car *synthetic-regression-train-cache*)
          (cdr *synthetic-regression-train-cache*)))

(defun synthetic-regression-test ()
  "Return (values datamatrix target) for the synthetic regression test set."
  (unless *synthetic-regression-test-cache*
    (multiple-value-bind (datamatrix target) (generate-regression 43 500 6)
      (setf *synthetic-regression-test-cache* (cons datamatrix target))))
  (values (car *synthetic-regression-test-cache*)
          (cdr *synthetic-regression-test-cache*)))

(defun synthetic-classification-train ()
  "Return (values datamatrix target) for the synthetic classification training set."
  (unless *synthetic-classification-train-cache*
    (multiple-value-bind (datamatrix target)
        (generate-classification 44 1500 12 +synthetic-n-class+ 1.5)
      (setf *synthetic-classification-train-cache* (cons datamatrix target))))
  (values (car *synthetic-classification-train-cache*)
          (cdr *synthetic-classification-train-cache*)))

(defun synthetic-classification-test ()
  "Return (values datamatrix target) for the synthetic classification test set."
  (unless *synthetic-classification-test-cache*
    (multiple-value-bind (datamatrix target)
        (generate-classification 45 600 12 +synthetic-n-class+ 1.5)
      (setf *synthetic-classification-test-cache* (cons datamatrix target))))
  (values (car *synthetic-classification-test-cache*)
          (cdr *synthetic-classification-test-cache*)))

(defun target-stddev (target)
  "Standard deviation of TARGET -- the RMSE a constant predictor would achieve."
  (let* ((n (length target))
         (mean (/ (reduce #'+ target) n)))
    (sqrt (/ (reduce #'+ (map 'vector (lambda (y) (expt (- y mean) 2)) target)) n))))
