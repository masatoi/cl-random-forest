(in-package :cl-user)
(defpackage cl-random-forest-test
  (:use :cl
   :cl-random-forest
        :cl-random-forest.utils
   :prove))
(in-package :cl-random-forest-test)

;; NOTE: To run this test file, execute `(asdf:test-system :cl-random-forest)' in your Lisp.

(defparameter *dataset-dir* (merge-pathnames #P"t/dataset/" (asdf:system-source-directory :cl-random-forest)))

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

(defun fetch-letter ()
  (let ((file (merge-pathnames "letter.scale" *dataset-dir*))
        (file.t (merge-pathnames "letter.scale.t" *dataset-dir*)))
    ;; fetch dataset
    (when (not (uiop:file-exists-p file))
      (uiop:run-program
       (list "wget"
             "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale"
             "-O" (format-pathname file))))
    (when (not (uiop:file-exists-p file.t))
      (uiop:run-program
       (list "wget"
             "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t"
             "-O" (format-pathname file.t))))))

(defun fetch-a9a ()
  (let ((file (merge-pathnames "a9a" *dataset-dir*))
        (file.t (merge-pathnames "a9a.t" *dataset-dir*)))
    ;; fetch dataset
    (when (not (uiop:file-exists-p file))
      (uiop:run-program
       (list "wget"
             "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
             "-O" (format-pathname file))))
    (when (not (uiop:file-exists-p file.t))
      (uiop:run-program
       (list "wget"
             "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t"
             "-O" (format-pathname file.t))))))

(defun approximately-equal (x y &optional (delta 0.2d0))
  (flet ((andf (x y) (and x y))
         (close? (x y) (< (abs (- x y)) delta)))
    (etypecase x
      (double-float (close? x y))
      (vector (reduce #'andf (map 'vector #'close? x y)))
      (list (reduce #'andf (mapcar #'close? x y))))))

(defmacro n-times-average (n-times &body body)
  `(coerce (/ (loop repeat ,n-times
                    sum (progn ,@body))
              ,n-times)
           'double-float))

(plan nil)

(format t ";;; Fetch dataset~%")

(fetch-letter)
(fetch-a9a)

(format t ";;; File exists check~%")

(ok (and (uiop:file-exists-p (merge-pathnames "letter.scale" *dataset-dir*))
         (uiop:file-exists-p (merge-pathnames "letter.scale.t" *dataset-dir*))
         (uiop:file-exists-p (merge-pathnames "a9a" *dataset-dir*))
         (uiop:file-exists-p (merge-pathnames "a9a.t" *dataset-dir*))))

(format t ";;; Binary classification~%")

(format t ";; A9A: Prepare dataset~%")
(defparameter a9a-dim 123)
(defvar a9a-datamatrix)
(defvar a9a-target)
(defvar a9a-datamatrix-test)
(defvar a9a-target-test)

(ok
 (let ((a9a-train (clol.utils:read-data (merge-pathnames "a9a" *dataset-dir*) a9a-dim))
       (a9a-test (clol.utils:read-data (merge-pathnames "a9a.t" *dataset-dir*) a9a-dim)))
   (dolist (datum a9a-train)
     (if (> (car datum) 0d0)
         (setf (car datum) 0)
         (setf (car datum) 1)))
   (dolist (datum a9a-test)
     (if (> (car datum) 0d0)
         (setf (car datum) 0)
         (setf (car datum) 1)))

   (multiple-value-bind (datamat target)
       (clol-dataset->datamatrix/target a9a-train)
     (setf a9a-datamatrix datamat)
     (setf a9a-target target))

   (multiple-value-bind (datamat target)
       (clol-dataset->datamatrix/target a9a-test)
     (setf a9a-datamatrix-test datamat)
     (setf a9a-target-test target))

   (and a9a-datamatrix
        a9a-target
        a9a-datamatrix-test
        a9a-target-test)))

(format t ";; A9A: Make a decision tree~%")

(is
 (n-times-average
  100
  (let ((a9a-dtree (make-dtree 2 a9a-dim a9a-datamatrix a9a-target :max-depth 20)))
    (test-dtree a9a-dtree a9a-datamatrix-test a9a-target-test)))
 82.23217010498047d0
 :test #'approximately-equal)

(format t ";; A9A: Make a random forest~%")
(setf lparallel:*kernel* nil)

(is
 (n-times-average
  20
  (trivial-garbage:gc :full t)
  (let ((a9a-forest
          (make-forest 2 a9a-dim a9a-datamatrix a9a-target
                       :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10)))
    (test-forest a9a-forest a9a-datamatrix-test a9a-target-test)))
 84.07d0
 :test #'approximately-equal)

#+sbcl
(progn
  (format t ";; A9A: Make a random forest (Parallel)~%")
  (setf lparallel:*kernel* (lparallel:make-kernel 4))

  (is
   (n-times-average
    20
    (trivial-garbage:gc :full t)
    (let ((a9a-forest
            (make-forest 2 a9a-dim a9a-datamatrix a9a-target
                         :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10)))
      (test-forest a9a-forest a9a-datamatrix-test a9a-target-test)))
   84.07d0
   :test #'approximately-equal))

(format t ";; A9A: Make a random forest with global refinement~%")
(setf lparallel:*kernel* nil)

(is
 (n-times-average
  20
  (trivial-garbage:gc :full t)
  (let* ((a9a-forest
           (make-forest 2 a9a-dim a9a-datamatrix a9a-target
                        :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10))
         (a9a-refine-dataset (make-refine-dataset a9a-forest a9a-datamatrix))
         (a9a-refine-test (make-refine-dataset a9a-forest a9a-datamatrix-test))
         (a9a-refine-learner (make-refine-learner a9a-forest)))
    (train-refine-learner-process a9a-refine-learner
                                  a9a-refine-dataset a9a-target
                                  a9a-refine-test a9a-target-test)
    (test-refine-learner a9a-refine-learner a9a-refine-test a9a-target-test)))
 80.98789
 :test #'approximately-equal)


#+sbcl
(progn
  (format t ";; A9A: Make a random forest with global refinement (Parallel)~%")
  (setf lparallel:*kernel* (lparallel:make-kernel 4))
  
  (is
   (n-times-average
    20
    (trivial-garbage:gc :full t)
    (let* ((a9a-forest
             (make-forest 2 a9a-dim a9a-datamatrix a9a-target
                          :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10))
           (a9a-refine-dataset (make-refine-dataset a9a-forest a9a-datamatrix))
           (a9a-refine-test (make-refine-dataset a9a-forest a9a-datamatrix-test))
           (a9a-refine-learner (make-refine-learner a9a-forest)))
      (train-refine-learner-process a9a-refine-learner
                                    a9a-refine-dataset a9a-target
                                    a9a-refine-test a9a-target-test)
      (test-refine-learner a9a-refine-learner a9a-refine-test a9a-target-test)))
   80.98789
   :test #'approximately-equal))

;;; Multiclass classification
(format t ";;; Multiclass classification~%")

(format t ";; LETTER: Prepare dataset~%")
(defparameter letter-dim 16)
(defparameter letter-n-class 26)
(defvar letter-datamatrix)
(defvar letter-target)
(defvar letter-datamatrix-test)
(defvar letter-target-test)

(ok
 (let ((letter-train (clol.utils:read-data (merge-pathnames "letter.scale" *dataset-dir*)
                                           letter-dim :multiclass-p t))
       (letter-test (clol.utils:read-data (merge-pathnames "letter.scale.t" *dataset-dir*)
                                          letter-dim :multiclass-p t)))
   (multiple-value-bind (datamat target)
       (clol-dataset->datamatrix/target letter-train)
     (setf letter-datamatrix datamat)
     (setf letter-target target))

   (multiple-value-bind (datamat target)
       (clol-dataset->datamatrix/target letter-test)
     (setf letter-datamatrix-test datamat)
     (setf letter-target-test target))

   (and letter-datamatrix
        letter-target
        letter-datamatrix-test
        letter-target-test)))

(format t ";; LETTER: Make a decision tree~%")

(is
 (n-times-average
  100
  (let ((letter-dtree (make-dtree letter-n-class letter-dim letter-datamatrix letter-target :max-depth 20)))
    (test-dtree letter-dtree letter-datamatrix-test letter-target-test)))
 83.9534912109375d0
 :test #'approximately-equal)

(format t ";; LETTER: Make a random forest~%")
(setf lparallel:*kernel* nil)

(is
 (n-times-average
  20
  (trivial-garbage:gc :full t)
  (let ((letter-forest
          (make-forest letter-n-class letter-dim letter-datamatrix letter-target
                       :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10)))
    (test-forest letter-forest letter-datamatrix-test letter-target-test)))
 89.05402374267578d0
 :test #'approximately-equal)

#+sbcl
(progn
  (format t ";; LETTER: Make a random forest (Parallel)~%")
  (setf lparallel:*kernel* (lparallel:make-kernel 4))

  (is
   (n-times-average
    20
    (trivial-garbage:gc :full t)
    (let ((letter-forest
            (make-forest letter-n-class letter-dim letter-datamatrix letter-target
                         :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10)))
      (test-forest letter-forest letter-datamatrix-test letter-target-test)))
   89.05402374267578d0
   :test #'approximately-equal))

(format t ";; LETTER: Make a random forest with global refinement~%")
(setf lparallel:*kernel* nil)

(is
 (n-times-average
  20
  (trivial-garbage:gc :full t)
  (let* ((letter-forest
           (make-forest letter-n-class letter-dim letter-datamatrix letter-target
                        :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10))
         (letter-refine-dataset (make-refine-dataset letter-forest letter-datamatrix))
         (letter-refine-test (make-refine-dataset letter-forest letter-datamatrix-test))
         (letter-refine-learner (make-refine-learner letter-forest)))
    (train-refine-learner-process letter-refine-learner
                                  letter-refine-dataset letter-target
                                  letter-refine-test letter-target-test)
    (test-refine-learner letter-refine-learner letter-refine-test letter-target-test)))
 97.06802368164063d0
 :test #'approximately-equal)

#+sbcl
(progn
  (format t ";; LETTER: Make a random forest with global refinement (Parallel)~%")
  (setf lparallel:*kernel* (lparallel:make-kernel 4))

  (is
   (n-times-average
    20
    (trivial-garbage:gc :full t)
    (let* ((letter-forest
             (make-forest letter-n-class letter-dim letter-datamatrix letter-target
                          :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 10))
           (letter-refine-dataset (make-refine-dataset letter-forest letter-datamatrix))
           (letter-refine-test (make-refine-dataset letter-forest letter-datamatrix-test))
           (letter-refine-learner (make-refine-learner letter-forest)))
      (train-refine-learner-process letter-refine-learner
                                    letter-refine-dataset letter-target
                                    letter-refine-test letter-target-test)
      (test-refine-learner letter-refine-learner letter-refine-test letter-target-test)))
   97.06802368164063d0
   :test #'approximately-equal))

(finalize)
