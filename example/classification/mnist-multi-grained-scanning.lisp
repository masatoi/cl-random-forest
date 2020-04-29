(in-package :clrf)

(defparameter dir (asdf:system-relative-pathname :cl-random-forest "dataset/"))
(defparameter mnist-dim 784)
(defparameter mnist-n-class 10)
(defvar mnist-datamatrix)
(defvar mnist-target)
(defvar mnist-datamatrix-test)
(defvar mnist-target-test)

(defun read-mnist-dataset ()
  (format t "Reading training data~%")
  (multiple-value-bind (datamat target)
      (read-data (merge-pathnames "mnist.scale" dir) mnist-dim)
    (setf mnist-datamatrix datamat
          mnist-target target))

  (format t "Reading test data~%")
  (multiple-value-bind (datamat target)
      (read-data (merge-pathnames "mnist.scale.t" dir) mnist-dim)
    (setf mnist-datamatrix-test datamat
          mnist-target-test target))

  ;; Add 1 to labels because the labels of this dataset begin from 0
  (loop for i from 0 below (length mnist-target) do
           (incf (aref mnist-target i)))
  (loop for i from 0 below (length mnist-target-test) do
           (incf (aref mnist-target-test i))))

(read-mnist-dataset)

(setf lparallel:*kernel* (lparallel:make-kernel 4))

(load (asdf:system-relative-pathname :cl-random-forest "src/multi-grained-scanning.lisp"))

(defparameter patch-size '(26 26))
(defparameter stride 1)
(defparameter data-magnification (number-of-patches '(28 28) patch-size stride))

;;; Data magnification (12.721 sec)
(time 
 (progn
   (format t "data-magnification. magnification-rate: ~A~%" data-magnification)
   (defparameter *mnist-patch-datamatrix*
     (make-patch-datamatrix mnist-datamatrix '(28 28) patch-size stride))
   (defparameter *mnist-patch-datamatrix-test*
     (make-patch-datamatrix mnist-datamatrix-test '(28 28) patch-size stride))
   (defparameter *mnist-patch-target* (make-patch-target mnist-target '(28 28) patch-size stride))
   (defparameter *mnist-patch-target-test* (make-patch-target mnist-target-test '(28 28) patch-size stride))))

;;; Training (192.976 sec)
(time
 (progn
   (format t "Building patch random forest~%")
   (defparameter *mnist-patch-forest*
     (make-forest mnist-n-class *mnist-patch-datamatrix* *mnist-patch-target*
                  :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 20 :min-region-samples 5))

   (format t "Making refine dataset~%")
   (defparameter *mnist-refine-dataset*
     (make-refine-dataset-from-patch-datamatrix
      *mnist-patch-forest* mnist-datamatrix *mnist-patch-datamatrix*))

   (defparameter *mnist-refine-test*
     (make-refine-dataset-from-patch-datamatrix
      *mnist-patch-forest* mnist-datamatrix-test *mnist-patch-datamatrix-test*))

   (format t "Making refine learner~%")
   (defparameter mnist-refine-learner
     (make-data-augmented-refine-learner *mnist-patch-forest* data-magnification))

   (format t "Training refine learner~%")
   (train-refine-learner-process mnist-refine-learner
                                 *mnist-refine-dataset* mnist-target
                                 *mnist-refine-test* mnist-target-test)))

;;; Global Pruning
  
(time
 (loop
   repeat 20
   do (setf *mnist-refine-dataset* nil
            *mnist-refine-test* nil
            mnist-refine-learner nil)
      (sb-ext:gc :full t)
      (room)
      (format t "~%Making mnist-refine-dataset~%")
      (defparameter *mnist-refine-dataset*
        (make-refine-dataset-from-patch-datamatrix
         *mnist-patch-forest* mnist-datamatrix *mnist-patch-datamatrix*))
   
      (format t "Making mnist-refine-test~%")
      (defparameter *mnist-refine-test*
        (make-refine-dataset-from-patch-datamatrix
         *mnist-patch-forest* mnist-datamatrix-test *mnist-patch-datamatrix-test*))
      (format t "Re-learning~%")
      (defparameter mnist-refine-learner
        (make-data-augmented-refine-learner *mnist-patch-forest* data-magnification))
      (train-refine-learner-process mnist-refine-learner
                                    *mnist-refine-dataset* mnist-target
                                    *mnist-refine-test* mnist-target-test)
      (format t "Pruning. leaf-size: ~A" (length (collect-leaf-parent *mnist-patch-forest*)))
      (let ((*n-patch* data-magnification))
        (pruning! *mnist-patch-forest* mnist-refine-learner 0.5))
      (format t " -> ~A ~%" (length (collect-leaf-parent *mnist-patch-forest*)))))


;;; Cleaning
(progn
  (setf *mnist-patch-datamatrix* nil
        *mnist-patch-datamatrix-test* nil
        *mnist-patch-target* nil
        *mnist-patch-target-test* nil
        *mnist-patch-forest* nil
        *mnist-refine-dataset* nil
        *mnist-refine-test* nil
        mnist-refine-learner nil)

  (sb-ext:gc :full t))

;; (patch-size '(20 20))
;; (stride 4)
;; Accuracy: 98.56%, Correct: 9856, Total: 10000


;; (patch-size '(20 20))
;; (stride 2)
;; intractable

;; (patch-size '(24 24))
;; (stride 2)
;; Accuracy: 98.82
;; n-tree: 500
;; max-depth: 10

;; (patch-size '(24 24))
;; (stride 2)
;; Accuracy: 98.79
;; n-tree: 1000
;; max-depth: 10

;; (patch-size '(24 24))
;; (stride 2)
;; Accuracy: 98.82
;; n-tree: 500
;; max-depth: 10

;; (patch-size '(26 26))
;; (stride 1)
;; Accuracy: 98.84
;; n-tree: 500
;; max-depth: 10

;; (patch-size '(26 26))
;; (stride 1)
;; Accuracy: 98.92%, Correct: 9892, Total: 10000
;; n-tree: 500
;; max-depth: 10
;; n-trial: 20
;; :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 20 :min-region-samples 5
;; 99.01%!

;; (patch-size '(26 26))
;; (stride 1)
;; Accuracy: 98.88%, Correct: 9892, Total: 10000
;; n-tree: 500
;; max-depth: 10
;; n-trial: 20
;; :n-tree 500 :bagging-ratio 0.2 :max-depth 10 :n-trial 20 :min-region-samples 5

;; :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 30 :min-region-samples 50
;; Accuracy: 98.75%
