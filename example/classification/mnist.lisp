;; -*- coding:utf-8; mode:lisp -*-

;; set dynamic-space-size >= 2500

(in-package :cl-random-forest)

;;; Load Dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; MNIST data
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist
(defparameter dir (asdf:system-relative-pathname :cl-random-forest "dataset/"))
(defparameter mnist-dim 784)
(defparameter mnist-n-class 10)
(defvar mnist-datamatrix)
(defvar mnist-target)
(defvar mnist-datamatrix-test)
(defvar mnist-target-test)

(defun get-mnist-dataset ()
  (ensure-directories-exist dir)
  (let ((base-url "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass"))
    (flet ((download-file (filename)
             (uiop:run-program
              (format nil "cd ~A ; [ -e ~A ] || wget ~A/~A" dir filename base-url filename)))
           (expand-file (filename)
             (uiop:run-program (format nil "cd ~A ; [ -e ~A ] || bunzip2 ~A"
                                       dir (subseq filename 0 (- (length filename) 4))  filename))))
      (format t "Downloading mnist.scale.bz2~%")
      (download-file "mnist.scale.bz2")
      (format t "Expanding mnist.scale.bz2~%")
      (expand-file "mnist.scale.bz2")
      (format t "Downloading mnist.scale.t.bz2~%")
      (download-file "mnist.scale.t.bz2")
      (format t "Expanding mnist.scale.t.bz2~%")
      (expand-file "mnist.scale.t.bz2"))))

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

(get-mnist-dataset)
(read-mnist-dataset)

;;; Make Decision Tree ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter mnist-dtree
  (make-dtree mnist-n-class mnist-datamatrix mnist-target
              :max-depth 15 :n-trial 28 :min-region-samples 5))

;; Prediction
(predict-dtree mnist-dtree mnist-datamatrix 0) ; => 5 (correct)

;; Testing with training data
(test-dtree mnist-dtree mnist-datamatrix mnist-target)

;; Accuracy: 90.37333%, Correct: 54224, Total: 60000

;; Testing with test data
(test-dtree mnist-dtree mnist-datamatrix-test mnist-target-test)
;; Accuracy: 81.52%, Correct: 8152, Total: 10000

;;; Make Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Enable/Disable parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
(setf lparallel:*kernel* nil)

;; 6.079 seconds (1 core), 2.116 seconds (4 core)
(defparameter mnist-forest
  (make-forest mnist-n-class mnist-datamatrix mnist-target
               :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 10 :min-region-samples 5))

;; Prediction
(predict-forest mnist-forest mnist-datamatrix 0) ; => 5 (correct)

;; Testing with test data
;; 4.786 seconds, Accuracy: 93.38%
(test-forest mnist-forest mnist-datamatrix-test mnist-target-test)

;; 42.717 seconds (1 core), 13.24 seconds (4 core)
(defparameter mnist-forest-tall
  (make-forest mnist-n-class mnist-datamatrix mnist-target
               :n-tree 100 :bagging-ratio 1.0 :max-depth 15 :n-trial 28 :min-region-samples 5))

;; 2.023 seconds, Accuracy: 96.62%
(test-forest mnist-forest-tall mnist-datamatrix-test mnist-target-test)

;;; Global Refinement of Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Generate sparse data from Random Forest

;; 6.255 seconds (1 core), 1.809 seconds (4 core)
(defparameter mnist-refine-dataset
  (make-refine-dataset mnist-forest mnist-datamatrix))

;; 0.995 seconds (1 core), 0.322 seconds (4 core)
(defparameter mnist-refine-test
  (make-refine-dataset mnist-forest mnist-datamatrix-test))

(defparameter mnist-refine-learner (make-refine-learner mnist-forest))

;; 4.347 seconds (1 core), 2.281 seconds (4 core), Accuracy: 98.259%
(train-refine-learner-process mnist-refine-learner mnist-refine-dataset mnist-target
                              mnist-refine-test mnist-target-test)

(test-refine-learner mnist-refine-learner mnist-refine-test mnist-target-test)

;; 5.859 seconds (1 core), 4.090 seconds (4 core), Accuracy: 98.29%
(loop repeat 5 do
  (train-refine-learner mnist-refine-learner mnist-refine-dataset mnist-target)
  (test-refine-learner mnist-refine-learner mnist-refine-test mnist-target-test))

;; Make a prediction
(predict-refine-learner mnist-forest mnist-refine-learner mnist-datamatrix-test 0)

;;; Global Pruning of Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(length (collect-leaf-parent mnist-forest)) ; => 98008
(pruning! mnist-forest mnist-refine-learner 0.1) ; 0.328 seconds
(length (collect-leaf-parent mnist-forest)) ; => 93228

;; Re-learning refine learner
(defparameter mnist-refine-dataset (make-refine-dataset mnist-forest mnist-datamatrix))
(defparameter mnist-refine-test (make-refine-dataset mnist-forest mnist-datamatrix-test))
(defparameter mnist-refine-learner (make-refine-learner mnist-forest))
(time
 (loop repeat 10 do
   (train-refine-learner mnist-refine-learner mnist-refine-dataset mnist-target)
   (test-refine-learner mnist-refine-learner mnist-refine-test mnist-target-test)))

;; Accuracy: Accuracy: 98.27%

(loop repeat 10 do
  (sb-ext:gc :full t)
  (room)
  (format t "~%Making mnist-refine-dataset~%")
  (defparameter mnist-refine-dataset (make-refine-dataset mnist-forest mnist-datamatrix))
  (format t "Making mnist-refine-test~%")
  (defparameter mnist-refine-test (make-refine-dataset mnist-forest mnist-datamatrix-test))
  (format t "Re-learning~%")
  (defparameter mnist-refine-learner (make-refine-learner mnist-forest))
  (train-refine-learner-process mnist-refine-learner mnist-refine-dataset mnist-target
                                mnist-refine-test mnist-target-test)
  (test-refine-learner mnist-refine-learner mnist-refine-test mnist-target-test)
  (format t "Pruning. leaf-size: ~A" (length (collect-leaf-parent mnist-forest)))
  (pruning! mnist-forest mnist-refine-learner 0.5)
  (format t " -> ~A ~%" (length (collect-leaf-parent mnist-forest))))

;;; n-fold cross-validation

(defparameter n-fold 5)

(cross-validation-forest-with-refine-learner
 n-fold mnist-n-class mnist-datamatrix mnist-target
 :n-tree 100 :bagging-ratio 0.1 :max-depth 10 :n-trial 28 :gamma 10d0 :min-region-samples 5)
