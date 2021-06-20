;; -*- coding:utf-8; mode:lisp -*-

;; set dynamic-space-size >= 2500

(in-package :cl-random-forest)

;;; Load Dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; KMNIST data
;; https://github.com/rois-codh/kmnist
(defparameter dir (asdf:system-relative-pathname :cl-random-forest "dataset/kmnist/"))
(defparameter mnist-dim 784)
(defparameter mnist-n-class 10)

(defun get-mnist-dataset ()
  (ensure-directories-exist dir)
  (let ((base-url "http://codh.rois.ac.jp/kmnist/dataset/kmnist"))
    (flet ((download-file (filename)
             (uiop:run-program
              (format nil "cd ~A ; [ -e ~A ] || wget ~A/~A" dir filename base-url filename)))
           (expand-file (filename)
             (uiop:run-program (format nil "cd ~A ; [ -e ~A ] || gunzip ~A"
                                       dir (subseq filename 0 (- (length filename) 3))  filename))))
      (format t "Downloading train-images-idx3-ubyte.gz~%")
      (download-file "train-images-idx3-ubyte.gz")
      (format t "Expanding train-images-idx3-ubyte.gz~%")
      (expand-file "train-images-idx3-ubyte.gz")

      (format t "Downloading train-labels-idx1-ubyte.gz~%")
      (download-file "train-labels-idx1-ubyte.gz")
      (format t "Expanding train-labels-idx1-ubyte.gz~%")
      (expand-file "train-labels-idx1-ubyte.gz")

      (format t "Downloading t10k-images-idx3-ubyte.gz~%")
      (download-file "t10k-images-idx3-ubyte.gz")
      (format t "Expanding t10k-images-idx3-ubyte.gz~%")
      (expand-file "t10k-images-idx3-ubyte.gz")

      (format t "Downloading t10k-labels-idx1-ubyte.gz~%")
      (download-file "t10k-labels-idx1-ubyte.gz")
      (format t "Expanding t10k-labels-idx1-ubyte.gz~%")
      (expand-file "t10k-labels-idx1-ubyte.gz"))))

(get-mnist-dataset)

(ql:quickload :lisp-binary)

(lisp-binary:defbinary mnist-dataset (:byte-order :big-endian)
    (magic-number 0 :type (unsigned-byte 32))
    (number-of-images 0 :type (unsigned-byte 32))
    (number-of-rows 0 :type (unsigned-byte 32))
    (number-of-columns 0 :type (unsigned-byte 32)))

(lisp-binary:defbinary mnist-labels (:byte-order :big-endian)
    (magic-number 0 :type (unsigned-byte 32))
    (number-of-items 0 :type (unsigned-byte 32))
    (target 0 :type (simple-array (unsigned-byte 8) (number-of-items))))

(defun read-mnist-dataset (file &key scaling?)
  (lisp-binary:with-open-binary-file (in file :direction :input)
    (let* ((mnist-dataset (lisp-binary:read-binary 'mnist-dataset in))
           (number-of-images (slot-value mnist-dataset 'number-of-images))
           (number-of-rows (slot-value mnist-dataset 'number-of-rows))
           (number-of-columns (slot-value mnist-dataset 'number-of-columns))
           (datamatrix (make-array (list number-of-images (* number-of-rows number-of-columns))
                                   :element-type 'single-float
                                   :initial-element 0.0)))
      (loop for i from 0 below number-of-images
            do (let ((row (lisp-binary:read-bytes (* number-of-rows number-of-columns)
                                                  in :element-type '(unsigned-byte 8))))
                 (loop for j from 0 below (* number-of-rows number-of-columns)
                       for r across row
                       do (setf (aref datamatrix i j) (/ r (if scaling? 255.0 1.0))))))
      datamatrix)))

(defun read-mnist-labels (file)
  (lisp-binary:with-open-binary-file (in file :direction :input)
    (let* ((mnist-labels (lisp-binary:read-binary 'mnist-labels in))
           (n (slot-value mnist-labels 'number-of-items))
           (target-uint (slot-value mnist-labels 'target))
           (target (make-array n :element-type 'fixnum)))
      (loop for i from 0 below n
            for x across target-uint
            do (setf (aref target i) x))
      target)))

(defparameter mnist-datamatrix
  (read-mnist-dataset (merge-pathnames "train-images-idx3-ubyte" dir) :scaling? t))

(defparameter mnist-target
  (read-mnist-labels (merge-pathnames "train-labels-idx1-ubyte" dir)))

(defparameter mnist-datamatrix-test
  (read-mnist-dataset (merge-pathnames "t10k-images-idx3-ubyte" dir) :scaling? t))

(defparameter mnist-target-test
  (read-mnist-labels (merge-pathnames "t10k-labels-idx1-ubyte" dir)))

;; (ql:quickload :clgplot)

;; (let ((arr (make-array '(28 28))))
;;   (loop for k from 0 below 10 do
;;            (loop for i from 0 below 28 do
;;                     (loop for j from 0 below 28 do
;;                              (setf (aref arr i j) (aref mnist-datamatrix k (+ (* i 28) j)))))
;;            (clgplot:splot-matrix arr)))

;;; Make Decision Tree ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter mnist-dtree
  (make-dtree mnist-n-class mnist-datamatrix mnist-target
              :max-depth 15 :n-trial 28 :min-region-samples 5))

;; Prediction
(predict-dtree mnist-dtree mnist-datamatrix 0) ; => 8 (correct)

;; Testing with training data
(test-dtree mnist-dtree mnist-datamatrix mnist-target)

;; Accuracy: 82.450005%, Correct: 49470, Total: 60000

;; Testing with test data
(test-dtree mnist-dtree mnist-datamatrix-test mnist-target-test)
;; Accuracy: 56.97%, Correct: 5697, Total: 10000

;;; Make Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Enable/Disable parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
(setf lparallel:*kernel* nil)

;; 2.987 seconds (4 core)
(time
 (defparameter mnist-forest
   (make-forest mnist-n-class mnist-datamatrix mnist-target
                :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 10 :min-region-samples 5)))

;; Prediction
(predict-forest mnist-forest mnist-datamatrix 0) ; => 8 (correct)

;; Testing with test data

;; Accuracy: 69.4%, Correct: 6940, Total: 10000 (4.775 seconds)
(test-forest mnist-forest mnist-datamatrix-test mnist-target-test)

;; 16.847 seconds (4 core)
(time
 (defparameter mnist-forest-tall
   (make-forest mnist-n-class mnist-datamatrix mnist-target
                :n-tree 100 :bagging-ratio 1.0 :max-depth 15 :n-trial 28 :min-region-samples 5)))

;; 1.291 seconds, Accuracy: 81.23%
(time (test-forest mnist-forest-tall mnist-datamatrix-test mnist-target-test))

;;; Global Refinement of Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Generate sparse data from Random Forest

;; 3.303 seconds (4 core)
(time
 (defparameter mnist-refine-dataset
   (make-refine-dataset mnist-forest mnist-datamatrix)))

;; 0.423 seconds (4 core)
(time
 (defparameter mnist-refine-test
   (make-refine-dataset mnist-forest mnist-datamatrix-test)))

(defparameter mnist-refine-learner (make-refine-learner mnist-forest))

;; 2.495 seconds (4 core), Accuracy: 90.97
(time
 (train-refine-learner-process mnist-refine-learner mnist-refine-dataset mnist-target
                               mnist-refine-test mnist-target-test))

(test-refine-learner mnist-refine-learner mnist-refine-test mnist-target-test)

;; more training
(loop repeat 10 do
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
 :n-tree 100 :bagging-ratio 0.1 :max-depth 10 :n-trial 28 :gamma 10.0 :min-region-samples 5)
