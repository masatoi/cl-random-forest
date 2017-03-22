;; set dynamic-space-size >= 2500

(in-package :cl-random-forest)

;;; Load Dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; MNIST data
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(defparameter mnist-dim 780)
(defparameter mnist-n-class 10)

(let ((mnist-train (clol.utils:read-data "/home/wiz/tmp/mnist.scale" mnist-dim :multiclass-p t))
      (mnist-test (clol.utils:read-data "/home/wiz/tmp/mnist.scale.t" mnist-dim :multiclass-p t)))

  ;; Add 1 to labels in order to form class-labels beginning from 0
  (dolist (datum mnist-train) (incf (car datum)))
  (dolist (datum mnist-test)  (incf (car datum)))

  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target mnist-train)
    (defparameter mnist-datamatrix datamat)
    (defparameter mnist-target target))
  
  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target mnist-test)
    (defparameter mnist-datamatrix-test datamat)
    (defparameter mnist-target-test target)))

;;; Make Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Enable/Disable parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
(setf lparallel:*kernel* nil)

;; 6.238 seconds (1 core), 2.195 seconds (4 core)
(defparameter mnist-forest
  (make-forest mnist-n-class mnist-dim mnist-datamatrix mnist-target
               :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 10 :min-region-samples 5))

;; 5.223 seconds, Accuracy: 93.38%
(test-forest mnist-forest mnist-datamatrix-test mnist-target-test)

;; 42.717 seconds (1 core), 13.24 seconds (4 core)
(defparameter mnist-forest-tall
  (make-forest mnist-n-class mnist-dim mnist-datamatrix mnist-target
               :n-tree 100 :bagging-ratio 1.0 :max-depth 100 :n-trial 27 :min-region-samples 5))

;; 14.2 seconds, Accuracy: 96.62%
(test-forest mnist-forest-tall mnist-datamatrix-test mnist-target-test)

;;; Global Refinement of Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Generate sparse data from Random Forest

;; 20.250 seconds (1 core), 8.366 seconds (4 core)
(defparameter mnist-refine-dataset
  (make-refine-dataset mnist-forest mnist-datamatrix mnist-target))

;; 3.421 seconds (1 core), 1.444 seconds (4 core)
(defparameter mnist-refine-test
  (make-refine-dataset mnist-forest mnist-datamatrix-test mnist-target-test))

(defparameter mnist-refine-learner (make-refine-learner mnist-forest))

;; 11.046 seconds, Accuracy: 98.29%
(loop repeat 5 do
  (clol:train mnist-refine-learner mnist-refine-dataset)
  (clol:test  mnist-refine-learner mnist-refine-test))

;; In case of without making dataset
(time (train-refine-learner mnist-forest mnist-refine-learner mnist-datamatrix mnist-target))
(time (test-refine-learner mnist-forest mnist-refine-learner mnist-datamatrix-test mnist-target-test))

;; 1.880 seconds
(time
 (defparameter mnist-leaf-index-matrix
   (make-leaf-index-matrix mnist-forest mnist-datamatrix)))

(train-refine-learner-fast mnist-refine-learner mnist-leaf-index-matrix mnist-target)

(time
 (defparameter mnist-leaf-indices-vector
   (make-leaf-indices-vector mnist-forest mnist-datamatrix)))

(time
 (defparameter mnist-leaf-indices-vector-test
   (make-leaf-indices-vector mnist-forest mnist-datamatrix-test)))

(time
 (train-refine-learner-parallel mnist-refine-learner mnist-leaf-indices-vector mnist-target))

(time
 (test-refine-learner-fast mnist-refine-learner mnist-leaf-indices-vector-test mnist-target-test))

(time
 (test-refine-learner-parallel mnist-refine-learner mnist-leaf-indices-vector-test mnist-target-test
                               :mini-batch-size 1000))

(loop repeat 10 do
  (train-refine-learner-parallel mnist-refine-learner mnist-leaf-indices-vector mnist-target)
  (format t "train: ")
  (test-refine-learner-fast mnist-refine-learner mnist-leaf-indices-vector mnist-target)
  (format t "test: ")
  (test-refine-learner-fast mnist-refine-learner mnist-leaf-indices-vector-test mnist-target-test))

;; Make a prediction
(predict-refine-learner mnist-forest mnist-refine-learner mnist-datamatrix-test 0)

;;; Global Prunning of Random Forest ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(length (collect-leaf-parent mnist-forest)) ; => 98008
(pruning! mnist-forest mnist-refine-learner 0.1) ; 0.328 seconds
(length (collect-leaf-parent mnist-forest)) ; => 93228

;; Re-learning refine learner
(defparameter mnist-refine-dataset (make-refine-dataset mnist-forest mnist-datamatrix mnist-target))
(defparameter mnist-refine-test (make-refine-dataset mnist-forest mnist-datamatrix-test mnist-target-test))
(defparameter mnist-refine-learner (make-refine-learner mnist-forest))
(loop repeat 5 do
  (clol:train mnist-refine-learner mnist-refine-dataset)
  (clol:test  mnist-refine-learner mnist-refine-test))
;; Accuracy: Accuracy: 98.27%

;; (loop repeat 10 do
;;   (sb-ext:gc :full t)
;;   (room)

;;   (format t "~%making mnist-refine-dataset~%")
;;   (defparameter mnist-refine-dataset (make-refine-dataset mnist-forest mnist-datamatrix mnist-target))

;;   (format t "~%making mnist-refine-test~%")
;;   (defparameter mnist-refine-test (make-refine-dataset mnist-forest mnist-datamatrix-test mnist-target-test))

;;   (format t "~%re-learning~%")
;;   (defparameter mnist-refine-learner (make-refine-learner mnist-forest))
;;   (loop repeat 5 do
;;     (clol:train mnist-refine-learner mnist-refine-dataset)
;;     (clol:test  mnist-refine-learner mnist-refine-test))
  
;;   (format t "~%Pruning... ~%")
;;   (pruning! mnist-forest mnist-refine-learner 0.01)

;;   (format t "leaf-size: ~A ~%" (length (collect-leaf-parent mnist-forest))))
