;; set dynamic-space-size >= 2500

(in-package :cl-random-forest)

;;; MNIST
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(defparameter mnist-dim 780)

;; load dataset
(let ((mnist-train (clol.utils:read-data "/home/wiz/tmp/mnist.scale" mnist-dim :multiclass-p t))
      (mnist-test (clol.utils:read-data "/home/wiz/tmp/mnist.scale.t" mnist-dim :multiclass-p t)))

  ;; Add 1 to labels in order to form class-labels begin from 0
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

;;; Make Random Forest

;; ;;; Enable parallelizaion
;; (setf lparallel:*kernel* (lparallel:make-kernel 4))

;; 2.19 sec
(defparameter mnist-forest
  (make-forest 10 780 mnist-datamatrix mnist-target
               :n-tree 500 :bagging-ratio 0.1
               :max-depth 10 :n-trial 10 :min-region-samples 5))

;; Global Refinement of Random Forest

;; 18.952 seconds
(defparameter mnist-refine-dataset
  (make-refine-dataset mnist-forest mnist-datamatrix mnist-target))

;; 3.286 seconds
(defparameter mnist-refine-test
  (make-refine-dataset mnist-forest mnist-datamatrix-test mnist-target-test))

(defparameter mnist-refine-learner (make-refine-learner mnist-forest 1.0d0))

;; 23.396 seconds
(loop repeat 5 do
  (clol:train mnist-refine-learner mnist-refine-dataset)
  (clol:test  mnist-refine-learner mnist-refine-test))
;; Accuracy: 98.29%

;; Make a prediction
(predict-refine-learner mnist-forest mnist-refine-learner mnist-datamatrix-test 0)
