;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-random-forest)

;;; Small dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter *n-class* 4)

(defparameter *target*
  (make-array 11 :element-type 'fixnum
                 :initial-contents '(0 0 1 1 2 2 2 3 3 3 3)))

(defparameter *datamatrix*
  (make-array '(11 2) 
              :element-type 'double-float
              :initial-contents '((-1.0d0 -2.0d0)
                                  (-2.0d0 -1.0d0)
                                  (1.0d0 -2.0d0)
                                  (3.0d0 -1.5d0)
                                  (-2.0d0 2.0d0)
                                  (-3.0d0 1.0d0)
                                  (-2.0d0 1.0d0)
                                  (3.0d0 2.0d0)
                                  (2.0d0 2.0d0)
                                  (1.0d0 2.0d0)
                                  (1.0d0 1.0d0))))

;;; Decision tree ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
;; make decision tree
(defparameter *dtree*
  (make-dtree *n-class* *datamatrix* *target*
              :max-depth 5 :min-region-samples 1 :n-trial 10))

;; prediction
(predict-dtree *dtree* *datamatrix* 0)

(defun predict-node (node)
  (let ((max 0d0)
        (max-class 0)
        (dist (node-class-distribution node))
        (n-class (dtree-n-class (node-dtree node))))
    (loop for i fixnum from 0 to (1- n-class) do
      (when (> (aref dist i) max)
        (setf max (aref dist i)
              max-class i)))
    max-class))

(defun extract-node (node)
  (if (and node (node-test-attribute node))
      `(if (>= (aref d i ,(node-test-attribute node)) ,(node-test-threshold node))
           ,(extract-node (node-left-node node))
           ,(extract-node (node-right-node node)))
      (predict-node node)))

(defun construct-dtree-lambda (dtree)
  `(lambda (d i)
     (declare (optimize (speed 3) (space 0) (safety 0) (debug 0) (compilation-speed 0))
              (type (simple-array double-float) d)
              (type fixnum i))
     ,(extract-node (dtree-root dtree))))

(construct-dtree-lambda *dtree*)

;; 生成されるlambda式
(LAMBDA (DATAMATRIX DATUM-INDEX)
  (DECLARE
   (OPTIMIZE (SPEED 3) (SPACE 0) (SAFETY 0) (DEBUG 0) (COMPILATION-SPEED 0))
   (TYPE (SIMPLE-ARRAY DOUBLE-FLOAT) DATAMATRIX)
   (TYPE FIXNUM DATUM-INDEX))
  (IF (>= (AREF DATAMATRIX DATUM-INDEX 0) -0.7394168078526362d0)
      (IF (>= (AREF DATAMATRIX DATUM-INDEX 1) 0.8903535809681147d0)
          3
          1)
      (IF (>= (AREF DATAMATRIX DATUM-INDEX 1) 0.5876648784761986d0)
          2
          0)))

;; コンパイル
(defparameter compiled-dtree (compile nil (construct-dtree-lambda *dtree*)))

;;呼び出し
(funcall compiled-dtree *datamatrix* 0)

;;;;;

(defparameter mnist-dim 784)
(defparameter mnist-n-class 10)

(let ((mnist-train (clol.utils:read-data "/mnt/data2/datasets/mnist.scale" mnist-dim :multiclass-p t))
      (mnist-test (clol.utils:read-data "/mnt/data2/datasets/mnist.scale.t" mnist-dim :multiclass-p t)))

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

;;; Make Decision Tree ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter mnist-dtree
  (make-dtree mnist-n-class mnist-datamatrix mnist-target
              :max-depth 10 :n-trial 28 :min-region-samples 5))

(test-dtree mnist-dtree mnist-datamatrix mnist-target)
(test-dtree mnist-dtree mnist-datamatrix-test mnist-target-test)

(time
 (loop repeat 100 do
   (loop for i from 0 below (array-dimension mnist-datamatrix 0) do
     (predict-dtree mnist-dtree mnist-datamatrix i))))

(time (defparameter dtree-predictor (compile nil (construct-dtree-lambda mnist-dtree))))
(time (defparameter dtree-lambda (construct-dtree-lambda mnist-dtree))) ; これは非常に高速

(time
 (loop repeat 100 do
   (loop for i from 0 below (array-dimension mnist-datamatrix 0) do
     (funcall dtree-predictor mnist-datamatrix i))))

(defparameter mnist-forest
  (make-forest mnist-n-class mnist-datamatrix mnist-target
               :n-tree 500 :bagging-ratio 0.1 :max-depth 5 :n-trial 28 :min-region-samples 5))

(time
 (defparameter dtree-predictor-list
   (loop for dtree in (forest-dtree-list mnist-forest)
         for i from 0
         collect (progn
                   (print i)
                   (compile nil (construct-dtree-lambda dtree))))))

(defun argmax (arr)
  (let ((max 0)
        (max-i 0))
    (loop for i from 0 below (length arr) do
      (when (> (aref arr i) max)
        (setf max (aref arr i)
              max-i i)))
    max-i))

(defun predict-dtree-predictor-list (dtree-predictor-list datamatrix index)
  (let ((cnt (make-array (array-dimension datamatrix 1))))
    (loop for predictor in dtree-predictor-list do
      (incf (aref cnt (funcall predictor datamatrix index))))
    (argmax cnt)))

(predict-dtree-predictor-list dtree-predictor-list mnist-datamatrix 0)

(defun test-dtree-predictor-list (dtree-predictor-list datamatrix target)
  (loop for i from 0 below (array-dimension datamatrix 0)
        count (= (predict-dtree-predictor-list dtree-predictor-list datamatrix i)
                 (aref target i))))

(time (test-dtree-predictor-list dtree-predictor-list mnist-datamatrix-test mnist-target-test))
;; 9385
;; Evaluation took:
;;   0.286 seconds of real time
;;   0.284000 seconds of total run time (0.284000 user, 0.000000 system)
;;   99.30% CPU
;;   967,442,117 processor cycles
;;   62,876,912 bytes consed

(time (test-forest mnist-forest mnist-datamatrix-test mnist-target-test))
;; Accuracy: 94.33%, Correct: 9433, Total: 10000
;; Evaluation took:
;;   2.659 seconds of real time
;;   2.660000 seconds of total run time (2.660000 user, 0.000000 system)
;;   100.04% CPU
;;   9,021,268,236 processor cycles
;;   1,216 bytes consed

;; 事前に全ての葉の予測値を出しておく方式(多数決 majority-vote)
;; predict-all-leaf

(defun argmax (arr)
  (declare (optimize (speed 3) (safety 0))
           (type (simple-array double-float) arr))
  (let ((max 0d0)
        (max-i 0))
    (declare (type double-float max)
             (type fixnum max-i))
    (loop for i fixnum from 0 below (length arr) do
      (when (> (aref arr i) max)
        (setf max (aref arr i)
              max-i i)))
    max-i))

(defparameter leaf1 (find-leaf (dtree-root mnist-dtree) mnist-datamatrix 0))
(argmax (node-class-distribution leaf1))

(defun set-leaf-prediction (dtree)
  (do-leaf (lambda (node)
             (setf (node-leaf-prediction node)
                   (argmax (node-class-distribution node))))
    (dtree-root dtree)))

(time (set-leaf-prediction mnist-dtree))


(defun majority-voting (dtree data
