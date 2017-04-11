;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; letter 

(defparameter letter-dim 16)
(defparameter letter-n-class 26)

(let ((letter-train (clol.utils:read-data "/home/wiz/datasets/letter.scale" letter-dim :multiclass-p t))
      (letter-test (clol.utils:read-data "/home/wiz/datasets/letter.scale.t" letter-dim :multiclass-p t)))

  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target letter-train)
    (defparameter letter-datamatrix datamat)
    (defparameter letter-target target))

  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target letter-test)
    (defparameter letter-datamatrix-test datamat)
    (defparameter letter-target-test target)))

;; dtree
(defparameter letter-dtree
  (make-dtree letter-n-class letter-dim letter-datamatrix letter-target :max-depth 10))
(test-dtree letter-dtree letter-datamatrix-test letter-target-test)

;; random-forest
(defparameter letter-forest
  (make-forest letter-n-class letter-dim letter-datamatrix letter-target
               :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 10 :max-depth 15))
(test-forest letter-forest letter-datamatrix-test letter-target-test)
;; max-depth=5: 73.22% / max-depth=10: 89.03% / max-depth=15: 91.4%

(defparameter letter-forest-tall
  (make-forest letter-n-class letter-dim letter-datamatrix letter-target
               :n-tree 100 :bagging-ratio 1.0 :min-region-samples 5 :n-trial 10 :max-depth 100))
(test-forest letter-forest-tall letter-datamatrix-test letter-target-test) ; 96.82%

(defparameter letter-refine-dataset
  (make-refine-dataset letter-forest letter-datamatrix))

(defparameter letter-refine-test
  (make-refine-dataset letter-forest letter-datamatrix-test))

(defparameter letter-refine-learner (make-refine-learner letter-forest))

(train-refine-learner-process letter-refine-learner
                              letter-refine-dataset letter-target
                              letter-refine-test letter-target-test)

(test-refine-learner  letter-refine-learner letter-refine-test letter-target-test)

;; max-depth=5: 95.880005% / max-depth=10: 97.259995% / max-depth=15: 97.34%
