;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;; usps

(defparameter usps-dim 256)
(defparameter usps-n-class 10)

(let ((usps-train (clol.utils:read-data "/home/wiz/datasets/usps" usps-dim :multiclass-p t))
      (usps-test (clol.utils:read-data "/home/wiz/datasets/usps.t" usps-dim :multiclass-p t)))
  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target usps-train)
    (defparameter usps-datamatrix datamat)
    (defparameter usps-target target))
  
  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target usps-test)
    (defparameter usps-datamatrix-test datamat)
    (defparameter usps-target-test target)))

;; dtree
(defparameter usps-dtree
  (make-dtree usps-n-class usps-datamatrix usps-target :max-depth 10))
(test-dtree usps-dtree usps-datamatrix-test usps-target-test)

;; random-forest
(time 
 (defparameter usps-forest
   (make-forest usps-n-class usps-datamatrix usps-target
                :n-tree 500 :bagging-ratio 0.1 :min-region-samples 5 :n-trial 16 :max-depth 15)))
(test-forest usps-forest usps-datamatrix-test usps-target-test)
;; max-depth=5: 73.22% / max-depth=10: 89.03% / max-depth=15: 91.4%

(defparameter usps-forest-tall
  (make-forest usps-n-class usps-datamatrix usps-target
               :n-tree 100 :bagging-ratio 1.0 :min-region-samples 5 :n-trial 10 :max-depth 100))
(test-forest usps-forest-tall usps-datamatrix-test usps-target-test) ; 93.82%

(time
 (defparameter usps-refine-dataset
   (make-refine-dataset usps-forest usps-datamatrix)))

(time
 (defparameter usps-refine-test
   (make-refine-dataset usps-forest usps-datamatrix-test)))

(defparameter usps-refine-learner (make-refine-learner usps-forest))

(time (train-refine-learner-process
       usps-refine-learner usps-refine-dataset usps-target
       usps-refine-test usps-target-test))

(test-refine-learner usps-refine-learner usps-refine-test usps-target-test)

(loop repeat 10 do
  (train-refine-learner usps-refine-learner usps-refine-dataset usps-target)
  (test-refine-learner  usps-refine-learner usps-refine-test usps-target-test))
