;; -*- coding:utf-8; mode:lisp -*-

;; set dynamic-space-size >= 2500

(in-package :cl-random-forest)

;;; Load Dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; MNIST data
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(defparameter mnist-dim 784)
(defparameter mnist-n-class 10)

(let ((mnist-train (clol.utils:read-data "/home/wiz/datasets/mnist.scale" mnist-dim :multiclass-p t)))
  ;; Add 1 to labels in order to form class-labels beginning from 0
  (dolist (datum mnist-train) (incf (car datum)))
  (multiple-value-bind (datamat target)
      (clol-dataset->datamatrix/target mnist-train)
    (defparameter mnist-datamatrix datamat)
    (defparameter mnist-target target)))


;; Enable parallelization
(setf lparallel:*kernel* (lparallel:make-kernel 4))

;; Note that SAVE-PARENT-NODE? keyword option is true
(defparameter mnist-forest
  (make-forest mnist-n-class mnist-datamatrix mnist-target
               :n-tree 1000 :bagging-ratio 0.1 :max-depth 30 :n-trial 28 :min-region-samples 5
               :save-parent-node? t))

;; Execute reconstruction
(defparameter *reconstruction*
  (reconstruction-forest mnist-forest mnist-datamatrix 0))

;; Plot
(ql:quickload :clgplot)

(defparameter *original-image*
  (let ((arr (make-array '(28 28))))
    (loop for i from 0 below 28 do
      (loop for j from 0 below 28 do
        (setf (aref arr i j)
              (aref mnist-datamatrix 0 (+ (* i 28) j)))))
    arr))

(defparameter *reconstruction-image*
  (let ((arr (make-array '(28 28))))
    (loop for i from 0 below 28 do
      (loop for j from 0 below 28 do
        (setf (aref arr i j)
              (aref *reconstruction* (+ (* i 28) j)))))
    arr))

(clgp:splot-matrix *original-image* :palette :greys)
(clgp:splot-matrix *reconstruction-image* :palette :greys)

;; Encode/Decode a datum

(defparameter index-datum (encode-datum mnist-forest mnist-datamatrix 0))
(decode-datum mnist-forest index-datum)

;; Encode/Decode dataset

(defparameter encoded-dataset (make-refine-dataset mnist-forest mnist-datamatrix))
(defparameter leaf-node-vector (make-leaf-node-vector mnist-forest))
(loop repeat 10
      for image-i from 0
      for index-datum across encoded-dataset do
        (let* ((reconstruction (decode-datum mnist-forest index-datum leaf-node-vector))
               (original-image
                 (let ((arr (make-array '(28 28))))
                   (loop for i from 0 below 28 do
                     (loop for j from 0 below 28 do
                       (setf (aref arr i j)
                             (aref mnist-datamatrix image-i (+ (* i 28) j)))))
                   arr))
               (reconstruction-image
                 (let ((arr (make-array '(28 28))))
                   (loop for i from 0 below 28 do
                     (loop for j from 0 below 28 do
                       (setf (aref arr i j)
                             (aref reconstruction (+ (* i 28) j)))))
                   arr)))
          (clgp:splot-matrix
           original-image
           :palette :greys
           :output (format nil "/home/wiz/tmp/rf-reconstruction-orig-~3,'0d.png" image-i))
          (clgp:splot-matrix
           reconstruction-image
           :palette :greys
           :output (format nil "/home/wiz/tmp/rf-reconstruction-recn-~3,'0d.png" image-i))))

#|
mogrify -rotate 90 rf-reconstruction-*.png
mogrify -crop '329x329+85+156' rf-reconstruction-*.png
mogrify -filter box -resize 25% rf-reconstruction-*.png
convert +append rf-reconstruction-orig-*.png out.png
convert +append rf-reconstruction-recn-*.png out2.png
convert -append out.png out2.png out3.png
mv out3.png reconstruction-ntree1000-bagging0_1-depth30-ntrial28.png
|#
