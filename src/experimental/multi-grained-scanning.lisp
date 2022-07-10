(in-package :cl-random-forest)

(defun set-one-patch (datamatrix datum-index patch-datamatrix patch-index
                      anchor datum-shape patch-shape)
  (loop for i from 0 below (car patch-shape)
        do (loop for j from 0 below (cadr patch-shape)
                 do (setf (aref patch-datamatrix patch-index (+ (* i (cadr patch-shape)) j))
                          (aref datamatrix datum-index (+ (* (+ (car anchor) i)
                                                             (cadr datum-shape))
                                                          (+ (cadr anchor) j)))))))

(defun number-of-patches (datum-shape patch-shape stride)
  (apply #'* (mapcar (lambda (datum-size patch-size)
                       (1+ (/ (- datum-size patch-size) stride)))
                     datum-shape patch-shape)))

(defun make-patch-datamatrix (datamatrix datum-shape patch-shape stride)
  (assert (= (length datum-shape) (length patch-shape)))
  (assert (every (lambda (datum-size patch-size)
                   (and (> datum-size patch-size)
                        (zerop (mod (- datum-size patch-size) stride))))
                 datum-shape patch-shape))
  (let* ((patch-size (apply #'* patch-shape))
         (n-patches (number-of-patches datum-shape patch-shape stride))
         (patch-datamatrix (make-array (list (* n-patches (array-dimension datamatrix 0))
                                             patch-size)
                                       :element-type 'single-float
                                       :initial-element 0d0))
         (patch-index 0))
    (loop for datum-index fixnum from 0 below (array-dimension datamatrix 0)
          do (loop for anchor-i fixnum
                   from 0
                     to (- (car datum-shape) (car patch-shape))
                   by stride
                   do (loop for anchor-j fixnum
                            from 0
                              to (- (cadr datum-shape) (cadr patch-shape))
                            by stride
                            do (set-one-patch datamatrix
                                              datum-index
                                              patch-datamatrix
                                              patch-index
                                              (list anchor-i anchor-j)
                                              datum-shape
                                              patch-shape)
                               (incf patch-index))))
    patch-datamatrix))

(defun make-patch-target (target datum-shape patch-shape stride)
  (let* ((n-patches (number-of-patches datum-shape patch-shape stride))
         (patch-target (make-array (* n-patches (length target)) :element-type 'fixnum)))
    (loop for i from 0 below (length target)
          do (loop for j from 0 below n-patches
                   do (setf (aref patch-target (+ (* i n-patches) j))
                            (aref target i))))
    patch-target))

;; (defparameter *datamatrix* (make-array `(1 ,(* 3 5)) :element-type 'single-float))
;; (dotimes (j (* 3 5))
;;   (setf (aref *datamatrix* 0 j) (* 1d0 j)))

;; (defparameter *patch-datamatrix* (make-array '(3 2) :element-type 'single-float))

;; (set-one-patch *datamatrix* 0 *patch-datamatrix* 0 '(0 0) '(3 5) '(2 1))
;; (set-one-patch *datamatrix* 0 *patch-datamatrix* 1 '(1 0) '(3 5) '(2 1))
;; (set-one-patch *datamatrix* 0 *patch-datamatrix* 2 '(0 1) '(3 5) '(2 1))

;; ;; #2A((0.0d0 5.0d0) (5.0d0 10.0d0) (1.0d0 6.0d0))

;; (defparameter *patch-datamatrix2* (make-patch-datamatrix *datamatrix* '(3 5) '(2 1) 1))

;; ;; #2A((0.0d0 5.0d0)
;; ;;     (1.0d0 6.0d0)
;; ;;     (2.0d0 7.0d0)
;; ;;     (3.0d0 8.0d0)
;; ;;     (4.0d0 9.0d0)
;; ;;     (5.0d0 10.0d0)
;; ;;     (6.0d0 11.0d0)
;; ;;     (7.0d0 12.0d0)
;; ;;     (8.0d0 13.0d0)
;; ;;     (9.0d0 14.0d0))

;; (defparameter *datamatrix* (make-array `(2 ,(* 3 5)) :element-type 'single-float))
;; (defparameter *target* #(1 2))
;; (dotimes (j (* 3 5))
;;   (setf (aref *datamatrix* 0 j) (* 1d0 j)))
;; (dotimes (j (* 3 5))
;;   (setf (aref *datamatrix* 1 j) (* 2d0 j)))

;; (defparameter *patch-datamatrix3* (make-patch-datamatrix *datamatrix* '(3 5) '(2 1) 1))
;; (defparameter *patch-target3* (make-patch-target *target* '(3 5) '(2 1) 1))

;; (defparameter *datamatrix* (make-array '(1 81) :element-type 'single-float))
;; (dotimes (j 81)
;;   (setf (aref *datamatrix* 0 j) (* 1d0 j)))

;; (defparameter *patch-datamatrix3* (make-patch-datamatrix *datamatrix* '(9 9) '(3 3) 2))

;; ;; #2A((0.0d0 1.0d0 2.0d0 9.0d0 10.0d0 11.0d0 18.0d0 19.0d0 20.0d0)
;; ;;     (2.0d0 3.0d0 4.0d0 11.0d0 12.0d0 13.0d0 20.0d0 21.0d0 22.0d0)
;; ;;     (4.0d0 5.0d0 6.0d0 13.0d0 14.0d0 15.0d0 22.0d0 23.0d0 24.0d0)
;; ;;     (6.0d0 7.0d0 8.0d0 15.0d0 16.0d0 17.0d0 24.0d0 25.0d0 26.0d0)
;; ;;     (18.0d0 19.0d0 20.0d0 27.0d0 28.0d0 29.0d0 36.0d0 37.0d0 38.0d0)
;; ;;     (20.0d0 21.0d0 22.0d0 29.0d0 30.0d0 31.0d0 38.0d0 39.0d0 40.0d0)
;; ;;     (22.0d0 23.0d0 24.0d0 31.0d0 32.0d0 33.0d0 40.0d0 41.0d0 42.0d0)
;; ;;     (24.0d0 25.0d0 26.0d0 33.0d0 34.0d0 35.0d0 42.0d0 43.0d0 44.0d0)
;; ;;     (36.0d0 37.0d0 38.0d0 45.0d0 46.0d0 47.0d0 54.0d0 55.0d0 56.0d0)
;; ;;     (38.0d0 39.0d0 40.0d0 47.0d0 48.0d0 49.0d0 56.0d0 57.0d0 58.0d0)
;; ;;     (40.0d0 41.0d0 42.0d0 49.0d0 50.0d0 51.0d0 58.0d0 59.0d0 60.0d0)
;; ;;     (42.0d0 43.0d0 44.0d0 51.0d0 52.0d0 53.0d0 60.0d0 61.0d0 62.0d0)
;; ;;     (54.0d0 55.0d0 56.0d0 63.0d0 64.0d0 65.0d0 72.0d0 73.0d0 74.0d0)
;; ;;     (56.0d0 57.0d0 58.0d0 65.0d0 66.0d0 67.0d0 74.0d0 75.0d0 76.0d0)
;; ;;     (58.0d0 59.0d0 60.0d0 67.0d0 68.0d0 69.0d0 76.0d0 77.0d0 78.0d0)
;; ;;     (60.0d0 61.0d0 62.0d0 69.0d0 70.0d0 71.0d0 78.0d0 79.0d0 80.0d0))

;; ;; 8.264 sec
;; (time
;;  (defparameter *mnist-patch-datamatrix*
;;    (make-patch-datamatrix mnist-datamatrix '(28 28) '(16 16) 4)))

;; ;; 1.495 sec
;; (time
;;  (defparameter *mnist-patch-datamatrix-test*
;;    (make-patch-datamatrix mnist-datamatrix-test '(28 28) '(16 16) 4)))

;; (ql:quickload :clgplot)

;; (defun split-patch-matrix (patch-datamatrix patch-shape patch-index)
;;   (let ((patch-matrix (make-array patch-shape :element-type 'single-float)))
;;     (loop for i from 0 below (car patch-shape)
;;           do (loop for j from 0 below (cadr patch-shape)
;;                    do (setf (aref patch-matrix i j)
;;                             (aref patch-datamatrix patch-index (+ (* i (car patch-shape)) j)))))
;;     (clgp:splot-matrix patch-matrix)))

;; (loop for i from 0 to 16 do
;;   (split-patch-matrix *mnist-patch-datamatrix* '(16 16) i))

;; (defparameter *mnist-patch-target* (make-patch-target mnist-target '(28 28) '(16 16) 4))
;; (defparameter *mnist-patch-target-test* (make-patch-target mnist-target-test '(28 28) '(16 16) 4))

;; (setf lparallel:*kernel* (lparallel:make-kernel 4))

;; ;; 107.582 sec (n-parallel: 4)
;; (defparameter *mnist-patch-forest*
;;   (make-forest mnist-n-class *mnist-patch-datamatrix* *mnist-patch-target*
;;                :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 10 :min-region-samples 5))

;; ;; 16.555 sec (n-parallel: 4)
;; (defparameter *mnist-patch-cr-forest*
;;   (make-forest mnist-n-class *mnist-patch-datamatrix* *mnist-patch-target*
;;                :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 1 :min-region-samples 5))

;; (defun make-class-distribution-datamatrix
;;     (feature-extraction-forests datasize patch-datamatrix &key (print-progress? t))
;;   (let ((n-patches-per-datum (/ (array-dimension patch-datamatrix 0) datasize))
;;         (n-forests (length feature-extraction-forests))
;;         (n-class (forest-n-class (car feature-extraction-forests))))
;;     (assert (every (lambda (forest)
;;                      (= (forest-n-class forest) n-class))
;;                    (cdr feature-extraction-forests)))
;;     (let ((class-distribution-datamatrix
;;             (make-array (list datasize (* n-patches-per-datum n-class n-forests))
;;                         :element-type 'single-float
;;                         :initial-element 0d0)))
;;       (loop for datum-index from 0 below datasize
;;             do (when (and print-progress? (zerop (mod (1+ datum-index) (floor datasize 100))))
;;                  (format t ".")
;;                  (force-output))
;;                (loop for forest-index from 0 below n-forests
;;                      for forest in feature-extraction-forests
;;                      do (loop for patch-index from 0 below n-patches-per-datum
;;                               do (loop for class-index from 0 below n-class
;;                                        for class-confidence
;;                                          across (class-distribution-forest
;;                                                  forest
;;                                                  patch-datamatrix
;;                                                  (+ (* datum-index n-patches-per-datum)
;;                                                     patch-index))
;;                                        do (setf (aref class-distribution-datamatrix
;;                                                       datum-index
;;                                                       (+ (* forest-index (* n-class
;;                                                                             n-patches-per-datum))
;;                                                          (* class-index n-patches-per-datum)
;;                                                          patch-index))
;;                                                 class-confidence)))))
;;       class-distribution-datamatrix)))

;; intractable
;; (defparameter *mnist-class-distribution-datamatrix*
;;   (make-class-distribution-datamatrix (list mnist-forest mnist-cr-forest)
;;                                       (array-dimension mnist-datamatrix 0)
;;                                       *mnist-patch-datamatrix*))

;; mnist-patch-datamatrixに対してglobal-refinementをやってみる

;; 
(defun make-refine-dataset-from-patch-datamatrix (forest datamatrix patch-datamatrix)
  (let ((index-offset (forest-index-offset forest))
        (len (array-dimension datamatrix 0))
        (n-tree (forest-n-tree forest))
        (n-patch (/ (array-dimension patch-datamatrix 0)
                    (array-dimension datamatrix 0))))
    (declare (optimize (speed 3) (safety 0))
             (type (simple-array single-float) datamatrix patch-datamatrix)
             (type (simple-array fixnum) index-offset)
             (type fixnum len n-tree n-patch))
    (let ((refine-dataset (make-array len)))
      (loop for i from 0 below len do
        (setf (aref refine-dataset i) (make-array (* n-tree n-patch) :element-type 'fixnum)))
      (mapc/pmapc
       (lambda (dtree)
         (let* ((tree-id (dtree-id dtree))
                (offset (aref index-offset tree-id)))
           (declare (type fixnum tree-id offset))
           (loop for i fixnum from 0 below len do
             (loop for patch-index fixnum from 0 below n-patch do
               (let ((leaf-index (node-leaf-index (find-leaf (dtree-root dtree)
                                                             patch-datamatrix
                                                             (+ (* i n-patch) patch-index))))
                     (refine-datum (svref refine-dataset i)))
                 (declare (type fixnum leaf-index)
                          (type (simple-array fixnum) refine-datum))
                 (setf (aref refine-datum (+ (* patch-index n-tree) tree-id))
                       (+ leaf-index offset))))))
         (format t ".")
         (force-output))
       (forest-dtree-list forest))
      (terpri)
      refine-dataset)))

;; (defun make-refine-dataset-from-patch-datamatrix-and-multiple-forests
;;     (forests datamatrix patch-datamatrix)
;;   (let ((index-offset (forest-index-offset forest))
;;         (len (array-dimension datamatrix 0))
;;         (n-tree (forest-n-tree forest))
;;         (n-patch (/ (array-dimension patch-datamatrix 0)
;;                     (array-dimension datamatrix 0))))
;;     (declare (optimize (speed 3) (safety 0))
;;              (type (simple-array single-float) datamatrix patch-datamatrix)
;;              (type (simple-array fixnum) index-offset)
;;              (type fixnum len n-tree n-patch))
;;     (let ((refine-dataset (make-array len)))
;;       (loop for i from 0 below len do
;;         (setf (aref refine-dataset i) (make-array (* n-tree n-patch) :element-type 'fixnum)))
;;       (mapc/pmapc
;;        (lambda (dtree)
;;          (let* ((tree-id (dtree-id dtree))
;;                 (offset (aref index-offset tree-id)))
;;            (declare (type fixnum tree-id offset))
;;            (loop for i fixnum from 0 below len do
;;              (loop for patch-index fixnum from 0 below n-patch do
;;                (let ((leaf-index (node-leaf-index (find-leaf (dtree-root dtree)
;;                                                              patch-datamatrix
;;                                                              (+ (* i n-patch) patch-index))))
;;                      (refine-datum (svref refine-dataset i)))
;;                  (declare (type fixnum leaf-index)
;;                           (type (simple-array fixnum) refine-datum))
;;                  (setf (aref refine-datum (+ (* patch-index n-tree) tree-id))
;;                        (+ leaf-index offset))))))
;;          (format t ".")
;;          (force-output))
;;        (forest-dtree-list forest))
;;       refine-dataset)))

;; ;; 44.419 sec (n-parallel: 4)
;; (defparameter *mnist-refine-dataset*
;;   (make-refine-dataset-from-patch-datamatrix
;;    *mnist-patch-forest* mnist-datamatrix *mnist-patch-datamatrix*))

;; ;; 8.985 sec (n-parallel: 4)
;; (defparameter *mnist-refine-test*
;;   (make-refine-dataset-from-patch-datamatrix
;;    *mnist-patch-forest* mnist-datamatrix-test *mnist-patch-datamatrix-test*))

(defun make-data-augmented-refine-learner (forest n-patch &optional (gamma 10d0))
  (let ((n-class (forest-n-class forest))
        (input-dim (* n-patch
                      (loop for n-leaves
                              in (mapcar #'dtree-max-leaf-index (forest-dtree-list forest))
                            sum n-leaves))))
    (if (> n-class 2)
        (clol:make-one-vs-rest input-dim n-class 'sparse-arow gamma)
        (clol:make-sparse-arow input-dim gamma))))

;; (defparameter mnist-refine-learner (make-data-augmented-refine-learner *mnist-patch-forest* 16))

;; ;; 72.755 sec (n-parallel: 4)
;; (train-refine-learner-process mnist-refine-learner
;;                               *mnist-refine-dataset* mnist-target
;;                               *mnist-refine-test* mnist-target-test)

;; (test-refine-learner mnist-refine-learner *mnist-refine-test* mnist-target-test)

;; ;; Accuracy: 98.79%, Correct: 9879, Total: 10000
;; ;; (make-patch-datamatrix mnist-datamatrix '(28 28) '(16 16) 4)

;; mnist-datamatrix-test

;; (predict-refine-learner mnist-forest mnist-refine-learner *mnist-patch-datamatrix-test* 0)


;; redefine make-l2-norm-multiclass for patched refine-learner

(defparameter *n-patch* 16)

(defun make-l2-norm-multiclass (learner)
  (let* ((dim (/ (clol::one-vs-rest-input-dimension learner)
                 *n-patch*))
         (arr (make-array dim :element-type 'single-float :initial-element 0d0)))
    (loop for i from 0 below (clol::one-vs-rest-n-class learner) do
      (let* ((sub-learner (svref (clol::one-vs-rest-learners-vector learner) i))
             (weight-vec (funcall (clol::one-vs-rest-learner-weight learner) sub-learner)))
        (loop for j from 0 below dim do
          (loop for k from 0 below *n-patch* do
            (setf (aref arr j)
                  (+ (aref arr j)
                     (square (aref weight-vec (+ (* dim k) j)))))))))
    arr))
