;; https://www.kaggle.com/heptapod/titanic

(ql:quickload '(:fare-csv
                :parse-float
                :alexandria))

(in-package :clrf)

(defparameter csv (fare-csv:read-csv-file "/home/wiz/Downloads/train_and_test2.csv"))
(defparameter csv-header (car csv))
(defparameter csv-body
  (remove nil
          (mapcar (lambda (row)
                    (handler-case
                        (mapcar #'parse-float:parse-float row)
                      (error (c)
                        (format t "Error: ~A, Ignored row: ~A~%" c row))))
                  (cdr csv))))

;; shuffle order
(setf csv-body (alexandria:shuffle csv-body))

(defun split-target (data)
  (let ((two-part-list
          (mapcar (lambda (row)
                    (cons (subseq row 0 (1- (length row)))
                          (alexandria:lastcar row)))
                  data)))
    (values (mapcar #'car two-part-list)
            (mapcar #'cdr two-part-list))))

(multiple-value-bind (all-x all-y)
    (split-target csv-body)
  (let ((num-train-dataset 1000)
        (dim (length (car all-x)))
        (len (length all-x))
        (all-y (mapcar #'floor all-y)))
    (defparameter x (make-array (list num-train-dataset dim)
                                :element-type 'single-float
                                :initial-contents (subseq all-x 0 num-train-dataset)))
    (defparameter test-x (make-array (list (- len num-train-dataset) dim)
                                     :element-type 'single-float
                                     :initial-contents (subseq all-x num-train-dataset)))
    (defparameter y
      (make-array num-train-dataset
                  :element-type 'fixnum
                  :initial-contents (subseq all-y 0 num-train-dataset)))
    (defparameter test-y
      (make-array (- len num-train-dataset)
                  :element-type 'fixnum
                  :initial-contents (subseq all-y num-train-dataset)))))

;; train

(defparameter forest (clrf:make-forest 2 x y :n-tree 100 :max-depth 5 :bagging-ratio 0.9))

;; predict

(clrf:test-forest forest test-x test-y)
;; Accuracy: 88.27361%, Correct: 271, Total: 307

;; train refine-learner
(defparameter refine-x (clrf:make-refine-dataset forest x))
(defparameter refine-test-x (clrf:make-refine-dataset forest test-x))
(defparameter refine-learner (clrf:make-refine-learner forest))
(clrf:train-refine-learner-process refine-learner refine-x y refine-test-x test-y)

;; test refine-learner
(clrf:test-refine-learner refine-learner refine-test-x test-y)

;; train AROW

(ql:quickload :cl-debug-print)
(cl-syntax:use-syntax cl-debug-print:debug-print-syntax)

(defun datamatrix->clol-dataset (x y)
  (loop for i from 0 below (array-dimension x 0)
        collect
        (cons (if (> (aref y i) 0) 1.0 -1.0)
              (make-array (array-dimension x 1)
                          :element-type 'single-float
                          :initial-contents (loop for j
                                                  from 0 below (array-dimension x 1)
                                                  collect (aref x i j))))))

(defparameter clol-dataset (datamatrix->clol-dataset x y))
(defparameter clol-test-dataset (datamatrix->clol-dataset test-x test-y))

(defparameter arow-learner (clol:make-arow 27 10d0))
(clol:train arow-learner clol-dataset)
(clol:test arow-learner clol-test-dataset)
