(in-package :clrf)

;;; Small dataset ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter *dataset*
  '((0 . (-1d0 -2d0))
    (0 . (-2d0 -1d0))
    (1 . (1d0 -2d0))
    (1 . (3d0 -1.5d0))
    (2 . (-2d0 2d0))
    (2 . (-3d0 1d0))
    (2 . (-2d0 1d0))
    (3 . (3d0 2d0))
    (3 . (2d0 2d0))
    (3 . (1d0 2d0))
    (3 . (1d0 1d0))))

(defparameter *target* (make-array (length *dataset*) :element-type 'fixnum
                                   :initial-contents (mapcar #'car *dataset*)))

(defparameter *datamatrix* (make-array (list (length *dataset*) 2) :element-type 'double-float))
(loop for i from 0 below (length *dataset*)
      for elem in *dataset*
      do
   (loop for j from 0 to 1 do
     (setf (aref *datamatrix* i j)
           (nth j (cdr elem)))))

(defparameter *dtree* (make-dtree 4 2 *datamatrix* *target* :gain-test #'gini))

(region-min/max (make-array (array-dimension *datamatrix* 0) :element-type 'fixnum
                            :initial-contents (alexandria:iota (array-dimension *datamatrix* 0)))
                *datamatrix* 0)

(defparameter sample-indices1
  (make-array 10 :element-type 'fixnum :initial-contents '(0 1 2 3 4 5 6 7 8 9)))

(class-distribution sample-indices1 5 *dtree*)
(entropy sample-indices1 (length sample-indices1) *dtree*)

(traverse #'node-information-gain (dtree-root *dtree*))
(traverse #'node-sample-indices (dtree-root *dtree*))

(do-leaf (lambda (node)
           (format t "~%leaf-index: ~A, sample-indices: ~A"
                   (node-leaf-index node)
                   (node-sample-indices node)))
  (dtree-root *dtree*))

(set-leaf-index! *dtree*)

(dtree-max-leaf-index *dtree*)

(traverse #'node-leaf-index (dtree-root *dtree*))

(test-dtree *dtree* *datamatrix* *target*)

(defparameter *forest* (make-forest 4 2 *datamatrix* *target* :n-tree 3 :bagging-ratio 0.5))
(traverse #'node-sample-indices (dtree-root (car (forest-dtree-list *forest*))))

(test-forest *forest* *datamatrix* *target*)

(set-leaf-index-forest! *forest*)

(loop for i from 0 to (1- (array-dimension *datamatrix* 0)) do
  (print (clol.vector:sparse-vector-index-vector
          (make-forest-sparse-vector *forest* *datamatrix* i))))

(defparameter *forest-learner* (make-forest-learner *forest*))

(train-forest-learner! *forest* *forest-learner* *datamatrix* *target*)
(test-forest-learner *forest* *forest-learner* *datamatrix* *target*)

;;; MNIST ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(defparameter mnist-dim 780)

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
  
(time (defparameter mnist-dtree (make-dtree 10 780 mnist-datamatrix mnist-target :max-depth 10 :n-trial 27)))

;; (write-to-r-format-from-clol-dataset mnist-train "/home/wiz/datasets/mnist-for-R")

;; Evaluation took:
;;   2.329 seconds of real time
;;   2.332534 seconds of total run time (2.332534 user, 0.000000 system)
;;   [ Run times consist of 0.027 seconds GC time, and 2.306 seconds non-GC time. ]
;;   100.17% CPU
;;   7,902,772,046 processor cycles
;;   833,405,296 bytes consed

;; Evaluation took:
;;   1.913 seconds of real time
;;   1.913793 seconds of total run time (1.889773 user, 0.024020 system)
;;   [ Run times consist of 0.022 seconds GC time, and 1.892 seconds non-GC time. ]
;;   100.05% CPU
;;   6,488,816,835 processor cycles
;;   834,031,648 bytes consed

;; split-list => split-arr
;; Evaluation took:
;;   2.162 seconds of real time
;;   2.170621 seconds of total run time (2.170621 user, 0.000000 system)
;;   100.42% CPU
;;   7,333,558,776 processor cycles
;;   568,714,880 bytes consed

;; with datamatrix and target
;; Evaluation took:
;;   1.427 seconds of real time
;;   1.428196 seconds of total run time (1.421912 user, 0.006284 system)
;;   [ Run times consist of 0.011 seconds GC time, and 1.418 seconds non-GC time. ]
;;   100.07% CPU
;;   4,841,932,737 processor cycles
;;   566,070,512 bytes consed

;; split-arr => split-sample-indices
;; Evaluation took:
;;   1.157 seconds of real time
;;   1.159157 seconds of total run time (1.159157 user, 0.000000 system)
;;   [ Run times consist of 0.008 seconds GC time, and 1.152 seconds non-GC time. ]
;;   100.17% CPU
;;   3,925,551,873 processor cycles
;;   533,164,544 bytes consed

;; make-random-test exploit fixed array
;; Evaluation took:
;;   0.587 seconds of real time
;;   0.586155 seconds of total run time (0.578620 user, 0.007535 system)
;;   99.83% CPU
;;   1,988,323,468 processor cycles
;;   16,330,528 bytes consed

;; Pick 2-points in make-random-test (less accuracy)
;; Evaluation took:
;;   0.486 seconds of real time
;;   0.486908 seconds of total run time (0.486908 user, 0.000000 system)
;;   100.21% CPU
;;   1,648,894,569 processor cycles
;;   14,178,160 bytes consed

;; safety 0
;; Evaluation took:
;;   0.273 seconds of real time
;;   0.273069 seconds of total run time (0.269585 user, 0.003484 system)
;;   [ Run times consist of 0.010 seconds GC time, and 0.264 seconds non-GC time. ]
;;   100.00% CPU
;;   925,629,119 processor cycles
;;   16,496,000 bytes consed

;; Pick 2-points + safety 0
;; Evaluation took:
;;   0.215 seconds of real time
;;   0.215951 seconds of total run time (0.215951 user, 0.000000 system)
;;   100.47% CPU
;;   731,896,506 processor cycles
;;   13,927,600 bytes consed

(require 'sb-sprof)
(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (defparameter mnist-dtree (make-dtree 10 780 mnist-datamatrix mnist-target :max-depth 5 :n-trial 270)))

(time (test-dtree mnist-dtree mnist-datamatrix mnist-target))
(time (test-dtree mnist-dtree mnist-datamatrix-test mnist-target-test))
;; => 75.89d0

(time
 (loop repeat 10 do
   (print (test-dtree (make-dtree 10 780 mnist-datamatrix mnist-target :max-depth 10 :n-trial 27)
                      mnist-datamatrix mnist-target))))

;; numbers of datapoints each node has
(traverse (lambda (node) (length (node-sample-indices node)))
          (dtree-root mnist-dtree))

;; prediction
(ql:quickload :clgplot)
(clgp:plot (node-class-distribution
            (find-leaf (dtree-root mnist-dtree) mnist-datamatrix-test 0))
           :style 'impulse)
(predict-dtree mnist-dtree mnist-datamatrix-test 0)

;;; forest

;; 2.19 sec
(time (defparameter mnist-forest
        (make-forest 10 780 mnist-datamatrix mnist-target
                     :n-tree 500 :bagging-ratio 0.1
                     :max-depth 10 :n-trial 10 :min-region-samples 5)))

(time (print (test-forest mnist-forest mnist-datamatrix mnist-target)))
(time (print (test-forest mnist-forest mnist-datamatrix-test mnist-target-test)))

;; 18.952 seconds
(time (defparameter mnist-refine-dataset
        (make-refine-dataset mnist-forest mnist-datamatrix mnist-target)))

;; 3.286 seconds
(time (defparameter mnist-refine-test
        (make-refine-dataset mnist-forest mnist-datamatrix-test mnist-target-test)))

(defparameter mnist-refine-learner (make-refine-learner mnist-forest 1.0d0))
(defparameter mnist-refine-learner (make-refine-learner-scw mnist-forest 0.999d0 0.01d0))

;; 23.396 seconds
(time
 (loop repeat 10 do
   (clol:train mnist-refine-learner mnist-refine-dataset)
   (clol:test  mnist-refine-learner mnist-refine-test)))

(clol:test  mnist-refine-learner mnist-refine-dataset)
(predict-refine-learner mnist-forest mnist-refine-learner mnist-datamatrix-test 0)

;; Evaluation took:
;;   22.525 seconds of real time
;;   22.539637 seconds of total run time (22.458771 user, 0.080866 system)
;;   [ Run times consist of 0.557 seconds GC time, and 21.983 seconds non-GC time. ]
;;   100.07% CPU
;;   76,407,726,956 processor cycles
;;   11,103,020,544 bytes consed

;; Evaluation took:
;;   3.439 seconds of real time
;;   3.438304 seconds of total run time (3.376220 user, 0.062084 system)
;;   [ Run times consist of 0.074 seconds GC time, and 3.365 seconds non-GC time. ]
;;   99.97% CPU
;;   11,667,948,885 processor cycles
;;   731,683,488 bytes consed

;;; Parallelizaion
(setf lparallel:*kernel* (lparallel:make-kernel 4))
;; (setf lparallel:*kernel* nil)

(time (defparameter mnist-forest
        (make-forest 10 780 mnist-datamatrix mnist-target
                     :n-tree 100 :bagging-ratio 0.1
                     :max-depth 10 :n-trial 27)))

;; Evaluation took:
;;   1.095 seconds of real time
;;   4.126016 seconds of total run time (4.036967 user, 0.089049 system)
;;   [ Run times consist of 0.064 seconds GC time, and 4.063 seconds non-GC time. ]
;;   376.80% CPU
;;   3,714,638,160 processor cycles
;;   720,102,160 bytes consed

;; prediction
(clgp:plot (class-distribution-forest mnist-forest mnist-datamatrix-test 0) :style 'impulse)
(predict-forest mnist-forest mnist-datamatrix-test 0)
(time (print (test-forest mnist-forest mnist-datamatrix mnist-target)))
(time (print (test-forest mnist-forest mnist-datamatrix-test mnist-target-test)))

(time (defparameter mnist-forest
        (make-forest 10 780 mnist-datamatrix mnist-target
                     :n-tree 500 :bagging-ratio 1.0
                     :max-depth 10 :n-trial 27)))

;; train
;; 53.616 seconds

;; predict
;; train 99.885d0   58.639 seconds
;; test  96.65d0    9.791 seconds

(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (time (print (test-forest mnist-forest mnist-datamatrix mnist-target))))

;; => 93.43d0
;; Evaluation took:
;;   1.912 seconds of real time
;;   1.914804 seconds of total run time (1.914804 user, 0.000000 system)
;;   [ Run times consist of 0.017 seconds GC time, and 1.898 seconds non-GC time. ]
;;   100.16% CPU
;;   6,486,260,337 processor cycles
;;   803,371,520 bytes consed

;; Evaluation took:
;;   0.744 seconds of real time
;;   0.744223 seconds of total run time (0.744223 user, 0.000000 system)
;;   100.00% CPU
;;   2,524,311,970 processor cycles
;;   72,800 bytes consed

(mapcar (lambda (dtree)
          (test-dtree dtree mnist-datamatrix-test mnist-target-test))
        (forest-dtree-list mnist-forest))

;; (66.99000000000001d0 70.59d0 69.96d0 71.12d0 70.08d0 69.89d0 69.51d0 66.9d0
;;  69.57d0 68.49d0 70.16d0 69.1d0 69.6d0 67.7d0 70.23d0 70.98d0 66.88d0 69.16d0
;;  71.21d0 68.38d0 69.06d0 69.89999999999999d0 69.87d0 69.82000000000001d0
;;  69.84d0 70.17999999999999d0 70.34d0 66.95d0 70.83d0 68.22d0 69.23d0 68.16d0
;;  68.63d0 71.04d0 68.19d0 67.86d0 68.04d0 68.56d0 69.21000000000001d0 67.97d0
;;  69.04d0 69.39d0 67.85d0 69.57d0 69.78999999999999d0 70.17999999999999d0
;;  67.58d0 68.89d0 67.89d0 71.57d0 69.35d0 68.02d0 68.41000000000001d0 68.2d0
;;  71.14d0 70.88d0 70.42d0 70.73d0 68.73d0 67.57d0 70.58d0 69.67999999999999d0
;;  70.71d0 70.57d0 70.98d0 69.12d0 69.25d0 67.95d0 70.38d0 70.58d0 70.44d0
;;  67.92d0 69.53d0 70.59d0 68.02d0 68.83d0 70.08d0 69.28999999999999d0 71.81d0
;;  70.8d0 69.67999999999999d0 69.35d0 71.95d0 68.19d0 69.42d0 68.60000000000001d0
;;  68.96d0 67.73d0 70.75d0 70.14d0 68.16d0 69.58d0 70.41d0 70.03d0
;;  68.08999999999999d0 66.79d0 72.23d0 68.05d0 68.05d0 72.14d0)

;;;;;;;;;;;;; mushrooms

;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms

(defparameter *mushrooms-dim* 112)
(defparameter *mushrooms-train* (clol.utils:read-data "/home/wiz/datasets/mushrooms-train" *mushrooms-dim*))
(defparameter *mushrooms-test* (clol.utils:read-data "/home/wiz/datasets/mushrooms-test" *mushrooms-dim*))

(dolist (datum *mushrooms-train*)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(dolist (datum *mushrooms-test*)
  (if (> (car datum) 0d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix-and-target *mushrooms-train*)
  (defparameter mushrooms-datamatrix datamat)
  (defparameter mushrooms-target target))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix-and-target *mushrooms-test*)
  (defparameter mushrooms-datamatrix-test datamat)
  (defparameter mushrooms-target-test target))



(write-to-r-format-from-clol-dataset *mushrooms-train* "/home/wiz/datasets/mushrooms-for-R")
(write-to-r-format-from-clol-dataset *mushrooms-test* "/home/wiz/datasets/mushrooms-for-R.t")

;; decision tree

(defparameter mushrooms-dtree
  (make-dtree 2 *mushrooms-dim* mushrooms-datamatrix mushrooms-target))
(traverse #'node-information-gain (dtree-root mushrooms-dtree))
(test-dtree mushrooms-dtree mushrooms-datamatrix-test mushrooms-target-test)
;; => 98.21092278719398d0

(time
 (loop repeat 100 do
   (defparameter mushrooms-forest
     (make-forest 2 *mushrooms-dim* mushrooms-datamatrix mushrooms-target
                  :n-tree 100 :bagging-ratio 0.1 :n-trial 10 :max-depth 10))))

;; Evaluation took:
;;   1.901 seconds of real time
;;   1.902504 seconds of total run time (1.866545 user, 0.035959 system)
;;   [ Run times consist of 0.084 seconds GC time, and 1.819 seconds non-GC time. ]
;;   100.11% CPU
;;   6,448,467,269 processor cycles
;;   1,171,116,816 bytes consed

;; Evaluation took:
;;   0.447 seconds of real time
;;   0.447074 seconds of total run time (0.378886 user, 0.068188 system)
;;   [ Run times consist of 0.157 seconds GC time, and 0.291 seconds non-GC time. ]
;;   100.00% CPU
;;   1,515,547,477 processor cycles
;;   137,457,680 bytes consed

;; Evaluation took:
;;   0.018 seconds of real time
;;   0.059598 seconds of total run time (0.059598 user, 0.000000 system)
;;   333.33% CPU
;;   60,858,259 processor cycles
;;   28,202,128 bytes consed

(test-forest mushrooms-forest mushrooms-datamatrix mushrooms-target)
(test-forest mushrooms-forest mushrooms-datamatrix-test mushrooms-target-test)

;;;;;;;;;;;; covtype

(defparameter covtype-dim 54)
(defparameter covtype-train (clol.utils:read-data "/home/wiz/datasets/covtype.libsvm.binary.scale" covtype-dim))

(dolist (datum covtype-train)
  (if (> (car datum) 1d0)
      (setf (car datum) 0)
      (setf (car datum) 1)))

(multiple-value-bind (datamat target)
    (clol-dataset->datamatrix-and-target covtype-train)
  (defparameter covtype-datamatrix datamat)
  (defparameter covtype-target target))

;; (setf lparallel:*kernel* (lparallel:make-kernel 4))

(time
 (defparameter covtype-forest
   (make-forest 2 covtype-dim covtype-datamatrix covtype-target
                :n-tree 100 :bagging-ratio 0.1 :n-trial 10 :max-depth 10)))

(time (print (test-forest covtype-forest covtype-datamatrix covtype-target)))
(time (print (predict-forest covtype-forest covtype-datamatrix 0)))

;;;;;;;;;;;;;;;;;;;;

;; CL-RF> (sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
;;   (defparameter mnist-forest (make-forest 10 780 mnist-train.vec :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 27)))
;; Profiler sample vector full (290 traces / 10000 samples), doubling the size
;; Profiler sample vector full (579 traces / 20000 samples), doubling the size

;; Number of samples:   1000
;; Sample interval:     0.01 seconds
;; Total sampling time: 10.0 seconds
;; Number of cycles:    0
;; Sampled threads:
;;  #<SB-THREAD:THREAD "repl-thread" RUNNING {1003017FA3}>

;;            Self        Total        Cumul
;;   Nr  Count     %  Count     %  Count     %    Calls  Function
;; ------------------------------------------------------------------------
;;    1    238  23.8    238  23.8    238  23.8        -  RUN-TEST
;;    2    172  17.2    172  17.2    410  41.0        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF DOUBLE-FLOAT)
;;    3    145  14.5    427  42.7    555  55.5        -  MAKE-RANDOM-TEST
;;    4     65   6.5     88   8.8    620  62.0        -  CLASS-DISTRIBUTION
;;    5     49   4.9    347  34.7    669  66.9        -  SPLIT-LIST
;;    6     40   4.0     40   4.0    709  70.9        -  SB-KERNEL:TWO-ARG-<
;;    7     36   3.6     36   3.6    745  74.5        -  SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS
;;    8     31   3.1     52   5.2    776  77.6        -  (LAMBDA (INDEX) :IN SET-BEST-CHILDREN!)
;;    9     31   3.1     31   3.1    807  80.7        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF T)
;;   10     28   2.8     28   2.8    835  83.5        -  SB-KERNEL:TWO-ARG-+
;;   11     21   2.1     21   2.1    856  85.6        -  SB-KERNEL:TWO-ARG->
;;   12     20   2.0     20   2.0    876  87.6        -  LENGTH
;;   13     15   1.5     77   7.7    891  89.1        -  MIN/MAX
;;   14     12   1.2     72   7.2    903  90.3        -  SB-VM::GENERIC-+
;;   15      9   0.9      9   0.9    912  91.2        -  "foreign function pthread_sigmask"
;;   16      9   0.9      9   0.9    921  92.1        -  SB-KERNEL:TWO-ARG-*
;;   17      6   0.6    165  16.5    927  92.7        -  ENTROPY
;;   18      6   0.6     10   1.0    933  93.3        -  SB-KERNEL::INTEGER-/-INTEGER
;;   19      6   0.6      6   0.6    939  93.9        -  %MAKE-NODE
;;   20      6   0.6      6   0.6    945  94.5        -  SB-KERNEL:%COERCE-CALLABLE-TO-FUN
;;   21      5   0.5    997  99.7    950  95.0        -  SET-BEST-CHILDREN!
;;   22      5   0.5      5   0.5    955  95.5        -  SB-KERNEL:TWO-ARG-GCD
;;   23      3   0.3      3   0.3    958  95.8        -  LOG
;;   24      3   0.3      3   0.3    961  96.1        -  RANDOM
;;   25      2   0.2      7   0.7    963  96.3        -  SB-KERNEL:%DOUBLE-FLOAT
;;   26      2   0.2      2   0.2    965  96.5        -  MAKE-NODE
;;   27      1   0.1      2   0.2    966  96.6        -  SB-KERNEL::FLOAT-RATIO
;;   28      1   0.1      1   0.1    967  96.7        -  SCALE-FLOAT
;;   29      1   0.1      1   0.1    968  96.8        -  SB-KERNEL:TWO-ARG-/
;;   30      1   0.1      1   0.1    969  96.9        -  SB-KERNEL:TWO-ARG--
;;   31      1   0.1      1   0.1    970  97.0        -  ASH
;;   32      1   0.1      1   0.1    971  97.1        -  (LABELS SB-KERNEL::FLOAT-AND-SCALE :IN SB-KERNEL::FLOAT-RATIO)
;;   33      0   0.0   1000 100.0    971  97.1        -  SPLIT-NODE!
;;   34      0   0.0   1000 100.0    971  97.1        -  MAKE-DTREE
;;   35      0   0.0   1000 100.0    971  97.1        -  MAKE-FOREST
;;   36      0   0.0   1000 100.0    971  97.1        -  "Unknown component: #x102282D860"
;;   37      0   0.0   1000 100.0    971  97.1        -  SB-INT:SIMPLE-EVAL-IN-LEXENV
;;   38      0   0.0   1000 100.0    971  97.1        -  EVAL
;;   39      0   0.0   1000 100.0    971  97.1        -  SWANK::EVAL-REGION
;;   40      0   0.0   1000 100.0    971  97.1        -  (LAMBDA NIL :IN SWANK-REPL::REPL-EVAL)
;;   41      0   0.0   1000 100.0    971  97.1        -  SWANK-REPL::TRACK-PACKAGE
;;   42      0   0.0   1000 100.0    971  97.1        -  SWANK::CALL-WITH-RETRY-RESTART
;;   43      0   0.0   1000 100.0    971  97.1        -  SWANK::CALL-WITH-BUFFER-SYNTAX
;;   44      0   0.0   1000 100.0    971  97.1        -  SWANK-REPL::REPL-EVAL
;;   45      0   0.0   1000 100.0    971  97.1        -  SWANK:EVAL-FOR-EMACS
;;   46      0   0.0   1000 100.0    971  97.1        -  SWANK::PROCESS-REQUESTS
;;   47      0   0.0   1000 100.0    971  97.1        -  (LAMBDA NIL :IN SWANK::HANDLE-REQUESTS)
;;   48      0   0.0   1000 100.0    971  97.1        -  SWANK/SBCL::CALL-WITH-BREAK-HOOK
;;   49      0   0.0   1000 100.0    971  97.1        -  (FLET SWANK/BACKEND:CALL-WITH-DEBUGGER-HOOK :IN "/home/wiz/.roswell/lisp/quicklisp/dists/quicklisp/software/slime-v2.18/swank/sbcl.lisp")
;;   50      0   0.0   1000 100.0    971  97.1        -  SWANK::CALL-WITH-BINDINGS
;;   51      0   0.0   1000 100.0    971  97.1        -  SWANK::HANDLE-REQUESTS
;;   52      0   0.0   1000 100.0    971  97.1        -  (FLET #:WITHOUT-INTERRUPTS-BODY-1139 :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   53      0   0.0   1000 100.0    971  97.1        -  (FLET SB-THREAD::WITH-MUTEX-THUNK :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   54      0   0.0   1000 100.0    971  97.1        -  (FLET #:WITHOUT-INTERRUPTS-BODY-359 :IN SB-THREAD::CALL-WITH-MUTEX)
;;   55      0   0.0   1000 100.0    971  97.1        -  SB-THREAD::CALL-WITH-MUTEX
;;   56      0   0.0   1000 100.0    971  97.1        -  SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE
;;   57      0   0.0   1000 100.0    971  97.1        -  "foreign function call_into_lisp"
;;   58      0   0.0   1000 100.0    971  97.1        -  "foreign function new_thread_trampoline"
;;   59      0   0.0      9   0.9    971  97.1        -  "foreign function interrupt_handle_pending"
;;   60      0   0.0      9   0.9    971  97.1        -  "foreign function handle_trap"
;;   61      0   0.0      3   0.3    971  97.1        -  RANDOM-UNIFORM
;;   62      0   0.0      2   0.2    971  97.1        -  STOP-SPLIT?
;; ------------------------------------------------------------------------
;;          29   2.9                                     elsewhere
;; #<SB-SPROF::CALL-GRAPH 1000 samples {10756D72D3}>
;; CL-RF> (sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
;;   (test-forest mnist-forest mnist-test.vec))
;; Profiler sample vector full (321 traces / 10000 samples), doubling the size
;; Profiler sample vector full (640 traces / 20000 samples), doubling the size

;; Number of samples:   1000
;; Sample interval:     0.01 seconds
;; Total sampling time: 10.0 seconds
;; Number of cycles:    0
;; Sampled threads:
;;  #<SB-THREAD:THREAD "repl-thread" RUNNING {1003017FA3}>

;;            Self        Total        Cumul
;;   Nr  Count     %  Count     %  Count     %    Calls  Function
;; ------------------------------------------------------------------------
;;    1    405  40.5    510  51.0    405  40.5        -  CLASS-DISTRIBUTION
;;    2    232  23.2    396  39.6    637  63.7        -  FIND-LEAF
;;    3    167  16.7    167  16.7    804  80.4        -  RUN-TEST
;;    4     51   5.1     51   5.1    855  85.5        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF T)
;;    5     46   4.6     46   4.6    901  90.1        -  SB-KERNEL:TWO-ARG-+
;;    6     32   3.2    995  99.5    933  93.3        -  CLASS-DISTRIBUTION-FOREST
;;    7     28   2.8     28   2.8    961  96.1        -  SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS
;;    8     13   1.3     13   1.3    974  97.4        -  SB-VM::GENERIC-+
;;    9     13   1.3     13   1.3    987  98.7        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF DOUBLE-FLOAT)
;;   10      6   0.6      6   0.6    993  99.3        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-SET DOUBLE-FLOAT)
;;   11      4   0.4      4   0.4    997  99.7        -  SB-KERNEL:HAIRY-DATA-VECTOR-SET/CHECK-BOUNDS
;;   12      3   0.3      3   0.3   1000 100.0        -  "foreign function pthread_sigmask"
;;   13      0   0.0   1000 100.0   1000 100.0        -  PREDICT-FOREST
;;   14      0   0.0   1000 100.0   1000 100.0        -  TEST-FOREST
;;   15      0   0.0   1000 100.0   1000 100.0        -  "Unknown component: #x100B97BB40"
;;   16      0   0.0   1000 100.0   1000 100.0        -  SB-INT:SIMPLE-EVAL-IN-LEXENV
;;   17      0   0.0   1000 100.0   1000 100.0        -  EVAL
;;   18      0   0.0   1000 100.0   1000 100.0        -  SWANK::EVAL-REGION
;;   19      0   0.0   1000 100.0   1000 100.0        -  (LAMBDA NIL :IN SWANK-REPL::REPL-EVAL)
;;   20      0   0.0   1000 100.0   1000 100.0        -  SWANK-REPL::TRACK-PACKAGE
;;   21      0   0.0   1000 100.0   1000 100.0        -  SWANK::CALL-WITH-RETRY-RESTART
;;   22      0   0.0   1000 100.0   1000 100.0        -  SWANK::CALL-WITH-BUFFER-SYNTAX
;;   23      0   0.0   1000 100.0   1000 100.0        -  SWANK-REPL::REPL-EVAL
;;   24      0   0.0   1000 100.0   1000 100.0        -  SWANK:EVAL-FOR-EMACS
;;   25      0   0.0   1000 100.0   1000 100.0        -  SWANK::PROCESS-REQUESTS
;;   26      0   0.0   1000 100.0   1000 100.0        -  (LAMBDA NIL :IN SWANK::HANDLE-REQUESTS)
;;   27      0   0.0   1000 100.0   1000 100.0        -  SWANK/SBCL::CALL-WITH-BREAK-HOOK
;;   28      0   0.0   1000 100.0   1000 100.0        -  (FLET SWANK/BACKEND:CALL-WITH-DEBUGGER-HOOK :IN "/home/wiz/.roswell/lisp/quicklisp/dists/quicklisp/software/slime-v2.18/swank/sbcl.lisp")
;;   29      0   0.0   1000 100.0   1000 100.0        -  SWANK::CALL-WITH-BINDINGS
;;   30      0   0.0   1000 100.0   1000 100.0        -  SWANK::HANDLE-REQUESTS
;;   31      0   0.0   1000 100.0   1000 100.0        -  (FLET #:WITHOUT-INTERRUPTS-BODY-1139 :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   32      0   0.0   1000 100.0   1000 100.0        -  (FLET SB-THREAD::WITH-MUTEX-THUNK :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   33      0   0.0   1000 100.0   1000 100.0        -  (FLET #:WITHOUT-INTERRUPTS-BODY-359 :IN SB-THREAD::CALL-WITH-MUTEX)
;;   34      0   0.0   1000 100.0   1000 100.0        -  SB-THREAD::CALL-WITH-MUTEX
;;   35      0   0.0   1000 100.0   1000 100.0        -  SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE
;;   36      0   0.0   1000 100.0   1000 100.0        -  "foreign function call_into_lisp"
;;   37      0   0.0   1000 100.0   1000 100.0        -  "foreign function new_thread_trampoline"
;;   38      0   0.0      3   0.3   1000 100.0        -  "foreign function interrupt_handle_pending"
;;   39      0   0.0      3   0.3   1000 100.0        -  "foreign function handle_trap"
;; ------------------------------------------------------------------------
;;           0   0.0                                     elsewhere


(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (defparameter mnist-forest (make-forest 10 780 mnist-train.vec :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 27)))

;; Evaluation took:
;;   111.685 seconds of real time
;;   111.747163 seconds of total run time (110.816033 user, 0.931130 system)
;;   [ Run times consist of 3.490 seconds GC time, and 108.258 seconds non-GC time. ]
;;   100.06% CPU
;;   378,849,965,134 processor cycles
;;   55,365,768,176 bytes consed

;; safety 0
;; Evaluation took:
;;   95.479 seconds of real time
;;   95.532739 seconds of total run time (94.856241 user, 0.676498 system)
;;   [ Run times consist of 3.140 seconds GC time, and 92.393 seconds non-GC time. ]
;;   100.06% CPU
;;   323,879,100,523 processor cycles
;;   55,397,321,376 bytes consed

;; Evaluation took:
;;   95.163 seconds of real time
;;   95.201786 seconds of total run time (93.868038 user, 1.333748 system)
;;   [ Run times consist of 2.354 seconds GC time, and 92.848 seconds non-GC time. ]
;;   100.04% CPU
;;   322,807,001,080 processor cycles
;;   39,925,680,864 bytes consed

(test-forest mnist-forest mnist-test.vec)

;;; compile message

; in: DEFUN CLASS-DISTRIBUTION
;     (AREF (CL-RANDOM-FOREST::DTREE-DATASET CL-RANDOM-FOREST::DTREE)
;           CL-RANDOM-FOREST::DATUM-INDEX)
; ==>
;   (SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS ARRAY SB-INT:INDEX)
; 
; note: unable to
;   optimize
; because:
;   Upgraded element type of array is not known at compile time.

; in: DEFUN MAKE-RANDOM-TEST
;     (RANDOM (CL-RANDOM-FOREST::DTREE-DATUM-DIM CL-RANDOM-FOREST::DTREE))
; 
; note: unable to
;   Use inline float operations.
; due to type uncertainty:
;   The first argument is a (OR (SINGLE-FLOAT (0.0)) (DOUBLE-FLOAT (0.0d0))
;                               (INTEGER 1)), not a SINGLE-FLOAT.
; 
; note: unable to
;   Use inline float operations.
; due to type uncertainty:
;   The first argument is a (OR (SINGLE-FLOAT (0.0)) (DOUBLE-FLOAT (0.0d0))
;                               (INTEGER 1)), not a DOUBLE-FLOAT.

;     (= MIN MAX)
; 
; note: unable to
;   open-code FLOAT to RATIONAL comparison
; due to type uncertainty:
;   The first argument is a NUMBER, not a FLOAT.
;   The second argument is a NUMBER, not a RATIONAL.
; 
; note: unable to open code because: The operands might not be the same type.

;     (LOOP CL-RANDOM-FOREST::FOR CL-RANDOM-FOREST::INDEX CL-RANDOM-FOREST::ACROSS CL-RANDOM-FOREST::INDICES
;           CL-RANDOM-FOREST::COLLECT (AREF
;                                      (CDR
;                                       (AREF
;                                        (CL-RANDOM-FOREST::DTREE-DATASET
;                                         CL-RANDOM-FOREST::DTREE)
;                                        CL-RANDOM-FOREST::INDEX))
;                                      CL-RANDOM-FOREST::ATTRIBUTE))
; --> BLOCK LET SB-LOOP::WITH-LOOP-LIST-COLLECTION-HEAD LET* 
; --> SB-LOOP::LOOP-BODY TAGBODY SB-LOOP::LOOP-REALLY-DESETQ SETQ THE 
; --> AREF 
; ==>
;   (SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS ARRAY SB-INT:INDEX)
; 
; note: unable to
;   optimize
; because:
;   Upgraded element type of array is not known at compile time.

;     (AREF (CL-RANDOM-FOREST::DTREE-DATASET CL-RANDOM-FOREST::DTREE)
;           CL-RANDOM-FOREST::INDEX)
; ==>
;   (SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS ARRAY SB-INT:INDEX)
; 
; note: unable to
;   optimize
; because:
;   Upgraded element type of array is not known at compile time.

;     (AREF
;      (CDR
;       (AREF (CL-RANDOM-FOREST::DTREE-DATASET CL-RANDOM-FOREST::DTREE)
;             CL-RANDOM-FOREST::INDEX))
;      CL-RANDOM-FOREST::ATTRIBUTE)
; ==>
;   (SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS ARRAY SB-INT:INDEX)
; 
; note: unable to
;   optimize
; because:
;   Upgraded element type of array is not known at compile time.

;     (= MIN MAX)
; 
; note: forced to do GENERIC-= (cost 10)
;       unable to do inline float comparison (cost 3) because:
;       The first argument is a T, not a DOUBLE-FLOAT.
;       The second argument is a T, not a (COMPLEX DOUBLE-FLOAT).
;       unable to do inline float comparison (cost 3) because:
;       The first argument is a T, not a (COMPLEX DOUBLE-FLOAT).
;       The second argument is a T, not a (COMPLEX DOUBLE-FLOAT).
;       etc.


;; Profiler sample vector full (289 traces / 10000 samples), doubling the size
;; Profiler sample vector full (577 traces / 20000 samples), doubling the size

;; Number of samples:   1000
;; Sample interval:     0.01 seconds
;; Total sampling time: 10.0 seconds
;; Number of cycles:    0
;; Sampled threads:
;;  #<SB-THREAD:THREAD "repl-thread" RUNNING {1002DAFFA3}>

;;            Self        Total        Cumul
;;   Nr  Count     %  Count     %  Count     %    Calls  Function
;; ------------------------------------------------------------------------
;;    1    262  26.2    262  26.2    262  26.2        -  RUN-TEST
;;    2    152  15.2    455  45.5    414  41.4        -  MAKE-RANDOM-TEST
;;    3    152  15.2    152  15.2    566  56.6        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF DOUBLE-FLOAT)
;;    4     48   4.8     48   4.8    614  61.4        -  SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS
;;    5     46   4.6     68   6.8    660  66.0        -  CLASS-DISTRIBUTION
;;    6     45   4.5     45   4.5    705  70.5        -  SB-KERNEL:TWO-ARG-<
;;    7     37   3.7    354  35.4    742  74.2        -  SPLIT-ARR
;;    8     33   3.3     33   3.3    775  77.5        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF T)
;;    9     30   3.0     30   3.0    805  80.5        -  SB-KERNEL:TWO-ARG->
;;   10     26   2.6     46   4.6    831  83.1        -  (FLET TEST-FUNC :IN SET-BEST-CHILDREN!)
;;   11     21   2.1    105  10.5    852  85.2        -  MIN/MAX
;;   12     20   2.0     97   9.7    872  87.2        -  SB-VM::GENERIC-+
;;   13     19   1.9    145  14.5    891  89.1        -  ENTROPY
;;   14     10   1.0     10   1.0    901  90.1        -  SB-KERNEL:TWO-ARG-+
;;   15     10   1.0     10   1.0    911  91.1        -  "foreign function pthread_sigmask"
;;   16      9   0.9      9   0.9    920  92.0        -  SB-KERNEL:%COERCE-CALLABLE-TO-FUN
;;   17      7   0.7     12   1.2    927  92.7        -  SB-KERNEL:TWO-ARG-*
;;   18      6   0.6     10   1.0    933  93.3        -  SB-KERNEL::INTEGER-/-INTEGER
;;   19      6   0.6      7   0.7    939  93.9        -  RANDOM
;;   20      5   0.5      5   0.5    944  94.4        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF FIXNUM)
;;   21      4   0.4    996  99.6    948  94.8        -  SET-BEST-CHILDREN!
;;   22      4   0.4      4   0.4    952  95.2        -  SB-KERNEL:TWO-ARG-GCD
;;   23      3   0.3      7   0.7    955  95.5        -  COPY-TMP->BEST!
;;   24      3   0.3      3   0.3    958  95.8        -  LOG
;;   25      2   0.2    999  99.9    960  96.0        -  MAKE-DTREE
;;   26      2   0.2      2   0.2    962  96.2        -  SB-KERNEL:TWO-ARG-=
;;   27      2   0.2      2   0.2    964  96.4        -  SB-KERNEL:FLOAT-FORMAT-DIGITS
;;   28      1   0.1      6   0.6    965  96.5        -  SB-KERNEL:%DOUBLE-FLOAT
;;   29      1   0.1      1   0.1    966  96.6        -  ASH
;;   30      1   0.1      1   0.1    967  96.7        -  MAKE-PARTIAL-ARR
;;   31      1   0.1      1   0.1    968  96.8        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-SET FIXNUM)
;;   32      1   0.1      1   0.1    969  96.9        -  TRUNCATE
;;   33      1   0.1      1   0.1    970  97.0        -  SB-KERNEL::RANDOM-MT19937-UPDATE
;;   34      0   0.0   1000 100.0    970  97.0        -  MAKE-FOREST
;;   35      0   0.0   1000 100.0    970  97.0        -  "Unknown component: #x1001EF8E80"
;;   36      0   0.0   1000 100.0    970  97.0        -  SB-INT:SIMPLE-EVAL-IN-LEXENV
;;   37      0   0.0   1000 100.0    970  97.0        -  EVAL
;;   38      0   0.0   1000 100.0    970  97.0        -  SWANK::EVAL-REGION
;;   39      0   0.0   1000 100.0    970  97.0        -  (LAMBDA NIL :IN SWANK-REPL::REPL-EVAL)
;;   40      0   0.0   1000 100.0    970  97.0        -  SWANK-REPL::TRACK-PACKAGE
;;   41      0   0.0   1000 100.0    970  97.0        -  SWANK::CALL-WITH-RETRY-RESTART
;;   42      0   0.0   1000 100.0    970  97.0        -  SWANK::CALL-WITH-BUFFER-SYNTAX
;;   43      0   0.0   1000 100.0    970  97.0        -  SWANK-REPL::REPL-EVAL
;;   44      0   0.0   1000 100.0    970  97.0        -  SWANK:EVAL-FOR-EMACS
;;   45      0   0.0   1000 100.0    970  97.0        -  SWANK::PROCESS-REQUESTS
;;   46      0   0.0   1000 100.0    970  97.0        -  (LAMBDA NIL :IN SWANK::HANDLE-REQUESTS)
;;   47      0   0.0   1000 100.0    970  97.0        -  SWANK/SBCL::CALL-WITH-BREAK-HOOK
;;   48      0   0.0   1000 100.0    970  97.0        -  (FLET SWANK/BACKEND:CALL-WITH-DEBUGGER-HOOK :IN "/home/wiz/.roswell/lisp/quicklisp/dists/quicklisp/software/slime-v2.18/swank/sbcl.lisp")
;;   49      0   0.0   1000 100.0    970  97.0        -  SWANK::CALL-WITH-BINDINGS
;;   50      0   0.0   1000 100.0    970  97.0        -  SWANK::HANDLE-REQUESTS
;;   51      0   0.0   1000 100.0    970  97.0        -  (FLET #:WITHOUT-INTERRUPTS-BODY-1139 :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   52      0   0.0   1000 100.0    970  97.0        -  (FLET SB-THREAD::WITH-MUTEX-THUNK :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   53      0   0.0   1000 100.0    970  97.0        -  (FLET #:WITHOUT-INTERRUPTS-BODY-359 :IN SB-THREAD::CALL-WITH-MUTEX)
;;   54      0   0.0   1000 100.0    970  97.0        -  SB-THREAD::CALL-WITH-MUTEX
;;   55      0   0.0   1000 100.0    970  97.0        -  SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE
;;   56      0   0.0   1000 100.0    970  97.0        -  "foreign function call_into_lisp"
;;   57      0   0.0   1000 100.0    970  97.0        -  "foreign function new_thread_trampoline"
;;   58      0   0.0    997  99.7    970  97.0        -  SPLIT-NODE!
;;   59      0   0.0     10   1.0    970  97.0        -  "foreign function interrupt_handle_pending"
;;   60      0   0.0     10   1.0    970  97.0        -  "foreign function handle_trap"
;;   61      0   0.0      4   0.4    970  97.0        -  SB-KERNEL::FLOAT-RATIO
;;   62      0   0.0      4   0.4    970  97.0        -  RANDOM-UNIFORM
;;   63      0   0.0      1   0.1    970  97.0        -  BOOTSTRAP-SAMPLE-INDICES
;; ------------------------------------------------------------------------
;;          30   3.0                                     elsewhere
