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

(defparameter *dataset2*
  (map 'vector (lambda (line)
                 (cons (car line) (make-array 2 :element-type 'double-float :initial-contents (cdr line))))
       *dataset*))

(defparameter *dtree* (make-dtree 4 2 *dataset2*))
(traverse #'node-information-gain (dtree-root *dtree*))
(traverse #'node-sample-indices (dtree-root *dtree*))

(test-dtree *dtree* *dataset2*)

(defparameter node2 (make-node '(2 3) (dtree-root *dtree*)))
(entropy node2)
(gini node2)

(defparameter *forest* (make-forest 4 2 *dataset2* :n-tree 1 :bagging-ratio 0.5))
(traverse #'node-sample-indices (dtree-root (car (forest-dtree-list *forest*))))

;;; MNIST ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(defparameter mnist-dim 780)
(defparameter mnist-train (clol.utils:read-data "/home/wiz/tmp/mnist.scale" mnist-dim :multiclass-p t))
(defparameter mnist-test  (clol.utils:read-data "/home/wiz/tmp/mnist.scale.t" mnist-dim :multiclass-p t))

;; Add 1 to labels in order to form class-labels begin from 0
(dolist (datum mnist-train) (incf (car datum)))
(dolist (datum mnist-test)  (incf (car datum)))

(defparameter mnist-train.vec (coerce mnist-train 'vector))
(defparameter mnist-test.vec (coerce mnist-test 'vector))

(time (defparameter mnist-dtree (make-dtree 10 780 mnist-train.vec :max-depth 10 :n-trial 27)))

;; Evaluation took:
;;   2.329 seconds of real time
;;   2.332534 seconds of total run time (2.332534 user, 0.000000 system)
;;   [ Run times consist of 0.027 seconds GC time, and 2.306 seconds non-GC time. ]
;;   100.17% CPU
;;   7,902,772,046 processor cycles
;;   833,405,296 bytes consed

(traverse #'node-information-gain (dtree-root mnist-dtree))
;;(traverse #'node-sample-indices (dtree-root mnist-dtree))

(test-dtree mnist-dtree mnist-test.vec)
;; => 75.89d0

;; numbers of datapoints each node has
(traverse (lambda (node) (length (node-sample-indices node)))
          (dtree-root mnist-dtree))

;; prediction
(ql:quickload :clgplot)
(clgp:plot (class-distribution (find-leaf (dtree-root mnist-dtree) (cdr (car mnist-test)))) :style 'impulse)
(predict-dtree mnist-dtree (cdr (car mnist-test)))

;;; forest

(time (defparameter mnist-forest
        (make-forest 10 780 mnist-train.vec
                     :n-tree 100 :bagging-ratio 0.1
                     :max-depth 10 :n-trial 27)))

;; Evaluation took:
;;   22.525 seconds of real time
;;   22.539637 seconds of total run time (22.458771 user, 0.080866 system)
;;   [ Run times consist of 0.557 seconds GC time, and 21.983 seconds non-GC time. ]
;;   100.07% CPU
;;   76,407,726,956 processor cycles
;;   11,103,020,544 bytes consed

;; prediction
(clgp:plot (class-distribution-forest mnist-forest (cdr (car mnist-test))) :style 'impulse)
(predict-forest mnist-forest (cdr (car mnist-test)))

(time (print (test-forest mnist-forest mnist-test.vec)))

;; => 93.43d0
;; Evaluation took:
;;   1.912 seconds of real time
;;   1.914804 seconds of total run time (1.914804 user, 0.000000 system)
;;   [ Run times consist of 0.017 seconds GC time, and 1.898 seconds non-GC time. ]
;;   100.16% CPU
;;   6,486,260,337 processor cycles
;;   803,371,520 bytes consed

(mapcar (lambda (dtree)
          (test-dtree dtree mnist-test.vec))
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

(defparameter mushrooms-train.vec (coerce *mushrooms-train* 'vector))
(defparameter mushrooms-test.vec (coerce *mushrooms-test* 'vector))

;; decision tree

(defparameter mushrooms-dtree (make-dtree 2 *mushrooms-dim* mushrooms-train.vec))
(traverse #'node-information-gain (dtree-root mushrooms-dtree))
(test-dtree mushrooms-dtree mushrooms-test.vec)
;; => 98.21092278719398d0

(time
 (defparameter mushrooms-forest (make-forest 2 *mushrooms-dim* mushrooms-train.vec
                                             :n-tree 500 :bagging-ratio 0.2 :n-trial 10 :max-depth 5)))

;; Evaluation took:
;;   1.901 seconds of real time
;;   1.902504 seconds of total run time (1.866545 user, 0.035959 system)
;;   [ Run times consist of 0.084 seconds GC time, and 1.819 seconds non-GC time. ]
;;   100.11% CPU
;;   6,448,467,269 processor cycles
;;   1,171,116,816 bytes consed

(test-forest mushrooms-forest mushrooms-train.vec)
(test-forest mushrooms-forest mushrooms-test.vec)

(mapcar (lambda (tree-size)
          (print tree-size)
          (let ((forest (make-forest 2 *mushrooms-dim* mushrooms-train.vec :n-tree tree-size)))
            (test-forest forest mushrooms-train.vec)))
        '(5 10 25 50 75 100 200 300 400 500))

;; ;;;;;;;;;

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
