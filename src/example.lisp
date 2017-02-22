(in-package :clrf)

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

(defparameter *forest* (make-forest 4 2 *dataset2* :n-tree 1 :bagging-ratio 0.5))

(traverse #'node-sample-indices (dtree-root (car (forest-dtree-list *forest*))))
;;;;;;;;;;;;;;;;;;;; MNIST

(defparameter mnist-dim 780)
(defparameter mnist-train (clol.utils:read-data "/home/wiz/tmp/mnist.scale" mnist-dim :multiclass-p t))
(defparameter mnist-test  (clol.utils:read-data "/home/wiz/tmp/mnist.scale.t" mnist-dim :multiclass-p t))

;; Add 1 to labels because the labels of this dataset begin from 0
(dolist (datum mnist-train) (incf (car datum)))
(dolist (datum mnist-test)  (incf (car datum)))

(defparameter mnist-train.vec (coerce mnist-train 'vector))
(defparameter mnist-test.vec (coerce mnist-test 'vector))

(require :sb-sprof)

(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (defparameter mnist-dtree (make-dtree 10 780 mnist-train.vec :max-depth 100)))

(time (defparameter mnist-dtree
        (make-dtree 10 780 (subseq mnist-train.vec 0 6000) :max-depth 5 :n-trial 10 :n-random-feature 27)))

(traverse #'node-information-gain (dtree-root mnist-dtree))

;; ノードに入っている数
(traverse (lambda (node)
            (length (node-sample-indices node)))
          (dtree-root mnist-dtree))

(sb-sprof:with-profiling (:max-samples 1000 :mode :alloc :report :flat)
  (defparameter mnist-dtree (make-dtree 10 780 mnist-train.vec :max-depth 100)))

;; Evaluation took:
;;   51.805 seconds of real time
;;   51.949514 seconds of total run time (51.949514 user, 0.000000 system)
;;   [ Run times consist of 0.165 seconds GC time, and 51.785 seconds non-GC time. ]
;;   100.28% CPU
;;   175,730,792,502 processor cycles
;;   403,319,552 bytes consed

;;;; prediction
(ql:quickload :clgplot)
(clgp:plot-histogram (class-distribution (find-leaf (dtree-root mnist-dtree) (cdr (car mnist-test)))) 10)

(time (defparameter mnist-dtree
  (make-dtree 10 780 mnist-train.vec :max-depth 5 :n-trial 10)))

(loop for c across (class-distribution (dtree-root mnist-dtree)) sum c)

(class-distribution (make-node nil (dtree-root mnist-dtree)))
(entropy (make-node nil (dtree-root mnist-dtree)))
(gini (make-node nil (dtree-root mnist-dtree)))

(predict-dtree mnist-dtree (cdr (car mnist-test)))

(test-dtree mnist-dtree mnist-train.vec)
(test-dtree mnist-dtree mnist-test.vec)

;;; forest

(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (defparameter mnist-forest (make-forest 10 780 mnist-train.vec :n-tree 500 :bagging-ratio 0.1 :max-depth 10 :n-trial 27)))


(defparameter mnist-forest (make-forest 10 780 mnist-train.vec :n-tree 100 :bagging-ratio 0.1 :max-depth 5 :n-trial 10))

(loop for x across (class-distribution-forest mnist-forest (cdr (car mnist-test))) sum x)
(predict-forest mnist-forest (cdr (car mnist-test)))

(sb-sprof:with-profiling (:max-samples 1000 :report :flat :loop nil)
  (test-forest mnist-forest mnist-test.vec))

(mapcar (lambda (dtree)
          (test-dtree dtree mnist-test.vec))
        (forest-dtree-list mnist-forest))


(mapcar (lambda (dtree)
          (traverse #'node-information-gain (dtree-root dtree)))
        (forest-dtree-list mnist-forest))

(mapcar (lambda (dtree)
          (traverse (lambda (node)
                      (length (node-sample-indices node)))
                    (dtree-root dtree)))
        (forest-dtree-list mnist-forest))

(defparameter mnist-forest (make-forest 10 780 mnist-train.vec :n-tree 1 :bagging-ratio 1.0))
(predict-forest mnist-forest (cdr (car mnist-test)))
(test-forest mnist-forest mnist-test.vec)


(defparameter mnist-forest (make-forest 10 780 mnist-train.vec :n-tree 10 :bagging-ratio 0.1 :max-depth 5 :n-trial 10))

(dolist (dtree (forest-dtree-list mnist-forest))
  (clgp:plot-histogram (class-distribution (find-leaf (dtree-root dtree) (cdr (car mnist-test)))) 10))

(clgp:plot (coerce (count-class-forest mnist-forest (cdr (car mnist-test))) 'list))

(test-forest mnist-forest mnist-train.vec)

(defparameter sample-indices-list
  (mapcar #'node-sample-indices (mapcar #'dtree-root (forest-dtree-list mnist-forest))))

(clgp:plot (car sample-indices-list) :style 'impulses)
(clgp:plot (cadr sample-indices-list) :style 'impulses)

;;;;;;;;;;;;; mushrooms

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

(defparameter mushrooms-dtree (make-dtree 2 *mushrooms-dim* mushrooms-train.vec))
(traverse #'node-information-gain (dtree-root mushrooms-dtree))

(time
 (defparameter mushrooms-forest (make-forest 2 *mushrooms-dim* mushrooms-train.vec
                                             :n-tree 500 :bagging-ratio 1.0 :n-trial 100)))

(test-forest mushrooms-forest mushrooms-train.vec)
(test-forest mushrooms-forest mushrooms-test.vec)

(mapcar (lambda (tree-size)
          (print tree-size)
          (let ((forest (make-forest 2 *mushrooms-dim* mushrooms-train.vec :n-tree tree-size)))
            (test-forest forest mushrooms-train.vec)))
        '(5 10 25 50 75 100 200 300 400 500))

;; => なぜかたまにエラー
;; find-leafがnilになっている
;; どういうケース？ 左右の枝のバランスが崩れている場合
(car *dbg*)
(traverse #'node-information-gain (dtree-root (car *dbg*)))
(traverse #'node-sample-indices (dtree-root (car *dbg*)))
(find-leaf (dtree-root (car *dbg*)) (cadr *dbg*))

(entropy (make-node nil (dtree-root *dtree*)))

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
