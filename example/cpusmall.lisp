;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

;;; cpusmall
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#cpusmall

;; Shuffle and split
;; $ shuf -o cpusmall_scale.shuf cpusmall_scale
;; $ split -4915 cpusmall_scale.shuf cpusmall_scale.shuf
;; $ mv cpusmall_scale.shufaa cpusmall.train
;; $ mv cpusmall_scale.shufab cpusmall.test

(multiple-value-bind (datamat targetmat)
    (clrf.utils:clol-dataset->datamatrix/target-regression
     (clol.utils:read-data "/home/wiz/datasets/regression/cpusmall.train" 12))
  (defparameter datamatrix datamat)
  (defparameter target targetmat))

(multiple-value-bind (datamat targetmat)
    (clrf.utils:clol-dataset->datamatrix/target-regression
     (clol.utils:read-data "/home/wiz/datasets/regression/cpusmall.test" 12))
  (defparameter test-datamatrix datamat)
  (defparameter test-target targetmat))

(defparameter rtree (make-rtree datamatrix target :max-depth 10))
(test-rtree rtree test-datamatrix test-target)
;; RMSE: 3.846092375763221d0

(require :sb-sprof)

(sb-sprof:with-profiling (:max-samples 1000
                          :report :flat
                          :show-progress t)
  (defparameter rforest
    (make-regression-forest datamatrix target
                            :n-tree 100 :max-depth 15 :bagging-ratio 0.6 :n-trial 10)))

;; 0.794 seconds
(time
 (defparameter rforest
   (make-regression-forest datamatrix target
                          :n-tree 100 :max-depth 15 :bagging-ratio 0.6 :n-trial 10)))

(test-regression-forest rforest test-datamatrix test-target)
;; RMSE: 2.921160545860163d0


;;            Self        Total        Cumul
;;   Nr  Count     %  Count     %  Count     %    Calls  Function
;; ------------------------------------------------------------------------
;;    1    167  16.7    842  84.2    167  16.7        -  VARIANCE
;;    2    154  15.4    154  15.4    321  32.1        -  SB-KERNEL:TWO-ARG-+
;;    3    132  13.2    132  13.2    453  45.3        -  SB-KERNEL:TWO-ARG--
;;    4    100  10.0    100  10.0    553  55.3        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF DOUBLE-FLOAT)
;;    5     68   6.8     68   6.8    621  62.1        -  SB-KERNEL:TWO-ARG-*
;;    6     55   5.5     60   6.0    676  67.6        -  SB-VM::GENERIC-+
;;    7     48   4.8     48   4.8    724  72.4        -  REGION-MIN/MAX
;;    8     47   4.7     47   4.7    771  77.1        -  "foreign function pthread_sigmask"
;;    9     42   4.2     42   4.2    813  81.3        -  SPLIT-SAMPLE-INDICES
;;   10     39   3.9     39   3.9    852  85.2        -  SB-KERNEL:HAIRY-DATA-VECTOR-REF/CHECK-BOUNDS
;;   11     37   3.7     37   3.7    889  88.9        -  SB-IMPL::SLOW-HAIRY-DATA-VECTOR-REF
;;   12     33   3.3     33   3.3    922  92.2        -  (SB-IMPL::OPTIMIZED-DATA-VECTOR-REF FIXNUM)
;;   13     22   2.2     22   2.2    944  94.4        -  SB-KERNEL:HAIRY-DATA-VECTOR-REF
;;   14     12   1.2     13   1.3    956  95.6        -  RANDOM
;;   15      7   0.7     13   1.3    963  96.3        -  SB-KERNEL:TWO-ARG-/
;;   16      6   0.6      6   0.6    969  96.9        -  SB-KERNEL:%DOUBLE-FLOAT
;;   17      5   0.5    989  98.9    974  97.4        -  SET-BEST-CHILDREN!
;;   18      4   0.4      4   0.4    978  97.8        -  COPY-TMP->BEST!
;;   19      3   0.3      3   0.3    981  98.1        -  SB-KERNEL:TWO-ARG-<
;;   20      2   0.2     62   6.2    983  98.3        -  MAKE-RANDOM-TEST
;;   21      2   0.2      9   0.9    985  98.5        -  RANDOM-UNIFORM
;;   22      2   0.2      2   0.2    987  98.7        -  SB-KERNEL:TWO-ARG-=
;;   23      2   0.2      2   0.2    989  98.9        -  (LAMBDA (NODE) :IN SET-LEAF-INDEX!)
;;   24      1   0.1    993  99.3    990  99.0        -  MAKE-RTREE
;;   25      1   0.1      3   0.3    991  99.1        -  DO-LEAF
;;   26      1   0.1      1   0.1    992  99.2        -  DTREE?
;;   27      1   0.1      1   0.1    993  99.3        -  SB-KERNEL::RANDOM-MT19937-UPDATE
;;   28      1   0.1      1   0.1    994  99.4        -  MAKE-PARTIAL-ARR
;;   29      0   0.0   1000 100.0    994  99.4        -  MAKE-REGRESSION-FOREST
;;   30      0   0.0   1000 100.0    994  99.4        -  "Unknown component: #x1008467170"
;;   31      0   0.0   1000 100.0    994  99.4        -  SB-INT:SIMPLE-EVAL-IN-LEXENV
;;   32      0   0.0   1000 100.0    994  99.4        -  EVAL
;;   33      0   0.0   1000 100.0    994  99.4        -  SWANK::EVAL-REGION
;;   34      0   0.0   1000 100.0    994  99.4        -  (LAMBDA NIL :IN SWANK-REPL::REPL-EVAL)
;;   35      0   0.0   1000 100.0    994  99.4        -  SWANK-REPL::TRACK-PACKAGE
;;   36      0   0.0   1000 100.0    994  99.4        -  SWANK::CALL-WITH-RETRY-RESTART
;;   37      0   0.0   1000 100.0    994  99.4        -  SWANK::CALL-WITH-BUFFER-SYNTAX
;;   38      0   0.0   1000 100.0    994  99.4        -  SWANK-REPL::REPL-EVAL
;;   39      0   0.0   1000 100.0    994  99.4        -  SWANK:EVAL-FOR-EMACS
;;   40      0   0.0   1000 100.0    994  99.4        -  SWANK::PROCESS-REQUESTS
;;   41      0   0.0   1000 100.0    994  99.4        -  (LAMBDA NIL :IN SWANK::HANDLE-REQUESTS)
;;   42      0   0.0   1000 100.0    994  99.4        -  SWANK/SBCL::CALL-WITH-BREAK-HOOK
;;   43      0   0.0   1000 100.0    994  99.4        -  (FLET SWANK/BACKEND:CALL-WITH-DEBUGGER-HOOK :IN "/home/wiz/.emacs.d/elpa/slime-20170511.1221/swank/sbcl.lisp")
;;   44      0   0.0   1000 100.0    994  99.4        -  SWANK::CALL-WITH-BINDINGS
;;   45      0   0.0   1000 100.0    994  99.4        -  SWANK::HANDLE-REQUESTS
;;   46      0   0.0   1000 100.0    994  99.4        -  (FLET #:WITHOUT-INTERRUPTS-BODY-1138 :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   47      0   0.0   1000 100.0    994  99.4        -  (FLET SB-THREAD::WITH-MUTEX-THUNK :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   48      0   0.0   1000 100.0    994  99.4        -  (FLET #:WITHOUT-INTERRUPTS-BODY-358 :IN SB-THREAD::CALL-WITH-MUTEX)
;;   49      0   0.0   1000 100.0    994  99.4        -  SB-THREAD::CALL-WITH-MUTEX
;;   50      0   0.0   1000 100.0    994  99.4        -  SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE
;;   51      0   0.0   1000 100.0    994  99.4        -  "foreign function call_into_lisp"
;;   52      0   0.0   1000 100.0    994  99.4        -  "foreign function new_thread_trampoline"
;;   53      0   0.0    992  99.2    994  99.4        -  SPLIT-NODE!
;;   54      0   0.0     47   4.7    994  99.4        -  "foreign function interrupt_handle_pending"
;;   55      0   0.0     47   4.7    994  99.4        -  "foreign function handle_trap"
;;   56      0   0.0      4   0.4    994  99.4        -  BOOTSTRAP-SAMPLE-INDICES
;;   57      0   0.0      3   0.3    994  99.4        -  SET-LEAF-INDEX-FOREST!
;;   58      0   0.0      2   0.2    994  99.4        -  STOP-SPLIT?
;; ------------------------------------------------------------------------
;;           6   0.6                                     elsewhere
