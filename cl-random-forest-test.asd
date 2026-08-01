#|
  This file is a part of cl-random-forest project.
|#

(defsystem "cl-random-forest-test/fixture"
  :author "Satoshi Imai"
  :license "MIT Licence"
  :description "Shared fixtures for the cl-random-forest test suites. Contains no tests."
  :depends-on ("cl-random-forest" "cl-online-learning" "uiop" "trivial-garbage" "lparallel")
  :components ((:module "t" :components ((:file "fixture")))))

(defsystem "cl-random-forest-test/dataset"
  :description "Tests for dataset download and conversion to datamatrix/target"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "dataset"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/decision-tree"
  :description "Accuracy tests for decision trees"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "decision-tree"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/forest"
  :description "Accuracy tests for random forests"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "forest"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/refinement"
  :description "Accuracy tests for global refinement"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "refinement"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/parallel"
  :description "Accuracy tests for parallelized training (SBCL only)"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "parallel"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test"
  :author "Satoshi Imai"
  :license "MIT Licence"
  :description "Test system for cl-random-forest"
  :depends-on ("rove"
               "cl-random-forest-test/dataset"
               "cl-random-forest-test/decision-tree"
               "cl-random-forest-test/forest"
               "cl-random-forest-test/refinement"
               "cl-random-forest-test/parallel")
  ;; NOTE: This system has no components of its own, so rove's SYSTEM-SUITES would
  ;; return an empty list and (rove:run c) would silently report zero tests.
  ;; The feature systems must be listed explicitly.
  :perform (test-op (o c)
             (declare (ignore o c))
             (unless (symbol-call :rove :run
                                  '("cl-random-forest-test/dataset"
                                    "cl-random-forest-test/decision-tree"
                                    "cl-random-forest-test/forest"
                                    "cl-random-forest-test/refinement"
                                    "cl-random-forest-test/parallel"))
               (error "Tests failed."))))
