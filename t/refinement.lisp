(in-package :cl-user)

(defpackage cl-random-forest-test/refinement
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture)
  (:import-from #:cl-random-forest/src/random-forest
                #:train-refine-learner-process-inner))
(in-package :cl-random-forest-test/refinement)

(deftest a9a-refinement-accuracy
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let* ((forest (make-forest 2 datamatrix target
                                                 :n-tree 500 :bagging-ratio 0.1
                                                 :min-region-samples 5 :n-trial 10
                                                 :max-depth 10))
                            (refine-dataset (make-refine-dataset forest datamatrix))
                            (refine-test (make-refine-dataset forest datamatrix-test))
                            (refine-learner (make-refine-learner forest)))
                       (train-refine-learner-process refine-learner
                                                     refine-dataset target
                                                     refine-test target-test)
                       (test-refine-learner refine-learner refine-test target-test)))))
          (ok (approximately-equal acc 80.98789)
              (format nil "a9a refinement accuracy ~,4F (expected 80.9879 +/- 1.0)" acc)))))))

(deftest letter-refinement-accuracy
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let* ((forest (make-forest +letter-n-class+ datamatrix target
                                                 :n-tree 500 :bagging-ratio 0.1
                                                 :min-region-samples 5 :n-trial 10
                                                 :max-depth 10))
                            (refine-dataset (make-refine-dataset forest datamatrix))
                            (refine-test (make-refine-dataset forest datamatrix-test))
                            (refine-learner (make-refine-learner forest)))
                       (train-refine-learner-process refine-learner
                                                     refine-dataset target
                                                     refine-test target-test)
                       (test-refine-learner refine-learner refine-test target-test)))))
          (ok (approximately-equal acc 97.06802368164063d0)
              (format nil "letter refinement accuracy ~,4F (expected 97.0680 +/- 1.0)" acc)))))))

;;;; Convergence detection
;;;;
;;;; These need no forest. MAKE-REFINE-DATASET's output is a simple-vector of per-datum
;;;; leaf-index vectors and nothing downstream of it knows where the indices came from, so
;;;; synthesising one keeps the test deterministic and quick -- and keeps it about the
;;;; training process rather than about forest construction.

(defun synthetic-refine-dataset (n-datum n-tree block-size n-class seed
                                 &key (feature-noise 6) (label-noise 0))
  "Return (values dataset target) shaped like MAKE-REFINE-DATASET's output.

Each tree owns a block of BLOCK-SIZE leaf indices and a datum lands on the leaf its class
picks, except one time in FEATURE-NOISE when it lands elsewhere in the block. One label in
LABEL-NOISE is replaced by a random class, or none when LABEL-NOISE is 0.

The noise is not decoration. The process under test stops at the first epoch that fails to
improve, so a separable problem would peak at epoch one, hold, and never distinguish the
best epoch from the last -- which is exactly the confusion these tests exist to catch. At
the settings the tests use, accuracy peaks on epoch 1 and drops on epoch 2."
  (let ((dataset (make-array n-datum))
        (target (make-array n-datum :element-type 'fixnum))
        (state seed))
    (flet ((next ()
             (setf state (mod (+ (* 1103515245 state) 12345) 2147483648))))
      (dotimes (i n-datum)
        (let ((class (mod i n-class))
              (indices (make-array n-tree :element-type 'fixnum)))
          (dotimes (tree n-tree)
            (let ((offset (if (zerop (mod (next) feature-noise))
                              (mod (next) block-size)
                              (mod class block-size))))
              (setf (aref indices tree) (+ (* tree block-size) offset))))
          (setf (aref target i)
                (if (and (plusp label-noise) (zerop (mod (next) label-noise)))
                    (mod (next) n-class)
                    class))
          (setf (svref dataset i) indices))))
    (values dataset target)))

(defconstant +process-n-tree+ 6)
(defconstant +process-block-size+ 6)
(defconstant +process-n-class+ 4)

(defun process-fixture ()
  "Return (values learner train-dataset train-target test-dataset test-target).
Training labels carry noise the test set does not, so the learner overfits and test
accuracy falls after epoch 1: measured 97.00 then 96.67."
  (multiple-value-bind (train-dataset train-target)
      (synthetic-refine-dataset 600 +process-n-tree+ +process-block-size+
                                +process-n-class+ 20260806 :label-noise 6)
    (multiple-value-bind (test-dataset test-target)
        (synthetic-refine-dataset 300 +process-n-tree+ +process-block-size+
                                  +process-n-class+ 7777)
      (values (clol:make-one-vs-rest (* +process-n-tree+ +process-block-size+)
                                     +process-n-class+ 'clol::sparse-arow 10.0)
              train-dataset train-target test-dataset test-target))))

(deftest refine-learner-process-returns-the-learner-it-reports
  ;; TRAIN-REFINE-LEARNER-PROCESS-INNER hands back the best epoch's learner and that
  ;; epoch's accuracy, so measuring the learner it returns must reproduce the number
  ;; returned beside it. Before issue #20 it did not: the snapshot was defstruct's shallow
  ;; copy and shared the weight arrays, so what came back was the last epoch -- here the
  ;; 96.67 of epoch 2 reported as epoch 1's 97.00.
  (multiple-value-bind (learner train-dataset train-target test-dataset test-target)
      (process-fixture)
    (multiple-value-bind (returned reported)
        (train-refine-learner-process-inner learner train-dataset train-target
                                            test-dataset test-target :max-epoch 50)
      (let ((actual (test-refine-learner returned test-dataset test-target :quiet-p t))
            (last-epoch (test-refine-learner learner test-dataset test-target :quiet-p t)))
        (ok (> reported last-epoch)
            (format nil "the best epoch ~,4F really did beat the last ~,4F, so this test ~
can tell them apart" reported last-epoch))
        (ok (= actual reported)
            (format nil "returned learner scores ~,4F, process reported ~,4F"
                    actual reported))))))

(deftest refine-learner-process-holds-when-max-epoch-runs-out
  ;; The other exit. Snapshotting before each epoch is the best epoch only when the loop
  ;; breaks early; running out of MAX-EPOCH used to return the epoch before the last one
  ;; beside the last one's accuracy. At MAX-EPOCH 1 that snapshot is the untrained
  ;; learner, so the two disagree by everything the first epoch learned.
  (multiple-value-bind (learner train-dataset train-target test-dataset test-target)
      (process-fixture)
    (multiple-value-bind (returned reported)
        (train-refine-learner-process-inner learner train-dataset train-target
                                            test-dataset test-target :max-epoch 1)
      (let ((actual (test-refine-learner returned test-dataset test-target :quiet-p t)))
        (ok (> reported (/ 100.0 +process-n-class+))
            (format nil "one epoch beat chance: ~,4F against ~,4F"
                    reported (/ 100.0 +process-n-class+)))
        (ok (= actual reported)
            (format nil "after one epoch, returned ~,4F against reported ~,4F"
                    actual reported))))))

(deftest refine-learner-process-snapshot-outlives-further-training
  ;; The mechanism rather than the symptom: what the process returns must be unaffected by
  ;; training the argument learner goes on to receive. A shallow copy fails this even when
  ;; the accuracies happen to line up.
  (multiple-value-bind (learner train-dataset train-target test-dataset test-target)
      (process-fixture)
    (multiple-value-bind (returned reported)
        (train-refine-learner-process-inner learner train-dataset train-target
                                            test-dataset test-target :max-epoch 50)
      (dotimes (epoch 5)
        (train-refine-learner learner train-dataset train-target))
      (ok (= reported (test-refine-learner returned test-dataset test-target :quiet-p t))
          "five more epochs on the original left the returned learner untouched"))))
