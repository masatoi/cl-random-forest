(in-package :cl-user)

(defpackage cl-random-forest-test/refinement
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
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
