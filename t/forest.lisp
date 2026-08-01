(in-package :cl-user)

(defpackage cl-random-forest-test/forest
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/forest)

(deftest a9a-forest-accuracy
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let ((forest (make-forest 2 datamatrix target
                                                :n-tree 500 :bagging-ratio 0.1
                                                :min-region-samples 5 :n-trial 10
                                                :max-depth 10)))
                       (test-forest forest datamatrix-test target-test)))))
          (ok (approximately-equal acc 84.07d0)
              (format nil "a9a forest accuracy ~,4F (expected 84.07 +/- 1.0)" acc)))))))

(deftest letter-forest-accuracy
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let ((forest (make-forest +letter-n-class+ datamatrix target
                                                :n-tree 500 :bagging-ratio 0.1
                                                :min-region-samples 5 :n-trial 10
                                                :max-depth 10)))
                       (test-forest forest datamatrix-test target-test)))))
          (ok (approximately-equal acc 89.05402374267578d0)
              (format nil "letter forest accuracy ~,4F (expected 89.0540 +/- 1.0)" acc)))))))
