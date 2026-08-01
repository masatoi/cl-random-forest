(in-package :cl-user)

(defpackage cl-random-forest-test/decision-tree
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/decision-tree)

(deftest a9a-dtree-accuracy
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (let ((acc (n-times-average 10
                   (let ((dtree (make-dtree 2 datamatrix target :max-depth 20)))
                     (test-dtree dtree datamatrix-test target-test)))))
        (ok (approximately-equal acc 82.23217010498047d0)
            (format nil "a9a dtree accuracy ~,4F (expected 82.2322 +/- 1.0)" acc))))))

(deftest letter-dtree-accuracy
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (let ((acc (n-times-average 10
                   (let ((dtree (make-dtree +letter-n-class+ datamatrix target :max-depth 20)))
                     (test-dtree dtree datamatrix-test target-test)))))
        (ok (approximately-equal acc 83.9534912109375d0)
            (format nil "letter dtree accuracy ~,4F (expected 83.9535 +/- 1.0)" acc))))))
