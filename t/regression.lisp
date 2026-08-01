(in-package :cl-user)

(defpackage cl-random-forest-test/regression
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/regression)

(defun trained-forest ()
  "Build the regression forest the accuracy tests share."
  (multiple-value-bind (datamatrix target) (synthetic-regression-train)
    (make-regression-forest datamatrix target
                            :n-tree 50 :bagging-ratio 0.3
                            :max-depth 10 :n-trial 10)))

(deftest regression-tree-fits-the-signal
  (multiple-value-bind (datamatrix target) (synthetic-regression-train)
    (multiple-value-bind (datamatrix-test target-test) (synthetic-regression-test)
      (let* ((baseline (target-stddev target-test))
             (tree (make-rtree datamatrix target :max-depth 10))
             (rmse (test-rtree tree datamatrix-test target-test :quiet-p t)))
        ;; Measured 0.24-0.28 of baseline over five runs; 0.5 leaves ~2x headroom.
        (ok (< rmse (* 0.5 baseline))
            (format nil "rtree RMSE ~,4F < 0.5 * baseline ~,4F (= ~,4F)"
                    rmse baseline (* 0.5 baseline)))))))

(deftest regression-forest-beats-single-tree
  (multiple-value-bind (datamatrix target) (synthetic-regression-train)
    (multiple-value-bind (datamatrix-test target-test) (synthetic-regression-test)
      (let ((tree-rmse (test-rtree (make-rtree datamatrix target :max-depth 10)
                                   datamatrix-test target-test :quiet-p t))
            (forest-rmse (test-regression-forest (trained-forest)
                                                 datamatrix-test target-test :quiet-p t)))
        ;; Held in all five measured runs, by roughly 35%.
        (ok (< forest-rmse tree-rmse)
            (format nil "forest RMSE ~,4F < single tree RMSE ~,4F"
                    forest-rmse tree-rmse))))))

(deftest regression-forest-explains-the-signal
  (multiple-value-bind (datamatrix-test target-test) (synthetic-regression-test)
    (let* ((baseline (target-stddev target-test))
           (rmse (test-regression-forest (trained-forest)
                                         datamatrix-test target-test :quiet-p t)))
      ;; Measured 0.162-0.168 of baseline over five runs; 0.25 leaves ~1.5x headroom.
      (ok (< rmse (* 0.25 baseline))
          (format nil "forest RMSE ~,4F < 0.25 * baseline ~,4F (= ~,4F)"
                  rmse baseline (* 0.25 baseline))))))

(deftest regression-refinement-improves-on-the-forest
  (multiple-value-bind (datamatrix target) (synthetic-regression-train)
    (multiple-value-bind (datamatrix-test target-test) (synthetic-regression-test)
      (let* ((forest (trained-forest))
             (forest-rmse (test-regression-forest forest datamatrix-test target-test
                                                  :quiet-p t))
             (refine-train (make-regression-refine-dataset forest datamatrix))
             (refine-test (make-regression-refine-dataset forest datamatrix-test))
             ;; gamma 1.0 rather than the 0.99 default -- see the pinning test below.
             (learner (make-regression-refine-learner forest 1.0)))
        (dotimes (epoch 10)
          (train-regression-refine-learner learner refine-train target))
        (let ((refine-rmse (test-regression-refine-learner learner refine-test target-test
                                                           :quiet-p t)))
          (ok (< refine-rmse forest-rmse)
              (format nil "refined RMSE ~,4F < forest RMSE ~,4F"
                      refine-rmse forest-rmse)))))))

(deftest regression-refine-learner-default-gamma-diverges
  ;; KNOWN BUG (issue #16): make-regression-refine-learner defaults gamma to 0.99.
  ;; A forgetting factor below 1.0 scales the sparse RLS covariance by 1/gamma on
  ;; every update, including for the coordinates a sample does not excite, so it
  ;; grows as (1/gamma)^updates and overflows single-float. Measured: NaN at epoch
  ;; 9 with 1000 training data. This test pins the current behaviour -- when it
  ;; starts FAILING, the default has been fixed and this test should be deleted.
  (multiple-value-bind (datamatrix target) (synthetic-regression-train)
    (let* ((forest (trained-forest))
           (refine-train (make-regression-refine-dataset forest datamatrix))
           (learner (make-regression-refine-learner forest)) ; default gamma 0.99
           (diverged-at nil))
      (dotimes (epoch 12)
        (train-regression-refine-learner learner refine-train target)
        (let ((rmse (test-regression-refine-learner learner refine-train target
                                                    :quiet-p t)))
          (when (/= rmse rmse)          ; NaN is the only float not equal to itself
            (setf diverged-at (1+ epoch))
            (return))))
      (ok diverged-at
          (format nil "default gamma 0.99 produced NaN at epoch ~A (expected within 12)"
                  diverged-at)))))
