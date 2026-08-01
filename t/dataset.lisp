(in-package :cl-user)

(defpackage cl-random-forest-test/dataset
  (:use :cl :rove :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/dataset)

(deftest dataset-files-exist
  ;; Nothing else fetches eagerly any more, so this test does it itself.
  (fetch-a9a)
  (fetch-letter)
  (ok (and (uiop:file-exists-p (merge-pathnames "letter.scale" *dataset-dir*))
           (uiop:file-exists-p (merge-pathnames "letter.scale.t" *dataset-dir*))
           (uiop:file-exists-p (merge-pathnames "a9a" *dataset-dir*))
           (uiop:file-exists-p (merge-pathnames "a9a.t" *dataset-dir*)))
      "all four dataset files exist in *dataset-dir*"))

(deftest a9a-dataset-load
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (ok (and datamatrix target datamatrix-test target-test)
          "a9a train and test sets load into datamatrix/target"))))

(deftest letter-dataset-load
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (ok (and datamatrix target datamatrix-test target-test)
          "letter train and test sets load into datamatrix/target"))))
