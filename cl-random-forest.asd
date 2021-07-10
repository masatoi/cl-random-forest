(defpackage cl-random-forest-asd
  (:use :cl :asdf))
(in-package :cl-random-forest-asd)

(defsystem cl-random-forest
  :version "0.20"
  :author "Satoshi Imai"
  :license "MIT Licence"
  :class :package-inferred-system
  :depends-on (:cl-libsvm-format
               :cl-online-learning
               :alexandria
               :lparallel
               :cl-random-forest/cl-random-forest)
  :description "Random Forest for Common Lisp"
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.org"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input
                            :external-format :utf-8)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (test-op cl-random-forest-test))))
