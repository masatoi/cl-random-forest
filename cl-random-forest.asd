(defpackage cl-random-forest-asd
  (:use :cl :asdf))
(in-package :cl-random-forest-asd)

;; SVMFORMAT is a nickname of the CL-LIBSVM-FORMAT package, and package-inferred-system
;; derives a system name by downcasing the package name -- which would look for a
;; nonexistent "svmformat" system. Registering the mapping is what lets src/utils.lisp
;; declare its dependency on the package it actually uses.
(asdf:register-system-packages "cl-libsvm-format" '(#:svmformat))

;; CLOL and CLOL.VECTOR are nicknames of CL-ONLINE-LEARNING and CL-ONLINE-LEARNING.VECTOR
;; respectively, both provided by the single CL-ONLINE-LEARNING system. Same reasoning as
;; the CL-LIBSVM-FORMAT mapping above: without this, src/random-forest.lisp's :import-from
;; of these nicknames would send ASDF looking for nonexistent "clol"/"clol.vector" systems.
(asdf:register-system-packages "cl-online-learning" '(#:clol #:clol.vector))

(defsystem cl-random-forest
  :version "0.2"
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
