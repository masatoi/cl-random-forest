#|
  This file is a part of cl-random-forest project.
|#

(in-package :cl-user)
(defpackage cl-random-forest-asd
  (:use :cl :asdf))
(in-package :cl-random-forest-asd)

(defsystem cl-random-forest
  :version "0.1"
  :author ""
  :license ""
  :depends-on (:cl-online-learning :alexandria :lparallel)
  :components ((:module "src"
                :components
                ((:file "random-forest"))))
  :description ""
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.markdown"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (test-op cl-random-forest-test))))
