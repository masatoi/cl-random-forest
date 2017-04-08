#|
  This file is a part of cl-random-forest project.
|#

(in-package :cl-user)
(defpackage cl-random-forest-test-asd
  (:use :cl :asdf))
(in-package :cl-random-forest-test-asd)

(defsystem cl-random-forest-test
  :author ""
  :license ""
  :depends-on (:cl-random-forest
               :chipz
               :uiop
               :trivial-garbage
               :prove)
  :components ((:module "t"
                :components
                ((:test-file "cl-random-forest"))))
  :description "Test system for cl-random-forest"

  :defsystem-depends-on (:prove-asdf)
  :perform (test-op :after (op c)
                    (funcall (intern #.(string :run-test-system) :prove-asdf) c)
                    (asdf:clear-system c)))
