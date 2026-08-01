(in-package :cl-user)

(defpackage cl-random-forest-test/pruning
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture)
  (:import-from #:cl-random-forest/src/random-forest
                #:dtree-max-leaf-index
                #:collect-leaf-parent))
(in-package :cl-random-forest-test/pruning)
