(uiop:define-package :cl-random-forest/src/packed
    (:use :cl)
  (:use-reexport :cl-random-forest/src/packed/topology)
  (:use-reexport :cl-random-forest/src/packed/classifier))

(in-package :cl-random-forest/src/packed)
