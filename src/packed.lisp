(uiop:define-package :cl-random-forest/src/packed
    (:use :cl)
  (:nicknames :clrf.packed)
  (:use-reexport :cl-random-forest/src/packed/topology)
  (:use-reexport :cl-random-forest/src/packed/classifier)
  (:use-reexport :cl-random-forest/src/packed/io))

(in-package :cl-random-forest/src/packed)
