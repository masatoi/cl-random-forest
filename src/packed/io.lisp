(defpackage :cl-random-forest/src/packed/io
  (:use #:cl
        #:cl-random-forest/src/packed/topology
        #:cl-random-forest/src/packed/classifier)
  (:export #:packed-save
           #:packed-load
           #:packed-load-error
           #:packed-load-error-detail
           #:single-float-to-bits
           #:bits-to-single-float
           #:%portable-single-float-to-bits))

(in-package :cl-random-forest/src/packed/io)

(define-condition packed-load-error (error)
  ((detail :initarg :detail :reader packed-load-error-detail))
  (:report (lambda (c s)
             (format s "cannot load this packed model: ~A" (packed-load-error-detail c)))))

;; A DEFCONSTANT whose value is an array signals on reload, the new value not being EQL to
;; the old, so the magic is a DEFPARAMETER. The other two are numbers and are fine.
(defparameter *magic*
  #.(map '(simple-array (unsigned-byte 8) (*)) #'char-code "CLRFPACK"))
(defconstant +version+ 1)
(defconstant +byte-order-probe+ #x01020304)

;;;; Floats as bits
;;;;
;;;; The file holds IEEE-754 bit patterns. SBCL and CCL can produce them directly; the
;;;; portable version is the reference the tests hold the fast paths to, and the fallback
;;;; anywhere else.

(defun %portable-single-float-to-bits (x)
  "IEEE-754 single-precision bits of X, in ANSI Common Lisp only."
  (declare (type single-float x))
  (if (zerop x)
      (if (minusp (float-sign x)) #x80000000 0)
      (multiple-value-bind (significand exponent sign) (integer-decode-float x)
        (let ((biased (+ exponent 23 127)))
          (logior (if (minusp sign) #x80000000 0)
                  (ash (logand biased #xff) 23)
                  (logand significand #x7fffff))))))

(defun %portable-bits-to-single-float (bits)
  "The single-float whose IEEE-754 bits are BITS, in ANSI Common Lisp only."
  (declare (type (unsigned-byte 32) bits))
  (let* ((sign (if (logbitp 31 bits) -1.0 1.0))
         (exponent (ldb (byte 8 23) bits))
         (fraction (ldb (byte 23 0) bits)))
    (cond ((and (zerop exponent) (zerop fraction)) (* sign 0.0))
          ((zerop exponent) (* sign (scale-float (float fraction 1.0) -149)))
          (t (* sign (scale-float (float (logior fraction (ash 1 23)) 1.0)
                                  (- exponent 127 23)))))))

(defun single-float-to-bits (x)
  "IEEE-754 single-precision bits of X."
  (declare (type single-float x))
  #+sbcl (ldb (byte 32 0) (sb-kernel:single-float-bits x))
  #+ccl (ldb (byte 32 0) (ccl::single-float-bits x))
  #-(or sbcl ccl) (%portable-single-float-to-bits x))

(defun bits-to-single-float (bits)
  "The single-float whose IEEE-754 bits are BITS."
  (declare (type (unsigned-byte 32) bits))
  #+sbcl (sb-kernel:make-single-float
          (if (logbitp 31 bits) (- bits (ash 1 32)) bits))
  #+ccl (ccl::host-single-float-from-unsigned-byte-32 bits)
  #-(or sbcl ccl) (%portable-bits-to-single-float bits))

;;;; Byte-level primitives

(defun write-u32 (value stream)
  (declare (type (unsigned-byte 32) value))
  (dotimes (i 4) (write-byte (ldb (byte 8 (* 8 i)) value) stream)))

(defun read-u32 (stream)
  (let ((value 0))
    (dotimes (i 4 value)
      (let ((byte (read-byte stream nil nil)))
        (unless byte (error 'packed-load-error :detail "the file ends mid-header"))
        (setf value (logior value (ash byte (* 8 i))))))))

(defun write-array (array stream bits encoder)
  "Write ARRAY as little-endian words of BITS bits, ENCODER mapping element to integer."
  (let* ((bytes-per (floor bits 8))
         (n (length array))
         (buffer (make-array (* n bytes-per) :element-type '(unsigned-byte 8))))
    (dotimes (i n)
      (let ((word (funcall encoder (aref array i))))
        (dotimes (b bytes-per)
          (setf (aref buffer (+ (* i bytes-per) b)) (ldb (byte 8 (* 8 b)) word)))))
    (write-sequence buffer stream)))

(defun read-array (stream n bits element-type decoder)
  "Read N little-endian words of BITS bits, DECODER mapping integer to element."
  (let* ((bytes-per (floor bits 8))
         (buffer (make-array (* n bytes-per) :element-type '(unsigned-byte 8)))
         (got (read-sequence buffer stream))
         (out (make-array n :element-type element-type)))
    (unless (= got (* n bytes-per))
      (error 'packed-load-error
             :detail (format nil "wanted ~D bytes of array data, got ~D"
                             (* n bytes-per) got)))
    (dotimes (i n out)
      (let ((word 0))
        (dotimes (b bytes-per)
          (setf word (logior word (ash (aref buffer (+ (* i bytes-per) b)) (* 8 b)))))
        (setf (aref out i) (funcall decoder word))))))

(defun u32-of-s32 (x) (ldb (byte 32 0) x))
(defun s32-of-u32 (x) (if (logbitp 31 x) (- x (ash 1 32)) x))

;;;; Save and load

(defun packed-save (classifier pathname)
  "Write CLASSIFIER to PATHNAME. Returns PATHNAME.

The format is a header followed by the arrays as little-endian words. Byte order is *not*
converted on load -- the writing machine's is recorded and a mismatch is refused, which is
worth more than conversion code that could not be exercised here."
  (let* ((topology (packed-classifier-topology classifier))
         (n-leaf (packed-topology-n-leaf topology))
         (n-class (packed-classifier-n-class classifier))
         (csr (eq (packed-classifier-kind classifier) :csr)))
    (with-open-file (s pathname :direction :output :if-exists :supersede
                                :element-type '(unsigned-byte 8))
      (write-sequence *magic* s)
      (write-u32 +version+ s)
      (write-u32 +byte-order-probe+ s)
      (write-u32 (if csr 1 0) s)
      (write-u32 (packed-topology-n-tree topology) s)
      (write-u32 (packed-topology-n-internal topology) s)
      (write-u32 n-leaf s)
      (write-u32 n-class s)
      (write-array (packed-topology-feature topology) s 32 #'identity)
      (write-array (packed-topology-threshold topology) s 32 #'single-float-to-bits)
      (write-array (packed-topology-left topology) s 32 #'u32-of-s32)
      (write-array (packed-topology-right topology) s 32 #'u32-of-s32)
      (write-array (packed-topology-roots topology) s 32 #'u32-of-s32)
      (write-array (packed-topology-tree-leaf-offsets topology) s 32 #'identity)
      (if csr
          (progn
            (write-array (packed-classifier-offsets classifier) s 32 #'identity)
            (write-array (packed-classifier-class classifier) s 16 #'identity)
            (write-array (packed-classifier-probability classifier) s 32
                         #'single-float-to-bits))
          (let* ((table (packed-classifier-table classifier))
                 (flat (make-array (* (max n-leaf 1) n-class)
                                   :element-type 'single-float)))
            (dotimes (row (max n-leaf 1))
              (dotimes (k n-class)
                (setf (aref flat (+ (* row n-class) k)) (aref table row k))))
            (write-array flat s 32 #'single-float-to-bits))))
    pathname))

(defun packed-load (pathname)
  "Read a packed classifier written by PACKED-SAVE."
  (with-open-file (s pathname :direction :input :element-type '(unsigned-byte 8))
    (let ((magic (make-array (length *magic*) :element-type '(unsigned-byte 8))))
      (unless (and (= (read-sequence magic s) (length *magic*))
                   (equalp magic *magic*))
        (error 'packed-load-error :detail "not a packed forest file")))
    (let ((version (read-u32 s)))
      (unless (= version +version+)
        (error 'packed-load-error
               :detail (format nil "version ~D, this build reads ~D" version +version+))))
    (let ((order (read-u32 s)))
      (unless (= order +byte-order-probe+)
        (error 'packed-load-error
               :detail (format nil "written on a machine of different byte order (~8,'0X)"
                               order))))
    (let* ((csr (= 1 (read-u32 s)))
           (n-tree (read-u32 s))
           (n-internal (read-u32 s))
           (n-leaf (read-u32 s))
           (n-class (read-u32 s))
           (topology
             (%make-packed-topology
              :n-tree n-tree :n-internal n-internal :n-leaf n-leaf
              :feature (read-array s n-internal 32 '(unsigned-byte 32) #'identity)
              :threshold (read-array s n-internal 32 'single-float #'bits-to-single-float)
              :left (read-array s n-internal 32 '(signed-byte 32) #'s32-of-u32)
              :right (read-array s n-internal 32 '(signed-byte 32) #'s32-of-u32)
              :roots (read-array s n-tree 32 '(signed-byte 32) #'s32-of-u32)
              :tree-leaf-offsets (read-array s n-tree 32 '(unsigned-byte 32)
                                             #'identity))))
      (if csr
          (let* ((offsets (read-array s (1+ n-leaf) 32 '(unsigned-byte 32) #'identity))
                 (nnz (aref offsets n-leaf)))
            (%make-packed-classifier
             :topology topology :n-class n-class :kind :csr
             :offsets offsets
             :class (read-array s nnz 16 '(unsigned-byte 16) #'identity)
             :probability (read-array s nnz 32 'single-float #'bits-to-single-float)))
          (let ((flat (read-array s (* (max n-leaf 1) n-class) 32 'single-float
                                  #'bits-to-single-float))
                (table (make-array (list (max n-leaf 1) n-class)
                                   :element-type 'single-float)))
            (dotimes (row (max n-leaf 1))
              (dotimes (k n-class)
                (setf (aref table row k) (aref flat (+ (* row n-class) k)))))
            (%make-packed-classifier
             :topology topology :n-class n-class :kind :dense :table table))))))
