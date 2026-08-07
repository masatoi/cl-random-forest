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
           #:%portable-single-float-to-bits
           #:%portable-bits-to-single-float))

(in-package :cl-random-forest/src/packed/io)

(define-condition packed-load-error (error)
  ((detail :initarg :detail :reader packed-load-error-detail))
  (:report (lambda (c s)
             (format s "cannot load this packed model: ~A" (packed-load-error-detail c)))))

;; A DEFCONSTANT whose value is an array signals on reload, the new value not being EQL to
;; the old, so the magic is a DEFPARAMETER. The other two are numbers and are fine.
(defparameter *magic*
  #.(map '(simple-array (unsigned-byte 8) (*)) #'char-code "CLRFPACK"))
(defconstant +version+ 2)
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
        (let ((sign-bit (if (minusp sign) #x80000000 0)))
          (if (< significand (ash 1 23))
              ;; A denormal. INTEGER-DECODE-FLOAT leaves its significand unnormalised --
              ;; LEAST-POSITIVE-SINGLE-FLOAT comes back as significand 1, exponent -149 --
              ;; so the biased-exponent arithmetic below would write a bogus non-zero
              ;; exponent field. IEEE-754 stores a denormal as exponent field zero and a
              ;; fraction that is the value divided by 2^-149.
              (logior sign-bit (ash significand (+ exponent 149)))
              (logior sign-bit
                      (ash (logand (+ exponent 23 127) #xff) 23)
                      (logand significand #x7fffff)))))))

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

(defun check-array-fits (stream name n bits)
  "Signal PACKED-LOAD-ERROR unless N words of BITS bits still remain in STREAM.

READ-ARRAY has to size two buffers from N before it can discover that the file is shorter
than N claims, and N is a header field a corrupt file controls outright: a 36-byte file
declaring 4294967295 internal nodes would ask for 16 GB before reaching the short-read check.
Comparing against the bytes actually left costs one FILE-LENGTH and forecloses that."
  (let* ((wanted (* n (floor bits 8)))
         (remaining (- (file-length stream) (file-position stream))))
    (when (> wanted remaining)
      (error 'packed-load-error
             :detail (format nil "~A claims ~D entries, needing ~D bytes, but only ~D ~
remain in the file" name n wanted remaining)))))

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

;;;; Loaded-array validation
;;;;
;;;; BUILD-PACKED-TOPOLOGY validates exhaustively before anything reaches its (safety 0)
;;;; traversal. PACKED-LOAD reconstructs the same arrays from bytes it does not control, so
;;;; it owes them the same discipline: every index that PACKED-LEAF or the CSR loop in
;;;; PACKED-PREDICT will follow at (safety 0) is checked here first, where a bad one is an
;;;; ordinary error instead of an out-of-bounds read or write. Thresholds and the leaf
;;;; payload are not checked -- any bit pattern is a legal float, and NaN is not this
;;;; layer's problem.

(defun check-header-count (name value &key positive)
  "Signal PACKED-LOAD-ERROR unless VALUE is a valid header count: non-negative always, and
positive too when POSITIVE is true. READ-U32 cannot actually hand back a negative value, but
the check is stated explicitly rather than left as an accident of the encoding."
  (unless (<= 0 value)
    (error 'packed-load-error :detail (format nil "header ~A is negative: ~D" name value)))
  (when (and positive (not (plusp value)))
    (error 'packed-load-error
           :detail (format nil "header ~A must be positive, got ~D" name value))))

(defun check-child-reference (array-name index value n-internal n-leaf &key parent)
  "Signal PACKED-LOAD-ERROR unless VALUE is a valid child reference at ARRAY-NAME[INDEX]:
either an internal-node index in [0, N-INTERNAL), or the BUILD-PACKED-TOPOLOGY encoding of a
leaf, `~L` for some leaf L in [0, N-LEAF).

When PARENT is the index of the node VALUE is a child of, an internal VALUE must also exceed
it. Bounds alone do not make the walk terminate: PACKED-LEAF loops while the node index is
non-negative, so `left[i] = i` -- in range, and a cycle -- hangs it at (safety 0). Requiring
each step to move forward through a bounded array makes termination provable instead, which
is exactly the numbering BUILD-PACKED-TOPOLOGY produces and now checks."
  (if (minusp value)
      (let ((leaf (lognot value)))
        (unless (< -1 leaf n-leaf)
          (error 'packed-load-error
                 :detail (format nil "~A[~D] = ~D references leaf ~D, outside [0,~D)"
                                 array-name index value leaf n-leaf))))
      (progn
        (unless (< value n-internal)
          (error 'packed-load-error
                 :detail (format nil "~A[~D] = ~D, outside the internal-node range [0,~D)"
                                 array-name index value n-internal)))
        (when (and parent (<= value parent))
          (error 'packed-load-error
                 :detail (format nil "~A[~D] = ~D does not exceed its parent index ~D, so ~
the walk could revisit a node and never terminate" array-name index value parent))))))

(defun validate-feature-array (n-internal datum-dim feature)
  "Signal PACKED-LOAD-ERROR unless every FEATURE entry is a column of a DATUM-DIM datamatrix.

PACKED-LEAF uses FEATURE[node] as the second subscript of the datamatrix with bounds checking
off, so an unchecked entry from a corrupt file is a read at an arbitrary offset from the
array. CHECK-PACKABLE applies the same bound when building; the file carries DATUM-DIM so
that the bound survives the round trip.

DATUM-DIM is itself a number from the file, so this check alone would be vacuous against a
corruption that inflated both. It is half of a pair: this bounds FEATURE by the claimed
width, and CHECK-DATAMATRIX-WIDTH bounds the claimed width by the matrix actually handed to
the walk. Composed, feature < datum-dim <= the real column count, and an inflated DATUM-DIM
fails closed at the first prediction rather than reading out of bounds."
  (dotimes (i n-internal)
    (unless (< (aref feature i) datum-dim)
      (error 'packed-load-error
             :detail (format nil "feature[~D] = ~D, outside the datamatrix's [0,~D) columns"
                             i (aref feature i) datum-dim)))))

(defun validate-topology-arrays (n-internal n-leaf n-tree left right roots tree-leaf-offsets)
  "Signal PACKED-LOAD-ERROR if LEFT, RIGHT, ROOTS or TREE-LEAF-OFFSETS are not internally
consistent with the header counts N-INTERNAL, N-LEAF and N-TREE."
  (dotimes (i n-internal)
    (check-child-reference "left" i (aref left i) n-internal n-leaf :parent i)
    (check-child-reference "right" i (aref right i) n-internal n-leaf :parent i))
  ;; A root has no parent to exceed; it only has to be in range. Termination still follows,
  ;; since every step after it moves strictly forward.
  (dotimes (tree n-tree)
    (check-child-reference "roots" tree (aref roots tree) n-internal n-leaf))
  (unless (zerop (aref tree-leaf-offsets 0))
    (error 'packed-load-error
           :detail (format nil "tree-leaf-offsets[0] = ~D, must start at 0"
                           (aref tree-leaf-offsets 0))))
  (let ((previous 0))
    (dotimes (tree n-tree)
      (let ((offset (aref tree-leaf-offsets tree)))
        (when (< offset previous)
          (error 'packed-load-error
                 :detail (format nil "tree-leaf-offsets[~D] = ~D is less than the previous ~
entry ~D" tree offset previous)))
        (unless (or (< offset n-leaf) (and (zerop n-leaf) (zerop offset)))
          (error 'packed-load-error
                 :detail (format nil "tree-leaf-offsets[~D] = ~D, outside [0,~D)"
                                 tree offset n-leaf)))
        (setf previous offset)))))

(defun validate-csr-arrays (n-leaf n-class offsets class probability)
  "Signal PACKED-LOAD-ERROR unless OFFSETS, CLASS and PROBABILITY are a valid CSR partition
of N-LEAF rows over N-CLASS columns."
  (unless (zerop (aref offsets 0))
    (error 'packed-load-error
           :detail (format nil "csr offsets[0] = ~D, must start at 0" (aref offsets 0))))
  (let ((previous (aref offsets 0)))
    (loop for i from 1 to n-leaf
          do (let ((offset (aref offsets i)))
               (when (< offset previous)
                 (error 'packed-load-error
                        :detail (format nil "csr offsets[~D] = ~D is less than the previous ~
entry ~D" i offset previous)))
               (setf previous offset))))
  (let ((nnz (aref offsets n-leaf)))
    (unless (= nnz (length class))
      (error 'packed-load-error
             :detail (format nil "csr offsets[~D] = ~D but class has ~D entries"
                             n-leaf nnz (length class))))
    (unless (= nnz (length probability))
      (error 'packed-load-error
             :detail (format nil "csr offsets[~D] = ~D but probability has ~D entries"
                             n-leaf nnz (length probability)))))
  (dotimes (i (length class))
    (unless (< (aref class i) n-class)
      (error 'packed-load-error
             :detail (format nil "csr class[~D] = ~D, outside [0,~D)"
                             i (aref class i) n-class)))))

;;;; Save and load

(defun packed-save (classifier pathname)
  "Write CLASSIFIER to PATHNAME. Returns PATHNAME.

The format is a header followed by the arrays as little-endian words: WRITE-U32 always
serialises LSB-first and SINGLE-FLOAT-TO-BITS yields an endianness-independent IEEE-754 bit
pattern, so the file is byte-order independent by construction and round-trips identically
on any machine. +BYTE-ORDER-PROBE+ is not there to catch a real mismatch, then -- it is a
second, distinctive magic number, cheap insurance that the header was not merely truncated
or shifted past the point the first magic check looks at."
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
      (write-u32 (packed-topology-datum-dim topology) s)
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
  "Read a packed classifier written by PACKED-SAVE.

Every array is checked against the header counts before this returns: FEATURE against the
saved datamatrix width, LEFT, RIGHT and ROOTS for range and for forward motion, plus
TREE-LEAF-OFFSETS and, for a CSR file, OFFSETS and CLASS. A corrupt file the topology
traversal or the CSR prediction loop would otherwise read or write out of bounds at
(safety 0) -- or walk in a cycle forever -- instead signals PACKED-LOAD-ERROR here. Each
array's declared length is also checked against the bytes actually remaining before it is
allocated, so an inflated count cannot turn into a huge allocation."
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
           (datum-dim (read-u32 s)))
      (check-header-count "n-tree" n-tree :positive t)
      (check-header-count "n-internal" n-internal)
      (check-header-count "n-leaf" n-leaf)
      (check-header-count "n-class" n-class :positive t)
      (check-header-count "datum-dim" datum-dim :positive t)
      (check-array-fits s "feature" n-internal 32)
      (let* ((feature (read-array s n-internal 32 '(unsigned-byte 32) #'identity))
             (threshold (progn (check-array-fits s "threshold" n-internal 32)
                               (read-array s n-internal 32 'single-float
                                           #'bits-to-single-float)))
             (left (progn (check-array-fits s "left" n-internal 32)
                          (read-array s n-internal 32 '(signed-byte 32) #'s32-of-u32)))
             (right (progn (check-array-fits s "right" n-internal 32)
                           (read-array s n-internal 32 '(signed-byte 32) #'s32-of-u32)))
             (roots (progn (check-array-fits s "roots" n-tree 32)
                           (read-array s n-tree 32 '(signed-byte 32) #'s32-of-u32)))
             (tree-leaf-offsets (progn (check-array-fits s "tree-leaf-offsets" n-tree 32)
                                       (read-array s n-tree 32 '(unsigned-byte 32)
                                                   #'identity))))
        (validate-topology-arrays n-internal n-leaf n-tree left right roots
                                  tree-leaf-offsets)
        (validate-feature-array n-internal datum-dim feature)
        (let ((topology
                (%make-packed-topology
                 :n-tree n-tree :n-internal n-internal :n-leaf n-leaf
                 :datum-dim datum-dim
                 :feature feature :threshold threshold :left left :right right
                 :roots roots :tree-leaf-offsets tree-leaf-offsets)))
          (if csr
              (let* ((offsets (progn (check-array-fits s "csr offsets" (1+ n-leaf) 32)
                                     (read-array s (1+ n-leaf) 32 '(unsigned-byte 32)
                                                 #'identity)))
                     (nnz (aref offsets n-leaf))
                     (class (progn (check-array-fits s "csr class" nnz 16)
                                   (read-array s nnz 16 '(unsigned-byte 16) #'identity)))
                     (probability (progn (check-array-fits s "csr probability" nnz 32)
                                         (read-array s nnz 32 'single-float
                                                     #'bits-to-single-float))))
                (validate-csr-arrays n-leaf n-class offsets class probability)
                (%make-packed-classifier
                 :topology topology :n-class n-class :kind :csr
                 :offsets offsets :class class :probability probability))
              (let ((flat (progn
                            (check-array-fits s "dense table" (* (max n-leaf 1) n-class) 32)
                            (read-array s (* (max n-leaf 1) n-class) 32 'single-float
                                        #'bits-to-single-float)))
                    (table (make-array (list (max n-leaf 1) n-class)
                                       :element-type 'single-float)))
                (dotimes (row (max n-leaf 1))
                  (dotimes (k n-class)
                    (setf (aref table row k) (aref flat (+ (* row n-class) k)))))
                (%make-packed-classifier
                 :topology topology :n-class n-class :kind :dense :table table))))))))
