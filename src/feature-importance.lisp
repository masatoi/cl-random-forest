;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

(defun make-oob-sample-indices (total-size sample-indices)
  (let ((bitvec (make-array total-size :element-type 'bit :initial-element 0)))
    (loop for index across sample-indices do
      (setf (aref bitvec index) 1))
    (let* ((len (loop for i from 0 below total-size count (= (aref bitvec i) 0)))
           (oob-sample-indices (make-array len :element-type 'fixnum))
           (i-oob 0))
      (loop for i-bitvec from 0 below total-size do
        (when (= (aref bitvec i-bitvec) 0)
          (setf (aref oob-sample-indices i-oob) i-bitvec)
          (incf i-oob)))
      oob-sample-indices)))

(defun dtree-oob-sample-indices (dtree)
  (make-oob-sample-indices (length (dtree-target dtree))
                           (node-sample-indices (dtree-root dtree))))

(defun test-dtree-oob (dtree datamatrix target
                       &key quiet-p oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array double-float) datamatrix)
           (type (simple-array fixnum (*)) target))
  (let* ((n-correct 0)
         (oob-sample-indices (if (null oob-sample-indices)
                                 (dtree-oob-sample-indices dtree)
                                 oob-sample-indices))
         (len-oob (length oob-sample-indices)))
    (declare (type fixnum n-correct len-oob)
             (type (simple-array fixnum (*)) oob-sample-indices))
    (loop for i fixnum from 0 below len-oob do
      (let ((j (aref oob-sample-indices i)))
        (declare (type fixnum j))
        (when (= (predict-dtree dtree datamatrix j)
                 (aref target j))
          (incf n-correct))))
    (calc-accuracy n-correct len-oob :quiet-p quiet-p)))

(defun find-leaf-randomized (node datamatrix datum-index randomized-attribute oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum datum-index)
           (type (simple-array double-float) datamatrix)
           (type (simple-array fixnum) oob-sample-indices))
  (flet ((random-pick-oob-index ()
           (aref oob-sample-indices (random (length oob-sample-indices)))))
    (cond ((null node) nil)
          ((null (node-test-attribute node)) node)
          (t (let* ((attribute (node-test-attribute node))
                    (threshold (node-test-threshold node))
                    (datum (if (= attribute randomized-attribute)
                               (aref datamatrix (random-pick-oob-index) attribute)
                               (aref datamatrix datum-index attribute))))
               (declare (type fixnum attribute)
                        (type double-float threshold datum))
               (if (>= datum threshold)
                   (find-leaf-randomized (node-left-node node) datamatrix datum-index
                                         randomized-attribute oob-sample-indices)
                   (find-leaf-randomized (node-right-node node) datamatrix datum-index
                                         randomized-attribute oob-sample-indices)))))))

(defun predict-dtree-randomized (dtree datamatrix datum-index randomized-attribute oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array double-float) datamatrix)
           (type fixnum datum-index randomized-attribute)
           (type (simple-array fixnum) oob-sample-indices))
  (let ((max 0d0)
        (max-class 0)
        (dist (node-class-distribution
               (find-leaf-randomized (dtree-root dtree) datamatrix datum-index
                                     randomized-attribute oob-sample-indices)))
        (n-class (dtree-n-class dtree)))
    (declare (type double-float max)
             (type fixnum max-class n-class)
             (type (simple-array double-float) dist))
    (loop for i fixnum from 0 to (1- n-class) do
      (when (> (aref dist i) max)
        (setf max (aref dist i)
              max-class i)))
    max-class))

(defun test-dtree-oob-randomized (dtree datamatrix target randomized-attribute
                                  &key quiet-p oob-sample-indices)
  (declare (optimize (speed 3) (safety 0))
           (type dtree dtree)
           (type (simple-array double-float) datamatrix)
           (type (simple-array fixnum (*)) target))
  (let* ((n-correct 0)
         (oob-sample-indices (if (null oob-sample-indices)
                                 (dtree-oob-sample-indices dtree)
                                 oob-sample-indices))
         (len-oob (length oob-sample-indices)))
    (declare (type fixnum n-correct len-oob)
             (type (simple-array fixnum (*)) oob-sample-indices))
    (loop for i fixnum from 0 below len-oob do
      (let ((j (aref oob-sample-indices i)))
        (declare (type fixnum j))
        (when (= (predict-dtree-randomized dtree datamatrix j randomized-attribute oob-sample-indices)
                 (aref target j))
          (incf n-correct))))
    (calc-accuracy n-correct len-oob :quiet-p quiet-p)))

(defun normalize-arr! (arr)
  (let ((sum (loop for elem across arr sum elem)))
    (loop for i from 0 below (length arr) do
      (setf (aref arr i) (/ (aref arr i) sum)))
    arr))

;; Mean Decrease Accuracy
(defun dtree-feature-importance (dtree datamatrix target)
  (let* ((oob-sample-indices (dtree-oob-sample-indices dtree))
         (accuracy-oob (test-dtree-oob dtree datamatrix target
                                       :quiet-p t :oob-sample-indices oob-sample-indices))
         (result (make-array (dtree-datum-dim dtree) :initial-element 0.0)))
    (loop for i from 0 below (dtree-datum-dim dtree) do
          (setf (aref result i)
                (- accuracy-oob
                   (test-dtree-oob-randomized dtree datamatrix target i :quiet-p t))))
    (normalize-arr! result)))

(defun forest-feature-importance (forest datamatrix target)
  (let* ((len (forest-datum-dim forest))
         (result (make-array len :initial-element 0.0)))
    (dolist (importance-vec
             (mapcar/pmapcar (lambda (dtree)
                               (dtree-feature-importance dtree datamatrix target))
                             (forest-dtree-list forest)))
      (loop for i from 0 below len do
        (incf (aref result i)
              (aref importance-vec i))))
    (loop for i from 0 below len do
      (setf (aref result i) (/ (aref result i) (forest-n-tree forest))))
    result))

;; (forest-feature-importance a9a-forest a9a-datamatrix a9a-target)

;; (clgp:plots
;;  (list
;;   #(0.031556085 0.008169866 0.001642767 0.007866017 0.007471943 0.0027897942
;;     0.0059052594 0.0033697041 0.0024998758 7.4866717e-4 3.3638533e-4 4.687674e-6
;;     0.0 6.991188e-4 -8.654844e-4 1.2483746e-4 -3.8926152e-4 4.2442832e-4
;;     0.031896256 0.004709582 5.9628667e-4 0.028209627 0.004857843 2.0179065e-4
;;     -1.2588486e-4 4.1931937e-4 0.0015418661 1.4535125e-6 0.014708984 -3.835361e-7
;;     0.0012147494 0.0039799083 8.48993e-5 1.466223e-5 0.02978885 0.020701244
;;     0.005981305 0.0011484713 0.13833195 0.1533228 0.0098998565 0.042916078
;;     8.8027684e-4 3.0840503e-4 4.9195496e-6 -3.7968588e-5 0.002799246 0.0033264512
;;     0.007649668 0.001398146 0.029665627 0.027226122 0.0016594284 0.0025693031
;;     0.0019041453 0.0032853417 0.0013028537 9.7295415e-5 5.6518593e-6 0.0
;;     0.016307456 0.020263273 0.081056595 0.02609506 0.0013496294 0.007245719
;;     0.0013111441 4.641826e-6 1.321099e-5 2.4162681e-4 0.001102319 0.025793465
;;     0.015695693 0.047399435 0.039139897 0.008956734 0.009323894 0.024016617
;;     5.832905e-4 0.003970653 0.0010766624 0.016190851 0.001249702 7.5704393e-6
;;     -1.0301604e-4 -2.476394e-6 9.275609e-5 -5.2448297e-5 0.0 3.3868222e-5
;;     -1.4276775e-5 -1.5313253e-5 2.8350893e-5 2.7447799e-5 3.950795e-5
;;     4.7484726e-5 0.0 8.9978595e-5 1.18875156e-4 -6.0210055e-5 1.9715058e-5
;;     5.884098e-5 6.677074e-4 -1.2622298e-5 -2.1929081e-6 -7.235704e-5 1.4052065e-5
;;     -2.821972e-5 3.6381257e-6 -4.6756733e-5 7.895781e-6 1.4020228e-5 2.9800694e-7
;;     -2.3473178e-5 2.7679529e-5 0.0 -4.540316e-6 -2.9208622e-5 2.1442494e-5
;;     -3.274165e-5 4.890406e-6 -1.491582e-5 0.0)
;;   #(0.009099525332182844d0 0.013724475441953703d0 0.016025755904155545d0
;;     0.016343252491679273d0 0.0155351966378741d0 0.016647227936871657d0
;;     0.014131384666540453d0 0.010526602669866221d0 0.010040422253253318d0
;;     0.01258224978299285d0 0.009834448377302175d0 3.7927409341715533d-4
;;     2.0966685563980248d-5 0.015404201959827835d0 0.014586798291288498d0
;;     0.016179390093310587d0 0.015052605662228484d0 0.014942879242448262d0
;;     0.013894567692214522d0 0.012772515069607186d0 0.0074252435103910296d0
;;     0.010551000622108913d0 0.010416113279024076d0 0.012466866697929804d0
;;     0.009734871750523809d0 0.007101559831641318d0 0.0068490956153691605d0
;;     0.0066892348663041366d0 0.010777582975962575d0 0.0024451803679410456d0
;;     0.008618723747241111d0 0.009273379707404985d0 0.0043497791432946355d0
;;     5.495016504533179d-4 0.008196857341761912d0 0.010704288862427445d0
;;     0.012403048167722674d0 0.0105974957700304d0 0.013408090999741098d0
;;     0.017062063570959553d0 0.014295967865819943d0 0.01428195500848231d0
;;     0.008801211321583008d0 0.011585867797042846d0 0.007575410571491906d0
;;     0.003842932867532993d0 0.009021568458517024d0 0.013048324650837401d0
;;     0.007515332698191061d0 0.012058917263300018d0 0.012685585034840414d0
;;     0.013680510042809058d0 0.0048663749150211644d0 0.007187873857657903d0
;;     0.011886215012503398d0 0.007411844935375947d0 0.007959196882444858d0
;;     0.0025679241316400835d0 0.007690957544430464d0 4.568993920835681d-5
;;     0.01677585014073624d0 0.009683790696848526d0 0.011126853276398977d0
;;     0.014116631389350882d0 0.006367521149807252d0 0.013061890879702602d0
;;     0.011922177196405554d0 0.011045943515511313d0 0.006503136807039959d0
;;     0.0038627884605990444d0 0.010920535354217966d0 0.013690143222817799d0
;;     0.011405529745942557d0 0.012882265753526849d0 0.013356223700504037d0
;;     0.009353995410444854d0 0.010329757489260571d0 0.01198883016937037d0
;;     0.011908362154492851d0 0.01659932000512164d0 0.013995284575841404d0
;;     0.015957479271707126d0 0.011655775930168978d0 0.0024126763841930492d0
;;     0.006871507934744008d0 0.003440078729233792d0 0.007103893561058905d0
;;     0.004747461884429789d0 2.705732041014793d-4 0.006076610202503123d0
;;     0.006835264469343019d0 0.005183431078684287d0 0.006305504147809731d0
;;     0.005870090718911699d0 0.004345853046018106d0 0.0027742808425932266d0
;;     2.1392134111443049d-4 0.0064680363954134155d0 0.006310810810352905d0
;;     0.002635951982919743d0 0.002178235227296606d0 0.0041703416441554704d0
;;     0.00519353105899104d0 0.002318969980601779d0 8.284775369820793d-4
;;     0.002965171031778026d0 0.0037251566933290776d0 0.002034859681172897d0
;;     0.0016410888634593285d0 0.0027865594711965177d0 0.003107003527105721d0
;;     0.0011123413387193657d0 0.0018972986956913493d0 0.0025736219078891067d0
;;     0.0028285023400464247d0 3.9293148643346486d-4 8.640807318259196d-4
;;     0.0016739837861215775d0 0.002719113005433924d0 0.0019914348578576745d0
;;     5.188418357285138d-4 7.474093253505034d-4 1.6333820497587898d-6)
;;   '(2.09714878e-02   1.70043900e-02   1.86884186e-02
;;     2.00281880e-02   1.88674696e-02   2.18449889e-02
;;     1.44609045e-02   9.78802874e-03   7.66704443e-03
;;     1.07513505e-02   7.57822963e-03   8.37152019e-05
;;     4.08427984e-06   2.18105776e-02   2.13719788e-02
;;     2.21201286e-02   2.13528341e-02   2.09890353e-02
;;     1.44948906e-02   5.68696688e-03   2.37787047e-03
;;     9.09977162e-03   5.87161773e-03   3.56113989e-03
;;     4.06402276e-03   1.55291681e-03   2.32011130e-03
;;     1.06880599e-03   1.06463082e-02   3.08853690e-04
;;     1.95491269e-03   4.70308238e-03   8.06990665e-04
;;     3.09780186e-05   1.53126607e-02   8.00577270e-03
;;     5.58262608e-03   4.70082610e-03   4.63578869e-02
;;     6.60710697e-02   9.04838621e-03   3.73225583e-02
;;     2.46371961e-03   2.29810276e-03   1.40091657e-03
;;     4.08156346e-04   7.61201127e-03   1.40096656e-02
;;     9.02063804e-03   1.40917478e-02   2.63075600e-02
;;     2.21749227e-02   5.13079050e-03   7.76569484e-03
;;     1.09193164e-02   6.43831797e-03   9.15562348e-03
;;     2.14312332e-04   5.09174651e-03   1.48163435e-05
;;     1.18135326e-02   1.13466117e-02   4.54886708e-02
;;     1.47928515e-02   1.76780986e-03   6.50472516e-03
;;     1.04732973e-02   3.94232221e-03   1.87670965e-03
;;     1.12813310e-03   7.64970204e-03   1.13416278e-02
;;     1.11901607e-02   2.61280659e-02   2.55661042e-02
;;     9.25142527e-03   9.11476076e-03   1.64076591e-02
;;     7.47088664e-03   1.96397699e-02   1.24814595e-02
;;     2.29319246e-02   1.10524890e-02   3.47088726e-04
;;     1.04000194e-03   6.28417960e-04   1.38795152e-03
;;     1.50990614e-03   3.11873199e-05   9.44754619e-04
;;     6.73190114e-04   4.78771956e-04   7.73450811e-04
;;     6.35353846e-04   9.31650273e-04   4.66218437e-04
;;     7.41660529e-06   1.09672486e-03   9.87824531e-04
;;     6.78173455e-04   5.00531693e-04   4.64788921e-04
;;     2.15917098e-03   2.29250211e-04   2.28406346e-04
;;     4.13245058e-04   2.09922803e-04   1.10092848e-04
;;     1.97914541e-04   3.72918413e-04   2.33225325e-04
;;     2.92330549e-04   1.61720833e-04   1.73153592e-04
;;     1.63635233e-04   1.06088474e-04   9.51932589e-05
;;     3.03991973e-04   3.78570532e-04   1.17241105e-04
;;     9.96394264e-05   1.58245551e-04   0.00000000e+00))
;;  :title-list '("clrf(MeanDecreaseAccuracy)"
;;                "clrf(MeanDecreaseEntropy)"
;;                "sklearn(MeanDecreaseGini)"))

;;;;;;;;;;;;;; Mean Decrease Information gain

;; (defparameter *n-class* 4)

;; (defparameter *target*
;;   (make-array 11 :element-type 'fixnum
;;                  :initial-contents '(0 0 1 1 2 2 2 3 3 3 3)))

;; (defparameter *datamatrix*
;;   (make-array '(11 2) 
;;               :element-type 'double-float
;;               :initial-contents '((-1.0d0 -2.0d0)
;;                                   (-2.0d0 -1.0d0)
;;                                   (1.0d0 -2.0d0)
;;                                   (3.0d0 -1.5d0)
;;                                   (-2.0d0 2.0d0)
;;                                   (-3.0d0 1.0d0)
;;                                   (-2.0d0 1.0d0)
;;                                   (3.0d0 2.0d0)
;;                                   (2.0d0 2.0d0)
;;                                   (1.0d0 2.0d0)
;;                                   (1.0d0 1.0d0))))

;; ;; make decision tree
;; (defparameter *dtree*
;;   (make-dtree *n-class* *datamatrix* *target*
;;               :max-depth 5 :min-region-samples 1 :n-trial 10
;;               :remove-sample-indices? nil))

;; (entropy (node-sample-indices (dtree-root *dtree*)) (1- (node-n-sample (dtree-root *dtree*))) *dtree*)

;; (traverse #'node-information-gain (dtree-root *dtree*))
;; (traverse #'node-n-sample (dtree-root *dtree*))

;; (+ (* 6/11 0.6365) (* 5/11 0.6730))

;; (entropy (node-sample-indices (dtree-root adult-dtree)) (1- (node-n-sample (dtree-root adult-dtree))) adult-dtree)
;; (traverse #'node-information-gain (dtree-root adult-dtree))
;; (traverse #'node-n-sample (dtree-root adult-dtree))

(defun dtree-feature-importance-ig (dtree)
  ;; set information gain of root node
  (setf (node-information-gain (dtree-root dtree))
        (entropy (node-sample-indices (dtree-root dtree))
                 (1- (node-n-sample (dtree-root dtree)))
                 dtree))
  (let* ((dim (dtree-datum-dim dtree))
         (acc-arr (clol::make-dvec dim 0d0))
         (cnt-arr (clol::make-dvec dim 0d0)))
    (traverse
     (lambda (node)
       (let ((left (node-left-node node))
             (right (node-right-node node))
             (len (node-n-sample node))
             (attr (node-test-attribute node)))
         (when attr
           (incf (aref acc-arr attr)
                 (- (node-information-gain node)
                    (+ (* (/ (node-n-sample  left) len) (node-information-gain left))
                       (* (/ (node-n-sample right) len) (node-information-gain right)))))
           (incf (aref cnt-arr attr) 1d0))))
     (dtree-root dtree))
    (loop for i from 0 below dim do
      (when (> (aref cnt-arr i) 0d0)
        (setf (aref acc-arr i) (/ (aref acc-arr i) (aref cnt-arr i)))))
    (normalize-arr! acc-arr)))

(defun forest-feature-importance-ig (forest)
  (let* ((len (forest-datum-dim forest))
         (result (make-array len :initial-element 0d0)))
    (dolist (importance-vec
             (mapcar/pmapcar #'dtree-feature-importance-ig (forest-dtree-list forest)))
      (loop for i from 0 below len do
        (incf (aref result i)
              (aref importance-vec i))))
    (loop for i from 0 below len do
      (setf (aref result i) (/ (aref result i) (forest-n-tree forest))))
    result))

;; (clgp:plots
;;  (list 
;;   '(0.06446574 0.020918515 0.0029304882 0.019344205 0.18145977 0.24037899
;;     0.06567786 0.10888982 0.005038251 0.034062333 0.17294767 0.043795574
;;     0.036222443 0.0038683075)

;;   #(0.07758707030914386d0 0.07370961369613219d0 0.09176867585855353d0
;;     0.06939021665218692d0 0.05762127780502801d0 0.06897882043634364d0
;;     0.06805473470067853d0 0.05986245702690493d0 0.058502855120952886d0
;;     0.06511136734975981d0 0.08129830102072307d0 0.08938614692749067d0
;;     0.07571288561327857d0 0.06301557748282345d0)

;;   '(0.15701544  0.03515836  0.17190927  0.02994242  0.0907904 
;;     0.11982955  0.07366134  0.04440293  0.01297517  0.01593711
;;     0.11148862  0.03706486  0.08427554  0.015549)))
