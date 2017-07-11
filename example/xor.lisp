;; -*- coding:utf-8; mode:lisp -*-

(in-package :clrf)

(defparameter *n-class* 2)
(defparameter *n-dim* 2)

;;; XOR (simple)
;; https://github.com/ozt-ca/tjo.hatenablog.samples/blob/master/r_samples/public_lib/jp/xor_simple.txt

(defparameter *target*
  (make-array 100 :element-type 'fixnum
                 :initial-contents
                 '(1 1 0 1 1 0 1 0 1 1 0 0 1 0 0 0 1 0 0 0 0 0 1 1 0 1 1 1 0 0 0 1 1 1 1 1 1 1 0
                   0 1 1 0 0 1 0 0 1 0 1 1 1 1 1 1 0 1 0 1 1 0 0 0 0 1 0 0 1 1 1 1 0 1 0 0 1 0 1
                   0 0 0 0 0 1 0 1 0 1 1 0 0 1 0 0 1 0 0 1 1 0)))

(defparameter *datamatrix*
  (make-array '(100 2) 
              :element-type 'double-float
              :initial-contents
              '((1.7128444910049438d0 -1.1600620746612549d0)
                (1.0938962697982788d0 -1.4356023073196411d0)
                (1.6034027338027954d0 0.6371392607688904d0)
                (-1.1244784593582153d0 1.4549428224563599d0)
                (-1.1737985610961914d0 0.9191796779632568d0)
                (-1.0736390352249146d0 -0.9582682251930237d0)
                (1.255593180656433d0 -0.08454591035842896d0)
                (-0.40516260266304016d0 -1.2746905088424683d0)
                (0.3457210063934326d0 -0.8529095649719238d0)
                (1.7238788604736328d0 -1.006415843963623d0)
                (1.3669304847717285d0 0.2891082465648651d0)
                (1.9628167152404785d0 1.2354862689971924d0)
                (-0.6150732636451721d0 0.9380374550819397d0)
                (0.6018468141555786d0 0.8490278720855713d0)
                (-1.1257253885269165d0 -1.2717514038085938d0)
                (-0.9879453778266907d0 -0.7902625203132629d0)
                (0.5180779099464417d0 -1.474269986152649d0)
                (1.037146806716919d0 1.0346773862838745d0)
                (-0.752565324306488d0 -0.7420031428337097d0)
                (-0.7901754379272461d0 -0.7280303239822388d0)
                (-0.4412689208984375d0 -0.9659240245819092d0)
                (-1.7952001094818115d0 -0.40594401955604553d0)
                (0.3462302088737488d0 -0.41289418935775757d0)
                (-1.049436092376709d0 1.3986499309539795d0)
                (-1.414657711982727d0 -0.7893358469009399d0)
                (1.1932321786880493d0 -1.6879595518112183d0)
                (-0.6416835784912109d0 0.5086838006973267d0)
                (0.832346498966217d0 -0.734119176864624d0)
                (1.096010446548462d0 0.8061819076538086d0)
                (-1.6849360466003418d0 0.021441224962472916d0)
                (1.2335214614868164d0 0.8725093603134155d0)
                (1.1486868858337402d0 -1.3730593919754028d0)
                (0.3496249318122864d0 -0.559252917766571d0)
                (0.4177611470222473d0 -0.40834882855415344d0)
                (-1.1480392217636108d0 1.329789161682129d0)
                (1.973744511604309d0 -0.733721137046814d0)
                (0.5807863473892212d0 -0.43928149342536926d0)
                (-0.7628920674324036d0 1.285264015197754d0)
                (-0.972287654876709d0 -0.42001980543136597d0)
                (1.1186366081237793d0 1.3673853874206543d0)
                (-1.3163546323776245d0 0.6725348830223083d0)
                (-0.9881784319877625d0 0.6318881511688232d0)
                (0.8554965853691101d0 1.1519123315811157d0)
                (0.612370491027832d0 2.0770184993743896d0)
                (-1.0267784595489502d0 0.6009082794189453d0)
                (-1.3884183168411255d0 -0.5292706489562988d0)
                (2.057957649230957d0 1.2295072078704834d0)
                (0.8134425282478333d0 -1.4238463640213013d0)
                (1.142179012298584d0 0.687593936920166d0)
                (0.9569900035858154d0 -0.5604260563850403d0)
                (1.4902722835540771d0 -0.6805275082588196d0)
                (0.5061191320419312d0 -0.3313533663749695d0)
                (1.7864404916763306d0 -1.5994666814804077d0)
                (1.7926682233810425d0 -1.0407146215438843d0)
                (-1.6785632371902466d0 1.027891993522644d0)
                (0.9113888740539551d0 0.9233872294425964d0)
                (-1.1383827924728394d0 1.7807469367980957d0)
                (-0.18319536745548248d0 0.15359532833099365d0)
                (-1.7828983068466187d0 1.8020942211151123d0)
                (-1.1144144535064697d0 2.00636625289917d0)
                (-0.8524760007858276d0 -0.43805041909217834d0)
                (1.1427139043807983d0 1.6416360139846802d0)
                (-0.8282607197761536d0 -0.27918684482574463d0)
                (0.014033292420208454d0 -0.7019109129905701d0)
                (-1.013120174407959d0 0.8187951445579529d0)
                (2.2335970401763916d0 0.6146752834320068d0)
                (0.821603000164032d0 2.0211005210876465d0)
                (-2.1095454692840576d0 1.4203413724899292d0)
                (-0.6772332191467285d0 1.6786290407180786d0)
                (-0.8481736779212952d0 0.5049799680709839d0)
                (-0.7229070067405701d0 1.171151876449585d0)
                (-0.6635993719100952d0 -1.8836901187896729d0)
                (-0.5061119794845581d0 1.7302289009094238d0)
                (1.114445447921753d0 0.4358004629611969d0)
                (1.3131698369979858d0 1.508039951324463d0)
                (1.4985765218734741d0 -1.3253287076950073d0)
                (1.6223644018173218d0 1.1848437786102295d0)
                (0.11975878477096558d0 -1.6456096172332764d0)
                (-1.0115238428115845d0 -0.7775087356567383d0)
                (0.5820426940917969d0 0.6848682165145874d0)
                (-1.0235430002212524d0 -1.1069453954696655d0)
                (-1.7597886323928833d0 -0.9781032800674438d0)
                (1.2666147947311401d0 1.2910274267196655d0)
                (-1.4970636367797852d0 2.1419179439544678d0)
                (0.8203398585319519d0 0.7451744675636292d0)
                (0.17994721233844757d0 -0.588549017906189d0)
                (-0.49248629808425903d0 -1.1635098457336426d0)
                (-0.19113850593566895d0 1.228506088256836d0)
                (-0.23580928146839142d0 1.4071252346038818d0)
                (0.9128731489181519d0 0.5728715062141418d0)
                (-1.0279027223587036d0 -1.5263631343841553d0)
                (1.228339672088623d0 -1.0910688638687134d0)
                (0.6380418539047241d0 2.075192451477051d0)
                (-0.7838084101676941d0 -0.9291013479232788d0)
                (-0.9204593896865845d0 1.0430328845977783d0)
                (-1.1959508657455444d0 -1.6891578435897827d0)
                (-0.9482574462890625d0 -1.036778211593628d0)
                (-1.1475433111190796d0 0.9125188589096069d0)
                (0.5465568900108337d0 -0.3704013228416443d0)
                (1.1223702430725098d0 2.0434348583221436d0))))

;;; XOR (complex)
;; https://github.com/ozt-ca/tjo.hatenablog.samples/blob/master/r_samples/public_lib/jp/xor_complex.txt

(defparameter *target-complex*
  (make-array 100 :element-type 'fixnum
                 :initial-contents
                 '(1 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 0 0 1 1 0 1 1 1 0 0 0 0 0 0 1 0
                   0 1 1 0 0 1 0 1 1 0 1 1 1 0 1 0 1 0 0 1 0 0 0 0 0 0 1 1 0 1 0 1 0 0 1 1 1 0 0
                   0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 1 1 1)))

(defparameter *datamatrix-complex*
  (make-array '(100 2)
              :element-type 'double-float
              :initial-contents '((0.4034355580806732d0 -2.4056382179260254d0)
                                  (-0.3767566680908203d0 -0.7262629270553589d0)
                                  (-2.2985403537750244d0 -1.5229682922363281d0)
                                  (0.5283093452453613d0 0.11962219327688217d0)
                                  (-2.2548398971557617d0 1.6154886484146118d0)
                                  (0.007411253172904253d0 -2.24187970161438d0)
                                  (2.2148327827453613d0 1.095436930656433d0)
                                  (1.6204214096069336d0 3.003624200820923d0)
                                  (-0.39863452315330505d0 -1.4344172477722168d0)
                                  (0.482808381319046d0 1.1411314010620117d0)
                                  (2.476661443710327d0 0.9367761015892029d0)
                                  (-0.8093363642692566d0 -0.9158605337142944d0)
                                  (-0.8420962691307068d0 2.2892391681671143d0)
                                  (-0.3557377755641937d0 1.5140557289123535d0)
                                  (-0.5381374359130859d0 -1.0172654390335083d0)
                                  (-0.6910051107406616d0 0.34726300835609436d0)
                                  (-0.4224027395248413d0 -2.2020320892333984d0)
                                  (-1.496415138244629d0 0.21315909922122955d0)
                                  (-0.18666408956050873d0 -1.8733288049697876d0)
                                  (-0.10248524695634842d0 -1.74650239944458d0)
                                  (0.2099926918745041d0 1.4443750381469727d0)
                                  (0.4747675061225891d0 -0.8544942736625671d0)
                                  (-2.1754510402679443d0 2.774359941482544d0)
                                  (0.4858465790748596d0 -0.7100752592086792d0)
                                  (-0.15415449440479279d0 1.207557201385498d0)
                                  (1.386326551437378d0 -1.0547404289245605d0)
                                  (0.9578530192375183d0 -0.5550774335861206d0)
                                  (2.109900712966919d0 2.615283727645874d0)
                                  (-0.7024237513542175d0 1.4116462469100952d0)
                                  (0.05270323157310486d0 -0.6144914031028748d0)
                                  (-1.298715353012085d0 -0.4161319434642792d0)
                                  (0.5599372386932373d0 0.8047501444816589d0)
                                  (0.6765029430389404d0 1.3647483587265015d0)
                                  (-0.46489202976226807d0 -1.8299870491027832d0)
                                  (-1.143085241317749d0 -1.1070141792297363d0)
                                  (-1.4176253080368042d0 -3.050014019012451d0)
                                  (1.3958582878112793d0 0.7617706060409546d0)
                                  (-0.8436087965965271d0 1.6674678325653076d0)
                                  (1.4047787189483643d0 -0.09963632375001907d0)
                                  (-1.5570333003997803d0 -0.12685704231262207d0)
                                  (1.8137458562850952d0 -1.425581693649292d0)
                                  (0.07024648785591125d0 -0.40010830760002136d0)
                                  (-0.7535247206687927d0 0.3371486961841583d0)
                                  (0.6725790500640869d0 -1.2861484289169312d0)
                                  (0.9243353605270386d0 -1.066495418548584d0)
                                  (-1.693636178970337d0 -2.052839756011963d0)
                                  (2.9208388328552246d0 -2.773191213607788d0)
                                  (0.42294222116470337d0 -1.0445746183395386d0)
                                  (1.1460599899291992d0 1.1797559261322021d0)
                                  (1.156691551208496d0 -2.17287540435791d0)
                                  (-0.9446756839752197d0 1.6630737781524658d0)
                                  (-1.2794888019561768d0 1.311888575553894d0)
                                  (-1.3294799327850342d0 -0.23829297721385956d0)
                                  (3.0060367584228516d0 -0.529233992099762d0)
                                  (1.1789709329605103d0 3.10229754447937d0)
                                  (-0.4477991461753845d0 0.4690931737422943d0)
                                  (-0.1235247254371643d0 -1.447838306427002d0)
                                  (-1.22459876537323d0 0.003981177695095539d0)
                                  (-1.8135451078414917d0 -0.6236857175827026d0)
                                  (-0.6252175569534302d0 0.052105024456977844d0)
                                  (1.4340219497680664d0 0.7852487564086914d0)
                                  (-2.032773494720459d0 -0.6313832402229309d0)
                                  (0.024456562474370003d0 -1.275212287902832d0)
                                  (1.2997491359710693d0 0.9445179104804993d0)
                                  (1.8310132026672363d0 2.8142433166503906d0)
                                  (-0.6088464260101318d0 0.4578162729740143d0)
                                  (0.5679804086685181d0 -3.1717369556427d0)
                                  (-0.8697279095649719d0 3.0944631099700928d0)
                                  (-1.5808067321777344d0 1.0526114702224731d0)
                                  (1.0054080486297607d0 0.787286639213562d0)
                                  (-0.5810078978538513d0 0.5414748787879944d0)
                                  (-1.3692512512207031d0 -0.21298255026340485d0)
                                  (0.6964770555496216d0 -0.5327919721603394d0)
                                  (0.9881920218467712d0 -0.9081218838691711d0)
                                  (0.1264992356300354d0 0.7935831546783447d0)
                                  (0.8089390397071838d0 -0.26789039373397827d0)
                                  (0.11994863301515579d0 0.5057891011238098d0)
                                  (1.0870332717895508d0 -0.36434054374694824d0)
                                  (-0.7657254338264465d0 -1.0628474950790405d0)
                                  (0.821785569190979d0 -0.04947595298290253d0)
                                  (1.3866432905197144d0 -0.1587466448545456d0)
                                  (-0.2793336510658264d0 -0.21237485110759735d0)
                                  (1.9732534885406494d0 -1.0063868761062622d0)
                                  (-0.5944344997406006d0 0.43110284209251404d0)
                                  (-1.8236985206604004d0 2.472883462905884d0)
                                  (-1.1346434354782104d0 1.2679016590118408d0)
                                  (-1.25001859664917d0 2.1546547412872314d0)
                                  (-0.10484754294157028d0 -0.42037391662597656d0)
                                  (-0.6731264591217041d0 0.1716737002134323d0)
                                  (1.5008915662765503d0 -1.4382739067077637d0)
                                  (-2.227886915206909d0 0.044186417013406754d0)
                                  (1.882254958152771d0 -1.378435492515564d0)
                                  (1.1181504726409912d0 -1.548943042755127d0)
                                  (1.1032636165618896d0 0.7330594062805176d0)
                                  (1.5704227685928345d0 1.0558465719223022d0)
                                  (-0.99101322889328d0 0.8037406802177429d0)
                                  (1.85750150680542d0 0.497977077960968d0)
                                  (1.5727083683013916d0 -2.015207052230835d0)
                                  (-1.0054093599319458d0 2.383617877960205d0)
                                  (-2.121652603149414d0 1.304545283317566d0))))
  
;;; make random forest
(defparameter *forest*
  (make-forest *n-class* *n-dim* *datamatrix* *target* :n-tree 100 :bagging-ratio 1.0 :max-depth 100))

;;; test random forest
(test-forest *forest* *datamatrix* *target*)

;;; plot decision boundary
(ql:quickload :clgplot)

(defun make-predict-matrix (n x0 xn y0 yn forest datamatrix target)
  (let ((x-span (/ (- xn x0) (1- n)))
        (y-span (/ (- yn y0) (1- n)))
        (mesh-predicted (make-array (list n n)))
        (tmp-datamatrix (make-array '(1 2) :element-type 'double-float)))
    ;; mark prediction
    (loop for i from 0 to (1- n)
          for x from x0 by x-span do
            (loop for j from 0 to (1- n)
                  for y from y0 by y-span do
                    (setf (aref tmp-datamatrix 0 0) x
                          (aref tmp-datamatrix 0 1) y)
                    (setf (aref mesh-predicted i j)
                          (if (> (predict-forest forest tmp-datamatrix 0) 0)
                              0.25 -0.25))))
    ;; mark datapoints
    (let* ((range (clgp:seq 0 (1- n)))
           (x-grid (mapcar (lambda (x) (+ (* x x-span) x0)) range))
           (y-grid (mapcar (lambda (y) (+ (* y y-span) y0)) range)))
      (loop for i from 0 to (1- (length target)) do
        (let ((data-i (position-if (lambda (x) (<= (aref datamatrix i 0) x)) x-grid))
              (data-j (position-if (lambda (y) (<= (aref datamatrix i 1) y)) y-grid)))
          (if (and data-i data-j)
              (setf (aref mesh-predicted data-i data-j)
                    (if (> (aref target i) 0) 1 -1))))))
    mesh-predicted))

(defparameter predict-matrix
  (make-predict-matrix 100 -3.5d0 3.5d0 -3.5d0 3.5d0 *forest* *datamatrix* *target*))

(clgp:splot-matrix predict-matrix)

;;; make refine learner
(defparameter *forest-learner* (make-refine-learner *forest*))
(defparameter *forest-refine-dataset* (make-refine-dataset *forest* *datamatrix*))

(train-refine-learner *forest-learner* *forest-refine-dataset* *target*)
(test-refine-learner  *forest-learner* *forest-refine-dataset* *target*)

;;; plot decision boundary (refine)

(defun make-refine-predict-matrix (n x0 xn y0 yn forest forest-learner datamatrix target)
  (let ((x-span (/ (- xn x0) (1- n)))
        (y-span (/ (- yn y0) (1- n)))
        (mesh-predicted (make-array (list n n)))
        (tmp-datamatrix (make-array '(1 2) :element-type 'double-float)))
    ;; mark prediction
    (loop for i from 0 to (1- n)
          for x from x0 by x-span do
            (loop for j from 0 to (1- n)
                  for y from y0 by y-span do
                    (setf (aref tmp-datamatrix 0 0) x
                          (aref tmp-datamatrix 0 1) y)
                    (setf (aref mesh-predicted i j)
                          (if (> (predict-refine-learner forest forest-learner tmp-datamatrix 0) 0)
                              0.25 -0.25))))
    ;; mark datapoints
    (let* ((range (clgp:seq 0 (1- n)))
           (x-grid (mapcar (lambda (x) (+ (* x x-span) x0)) range))
           (y-grid (mapcar (lambda (y) (+ (* y y-span) y0)) range)))
      (loop for i from 0 to (1- (length target)) do
        (let ((data-i (position-if (lambda (x) (<= (aref datamatrix i 0) x)) x-grid))
              (data-j (position-if (lambda (y) (<= (aref datamatrix i 1) y)) y-grid)))
          (if (and data-i data-j)
              (setf (aref mesh-predicted data-i data-j)
                    (if (> (aref target i) 0) 1 -1))))))
    mesh-predicted))

(defparameter refine-predict-matrix
  (make-refine-predict-matrix 100 -3.5d0 3.5d0 -3.5d0 3.5d0
                              *forest* *forest-learner* *datamatrix* *target*))

(clgp:splot-matrix refine-predict-matrix)


;;; AROW

(defparameter arow-dataset
  (loop for i from 0 to (1- (length *target*))
        collect (cons (coerce (aref *target* i) 'double-float)
                      (make-array 2 :element-type 'double-float
                                  :initial-contents (list (aref *datamatrix* i 0)
                                                          (aref *datamatrix* i 1))))))

(defparameter arow-learner (clol:make-arow 2 10d0))
(clol:train arow-learner arow-dataset)
(clol:test arow-learner arow-dataset)

(defun make-arow-predict-matrix (n x0 xn y0 yn learner datamatrix target)
  (let ((mesh (make-array (list (* n n) 2) :element-type 'double-float))
        (x-span (/ (- xn x0) (1- n)))
        (y-span (/ (- yn y0) (1- n))))
    (loop for i from 0 to (1- n)
          for x from x0 by x-span do
            (loop for j from 0 to (1- n)
                  for y from y0 by y-span do
                    (setf (aref mesh (+ (* i n) j) 0) x
                          (aref mesh (+ (* i n) j) 1) y)))
    (let ((mesh-predicted (make-array (list n n))))
      ;; mark prediction
      (loop for i from 0 to (1- n) do
        (loop for j from 0 to (1- n) do
          (setf (aref mesh-predicted i j)
                (if (> (clol:arow-predict learner
                                          (make-array 2 :element-type 'double-float
                                                      :initial-contents
                                                      (list (aref mesh (+ (* i n) j) 0)
                                                            (aref mesh (+ (* i n) j) 1))))
                       0)
                    0.25 -0.25))))
      ;; mark datapoints
      (let* ((range (clgp:seq 0 (1- n)))
             (x-grid (mapcar (lambda (x) (+ (* x x-span) x0)) range))
             (y-grid (mapcar (lambda (y) (+ (* y y-span) y0)) range)))
        (loop for i from 0 to (1- (length target)) do
          (let ((data-i (position-if (lambda (x) (<= (aref datamatrix i 0) x)) x-grid))
                (data-j (position-if (lambda (y) (<= (aref datamatrix i 1) y)) y-grid)))
            (if (and data-i data-j)
                (setf (aref mesh-predicted data-i data-j)
                      (if (> (aref target i) 0) 1 -1))))))
      mesh-predicted)))

(defparameter arow-predict-matrix
  (make-arow-predict-matrix 100 -3d0 3d0 -3d0 3d0 arow-learner *datamatrix* *target*))

(clgp:splot-matrix arow-predict-matrix)
