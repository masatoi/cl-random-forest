# prove → rove テストフレームワーク移行 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** テストスイートを prove から rove に移行し、cl-mcp の `run-tests` から機能単位・テスト単位で選択実行できるようにする。

**Architecture:** テストを機能軸で5つの ASDF システム（dataset / decision-tree / forest / refinement / parallel）と共有 fixture システムに分割する。rove の `run-system` は plain `defsystem` に対して「そのシステム自身のコンポーネントファイルに登録されたパッケージのスイート」だけを走らせるため、機能ごとの独立実行が成立する。データセット読み込みは遅延メモ化し、`lparallel:*kernel*` は `let` でスコープ化する。

**Tech Stack:** Common Lisp (SBCL / CCL), ASDF, rove 0.10.0, lparallel, cl-online-learning, trivial-garbage, Roswell

設計書: `docs/superpowers/specs/2026-08-01-prove-to-rove-migration-design.md`

## Global Constraints

- **期待精度・反復回数・許容誤差を一切変更しない。** 移行後に値がずれたらそれは移行のバグ。
  許容誤差は `approximately-equal` の既定 `delta` = `1d0`。
- **新規テストを追加しない。** 旧 13 アサーション（`ok` 3個 + `is` 10個）を 13 個の `deftest` に 1:1 移植する。
- **テスト側の ASDF システムは plain `defsystem` を維持する。** `package-inferred-system` にすると
  rove が依存システムのスイートまで走らせてしまい、機能分離が壊れる。
- **1ファイル = 1パッケージ = 1 rove スイート。** rove はファイルとパッケージの対応を
  `deftest` 評価時の `*load-pathname*` で記録するため、2ファイルが同じパッケージを使うと分離が壊れる。
- **`lparallel:*kernel*` をトップレベルで `setf` しない。** 必ず `with-serial-kernel` /
  `with-parallel-kernel` で束縛する。
- **並列テストは `#+sbcl` を維持する。** CCL では並列テストを走らせない現状の挙動を保つ。
- 作業ブランチは `prove-to-rove`（作成済み）。

## ファイル構成

| ファイル | 責務 | タスク |
|---|---|---|
| `cl-random-forest-test.asd` | 7つの defsystem（fixture / 5機能 / 集約）| 1 |
| `t/fixture.lisp` | データセットDL・遅延ロード・アサーションヘルパ・カーネルスコープ。**テストを持たない** | 1 |
| `t/dataset.lisp` | データ取得と変換のテスト 3本 | 1 |
| `t/decision-tree.lisp` | 決定木の精度テスト 2本 | 2 |
| `t/forest.lisp` | フォレストの精度テスト 2本 | 3 |
| `t/refinement.lisp` | Global Refinement の精度テスト 2本 | 4 |
| `t/parallel.lisp` | 並列実行の精度テスト 4本（`#+sbcl`）| 5 |
| `t/cl-random-forest.lisp` | **削除**（移植元。タスク5まで参照用に残す）| 6 |
| `.github/workflows/ci.yml` | `run-prove` → `rove` | 6 |
| `CLAUDE.md` | Commands 節の更新 | 6 |

## 検証に使うコマンド

すべて `ros -Q dynamic-space-size=2048` で起動した SBCL の REPL で実行する。

**構造チェック（速い。テストを実行せずスイートの割り当てだけ見る）:**

```lisp
;; どのスイートが対象になるか
(mapcar #'rove/core/suite/package:suite-name
        (rove/core/suite/package:system-suites
         (asdf:find-system "cl-random-forest-test/decision-tree")))

;; スイートに登録されたテスト名
(rove/core/suite/package:suite-tests
 (rove:find-suite :cl-random-forest-test/decision-tree))
```

**実行:**

```lisp
(asdf:load-system :cl-random-forest-test/decision-tree)
(rove:run :cl-random-forest-test/decision-tree)
```

cl-mcp を使う場合は `run-tests` に `system` / `test` を渡す。

---

### Task 1: .asd 全面書き換え + fixture + dataset スイート

このタスクだけでアーキテクチャ全体（rove 配線・遅延ロード・スイート分離）が検証できる。
最もリスクが高い部分なので最初に片付ける。

**Files:**
- Modify: `cl-random-forest-test.asd`（全面書き換え）
- Create: `t/fixture.lisp`
- Create: `t/dataset.lisp`
- Create: `t/decision-tree.lisp`（`defpackage` のみのスタブ）
- Create: `t/forest.lisp`（スタブ）
- Create: `t/refinement.lisp`（スタブ）
- Create: `t/parallel.lisp`（スタブ）

**Interfaces:**
- Produces（タスク2〜5が使う）:
  - `cl-random-forest-test/fixture` パッケージ。export するシンボル:
    - `*dataset-dir*` — `dataset/` ディレクトリの pathname
    - `+a9a-dim+` = 123, `+letter-dim+` = 16, `+letter-n-class+` = 26
    - `(fetch-a9a)` / `(fetch-letter)` → 副作用のみ
    - `(a9a-train)` / `(a9a-test)` / `(letter-train)` / `(letter-test)` → `(values datamatrix target)`
    - `(approximately-equal x y &optional (delta 1d0))` → boolean
    - `(n-times-average n-times &body body)` マクロ → `double-float`
    - `(with-serial-kernel &body body)` マクロ
    - `(with-parallel-kernel (&optional (n 4)) &body body)` マクロ
  - ASDF システム `cl-random-forest-test/decision-tree`, `/forest`, `/refinement`, `/parallel`
    （中身はタスク2〜5で埋める）

- [ ] **Step 1: `cl-random-forest-test.asd` を全面書き換え**

旧ファイルの `defpackage cl-random-forest-test-asd` 方式はやめ、ASDF 標準の `asdf-user`
パッケージで読まれる素の `defsystem` 形式にする（`symbol-call` は `asdf-user` が `uiop` を
use しているのでそのまま使える）。

```lisp
#|
  This file is a part of cl-random-forest project.
|#

(defsystem "cl-random-forest-test/fixture"
  :author "Satoshi Imai"
  :license "MIT Licence"
  :description "Shared fixtures for the cl-random-forest test suites. Contains no tests."
  :depends-on ("cl-random-forest" "cl-online-learning" "uiop" "trivial-garbage" "lparallel")
  :components ((:module "t" :components ((:file "fixture")))))

(defsystem "cl-random-forest-test/dataset"
  :description "Tests for dataset download and conversion to datamatrix/target"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "dataset"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/decision-tree"
  :description "Accuracy tests for decision trees"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "decision-tree"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/forest"
  :description "Accuracy tests for random forests"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "forest"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/refinement"
  :description "Accuracy tests for global refinement"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "refinement"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test/parallel"
  :description "Accuracy tests for parallelized training (SBCL only)"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "parallel"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))

(defsystem "cl-random-forest-test"
  :author "Satoshi Imai"
  :license "MIT Licence"
  :description "Test system for cl-random-forest"
  :depends-on ("rove"
               "cl-random-forest-test/dataset"
               "cl-random-forest-test/decision-tree"
               "cl-random-forest-test/forest"
               "cl-random-forest-test/refinement"
               "cl-random-forest-test/parallel")
  ;; NOTE: This system has no components of its own, so rove's SYSTEM-SUITES would
  ;; return an empty list and (rove:run c) would silently report zero tests.
  ;; The feature systems must be listed explicitly.
  :perform (test-op (o c)
             (declare (ignore o c))
             (unless (symbol-call :rove :run
                                  '("cl-random-forest-test/dataset"
                                    "cl-random-forest-test/decision-tree"
                                    "cl-random-forest-test/forest"
                                    "cl-random-forest-test/refinement"
                                    "cl-random-forest-test/parallel"))
               (error "Tests failed."))))
```

- [ ] **Step 2: `t/fixture.lisp` を作成**

```lisp
(in-package :cl-user)

(defpackage cl-random-forest-test/fixture
  (:use :cl)
  (:import-from :cl-random-forest/src/utils
                :clol-dataset->datamatrix/target)
  (:export :*dataset-dir*
           :+a9a-dim+
           :+letter-dim+
           :+letter-n-class+
           :fetch-a9a
           :fetch-letter
           :a9a-train
           :a9a-test
           :letter-train
           :letter-test
           :approximately-equal
           :n-times-average
           :with-serial-kernel
           :with-parallel-kernel))
(in-package :cl-random-forest-test/fixture)

;;;; Dataset location

(defparameter *dataset-dir*
  (merge-pathnames #P"dataset/" (asdf:system-source-directory :cl-random-forest))
  "Directory the LIBSVM-format test datasets are downloaded into.")

(ensure-directories-exist *dataset-dir*)

(defconstant +a9a-dim+ 123)
(defconstant +letter-dim+ 16)
(defconstant +letter-n-class+ 26)

;;;; Pathname helpers for wget

(defun cat (&rest args)
  (apply #'concatenate 'string args))

(defun format-directory (p)
  (assert (eq (car (pathname-directory p)) :absolute))
  (reduce (lambda (a b) (cat a b "/"))
          (cons "/" (cdr (pathname-directory p)))))

(defun format-filename (p)
  (if (pathname-type p)
      (format nil "~A.~A" (pathname-name p) (pathname-type p))
      (format nil "~A"    (pathname-name p))))

(defun format-pathname (p)
  (let ((filename (format-filename p)))
    (if filename
        (cat (format-directory p) (format-filename p))
        (format-directory p))))

;;;; Dataset download

(defun fetch-file (url pathname)
  "Download URL to PATHNAME with wget unless PATHNAME already exists."
  (unless (uiop:file-exists-p pathname)
    (uiop:run-program (list "wget" url "-O" (format-pathname pathname)))))

(defun fetch-a9a ()
  "Download the a9a dataset into *DATASET-DIR* unless it is already there."
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a"
              (merge-pathnames "a9a" *dataset-dir*))
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t"
              (merge-pathnames "a9a.t" *dataset-dir*)))

(defun fetch-letter ()
  "Download the letter dataset into *DATASET-DIR* unless it is already there."
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale"
              (merge-pathnames "letter.scale" *dataset-dir*))
  (fetch-file "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/letter.scale.t"
              (merge-pathnames "letter.scale.t" *dataset-dir*)))

;;;; Lazily loaded datasets
;;;
;;; Each accessor returns (values datamatrix target) and caches the result, so a
;;; single test can be run on its own and still get the data it needs.

(defvar *a9a-train-cache* nil)
(defvar *a9a-test-cache* nil)
(defvar *letter-train-cache* nil)
(defvar *letter-test-cache* nil)

(defun load-a9a (filename)
  "Read FILENAME from *DATASET-DIR* as a9a and return (values datamatrix target).
The a9a labels are +1/-1; they are remapped to class ids 0/1."
  (let ((data (clol.utils:read-data (merge-pathnames filename *dataset-dir*) +a9a-dim+)))
    (dolist (datum data)
      (setf (car datum) (if (> (car datum) 0d0) 0 1)))
    (clol-dataset->datamatrix/target data)))

(defun load-letter (filename)
  "Read FILENAME from *DATASET-DIR* as letter and return (values datamatrix target)."
  (let ((data (clol.utils:read-data (merge-pathnames filename *dataset-dir*)
                                    +letter-dim+ :multiclass-p t)))
    (clol-dataset->datamatrix/target data)))

(defun a9a-train ()
  "Return (values datamatrix target) for the a9a training set, loading it on first call."
  (unless *a9a-train-cache*
    (fetch-a9a)
    (multiple-value-bind (datamatrix target) (load-a9a "a9a")
      (setf *a9a-train-cache* (cons datamatrix target))))
  (values (car *a9a-train-cache*) (cdr *a9a-train-cache*)))

(defun a9a-test ()
  "Return (values datamatrix target) for the a9a test set, loading it on first call."
  (unless *a9a-test-cache*
    (fetch-a9a)
    (multiple-value-bind (datamatrix target) (load-a9a "a9a.t")
      (setf *a9a-test-cache* (cons datamatrix target))))
  (values (car *a9a-test-cache*) (cdr *a9a-test-cache*)))

(defun letter-train ()
  "Return (values datamatrix target) for the letter training set, loading it on first call."
  (unless *letter-train-cache*
    (fetch-letter)
    (multiple-value-bind (datamatrix target) (load-letter "letter.scale")
      (setf *letter-train-cache* (cons datamatrix target))))
  (values (car *letter-train-cache*) (cdr *letter-train-cache*)))

(defun letter-test ()
  "Return (values datamatrix target) for the letter test set, loading it on first call."
  (unless *letter-test-cache*
    (fetch-letter)
    (multiple-value-bind (datamatrix target) (load-letter "letter.scale.t")
      (setf *letter-test-cache* (cons datamatrix target))))
  (values (car *letter-test-cache*) (cdr *letter-test-cache*)))

;;;; Assertion helpers

(defun approximately-equal (x y &optional (delta 1d0))
  "Return true when X and Y are within DELTA. X may be a double-float, vector or list."
  (flet ((andf (x y) (and x y))
         (close? (x y) (< (abs (- x y)) delta)))
    (etypecase x
      (double-float (close? x y))
      (vector (reduce #'andf (map 'vector #'close? x y)))
      (list (reduce #'andf (mapcar #'close? x y))))))

(defmacro n-times-average (n-times &body body)
  "Evaluate BODY N-TIMES and return the mean of its values as a double-float."
  `(coerce (/ (loop repeat ,n-times
                    sum (progn ,@body))
              ,n-times)
           'double-float))

;;;; lparallel kernel scoping
;;;
;;; The old test file setf'd lparallel:*kernel* at toplevel and never called
;;; END-KERNEL, leaking a 4-worker kernel per parallel test and leaving the
;;; kernel set when the file finished. These macros bind it instead, so each
;;; test is independent of the order the suites run in.

(defmacro with-serial-kernel (&body body)
  "Evaluate BODY with lparallel:*kernel* bound to NIL, forcing serial execution."
  `(let ((lparallel:*kernel* nil))
     ,@body))

(defmacro with-parallel-kernel ((&optional (n 4)) &body body)
  "Evaluate BODY with lparallel:*kernel* bound to a fresh N-worker kernel, then shut it down."
  `(let ((lparallel:*kernel* (lparallel:make-kernel ,n)))
     (unwind-protect (progn ,@body)
       (lparallel:end-kernel :wait t))))
```

- [ ] **Step 3: `t/dataset.lisp` を作成**

```lisp
(in-package :cl-user)

(defpackage cl-random-forest-test/dataset
  (:use :cl :rove :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/dataset)

(deftest dataset-files-exist
  ;; Nothing else fetches eagerly any more, so this test does it itself.
  (fetch-a9a)
  (fetch-letter)
  (ok (and (uiop:file-exists-p (merge-pathnames "letter.scale" *dataset-dir*))
           (uiop:file-exists-p (merge-pathnames "letter.scale.t" *dataset-dir*))
           (uiop:file-exists-p (merge-pathnames "a9a" *dataset-dir*))
           (uiop:file-exists-p (merge-pathnames "a9a.t" *dataset-dir*)))
      "all four dataset files exist in *dataset-dir*"))

(deftest a9a-dataset-load
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (ok (and datamatrix target datamatrix-test target-test)
          "a9a train and test sets load into datamatrix/target"))))

(deftest letter-dataset-load
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (ok (and datamatrix target datamatrix-test target-test)
          "letter train and test sets load into datamatrix/target"))))
```

- [ ] **Step 4: 4つのスタブファイルを作成**

タスク2〜5で中身を埋める。`.asd` がこれらのファイルを参照しているので、今作らないと
`cl-random-forest-test` がロードできない。

`t/decision-tree.lisp`:

```lisp
(in-package :cl-user)

(defpackage cl-random-forest-test/decision-tree
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/decision-tree)
```

`t/forest.lisp`:

```lisp
(in-package :cl-user)

(defpackage cl-random-forest-test/forest
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/forest)
```

`t/refinement.lisp`:

```lisp
(in-package :cl-user)

(defpackage cl-random-forest-test/refinement
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/refinement)
```

`t/parallel.lisp`:

```lisp
(in-package :cl-user)

(defpackage cl-random-forest-test/parallel
  (:use :cl :rove :cl-random-forest :cl-random-forest-test/fixture))
(in-package :cl-random-forest-test/parallel)
```

- [ ] **Step 5: ロードして構造チェック**

`ros -Q dynamic-space-size=2048` で SBCL を起動して:

```lisp
(asdf:load-system :cl-random-forest-test)

;; dataset システムは自分のスイートだけを対象にする
(mapcar #'rove/core/suite/package:suite-name
        (rove/core/suite/package:system-suites
         (asdf:find-system "cl-random-forest-test/dataset")))
```

Expected: `("cl-random-forest-test/dataset")`
— fixture や他機能のスイートが混ざっていないこと。これが設計の要。

```lisp
(rove/core/suite/package:suite-tests
 (rove:find-suite :cl-random-forest-test/dataset))
```

Expected: 3つのシンボル `(DATASET-FILES-EXIST A9A-DATASET-LOAD LETTER-DATASET-LOAD)`

```lisp
;; fixture はテストを持たないので対象スイートが空
(rove/core/suite/package:system-suites
 (asdf:find-system "cl-random-forest-test/fixture"))
```

Expected: `NIL`

- [ ] **Step 6: dataset スイートを実行**

```lisp
(rove:run :cl-random-forest-test/dataset)
```

Expected: 3テストすべて pass、戻り値の第1値が `T`。
初回はデータセットのDLが走るので数分かかる（`wget` が `PATH` に必要）。

- [ ] **Step 7: 個別テスト実行を確認**

```lisp
(rove:run-test 'cl-random-forest-test/dataset::a9a-dataset-load)
```

Expected: 1テストだけ実行され pass。`letter-dataset-load` は走らない。

- [ ] **Step 8: コミット**

```bash
git add cl-random-forest-test.asd t/fixture.lisp t/dataset.lisp \
        t/decision-tree.lisp t/forest.lisp t/refinement.lisp t/parallel.lisp
git commit -m "Restructure test system around rove feature suites

Rewrite cl-random-forest-test.asd as a fixture system plus five feature
systems and an aggregate. Add t/fixture.lisp with lazily memoized dataset
accessors and lparallel kernel scoping macros, and t/dataset.lisp with the
three dataset tests ported to rove.

The remaining four feature files are package stubs for now."
```

---

### Task 2: decision-tree スイート

**Files:**
- Modify: `t/decision-tree.lisp`（スタブに2つの `deftest` を追加）

**Interfaces:**
- Consumes: Task 1 の `cl-random-forest-test/fixture`（`a9a-train`, `a9a-test`, `letter-train`,
  `letter-test`, `n-times-average`, `approximately-equal`, `+letter-n-class+`）
- Produces: なし（他タスクはこのファイルに依存しない）

- [ ] **Step 1: `t/decision-tree.lisp` に2つの deftest を追加**

`(in-package :cl-random-forest-test/decision-tree)` の後に追記する。

移植元は `t/cl-random-forest.lisp` の 130 行目と 244 行目の `is`。

```lisp
(deftest a9a-dtree-accuracy
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (let ((acc (n-times-average 10
                   (let ((dtree (make-dtree 2 datamatrix target :max-depth 20)))
                     (test-dtree dtree datamatrix-test target-test)))))
        (ok (approximately-equal acc 82.23217010498047d0)
            (format nil "a9a dtree accuracy ~,4F (expected 82.2322 +/- 1.0)" acc))))))

(deftest letter-dtree-accuracy
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (let ((acc (n-times-average 10
                   (let ((dtree (make-dtree +letter-n-class+ datamatrix target :max-depth 20)))
                     (test-dtree dtree datamatrix-test target-test)))))
        (ok (approximately-equal acc 83.9534912109375d0)
            (format nil "letter dtree accuracy ~,4F (expected 83.9535 +/- 1.0)" acc))))))
```

- [ ] **Step 2: 構造チェック（分離が効いていることの確認）**

```lisp
(asdf:load-system :cl-random-forest-test/decision-tree :force t)

(mapcar #'rove/core/suite/package:suite-name
        (rove/core/suite/package:system-suites
         (asdf:find-system "cl-random-forest-test/decision-tree")))
```

Expected: `("cl-random-forest-test/decision-tree")`
— `cl-random-forest-test/dataset` や `/fixture` が**混ざらないこと**。混ざったら
`:components` の書き方か 1ファイル1パッケージの原則が破れている。

```lisp
(rove/core/suite/package:suite-tests
 (rove:find-suite :cl-random-forest-test/decision-tree))
```

Expected: `(A9A-DTREE-ACCURACY LETTER-DTREE-ACCURACY)`

- [ ] **Step 3: スイートを実行**

```lisp
(rove:run :cl-random-forest-test/decision-tree)
```

Expected: 2テストとも pass。dataset スイートの3テストは走らない。数分かかる。
精度が期待値から `1.0` 以上ずれた場合は移行のバグ（乱数の使い方が変わっていないか、
データのラベル変換が正しいかを疑う）。

- [ ] **Step 4: 個別テスト実行を確認**

```lisp
(rove:run-test 'cl-random-forest-test/decision-tree::a9a-dtree-accuracy)
```

Expected: 1テストだけ pass。

- [ ] **Step 5: コミット**

```bash
git add t/decision-tree.lisp
git commit -m "Port decision tree accuracy tests to rove"
```

---

### Task 3: forest スイート

**Files:**
- Modify: `t/forest.lisp`（スタブに2つの `deftest` を追加）

**Interfaces:**
- Consumes: Task 1 の fixture（加えて `with-serial-kernel`）
- Produces: なし

- [ ] **Step 1: `t/forest.lisp` に2つの deftest を追加**

移植元は `t/cl-random-forest.lisp` の 141 行目と 255 行目の `is`。
旧コードは直前にトップレベルで `(setf lparallel:*kernel* nil)` していた。これを
`with-serial-kernel` に置き換える。

```lisp
(deftest a9a-forest-accuracy
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let ((forest (make-forest 2 datamatrix target
                                                :n-tree 500 :bagging-ratio 0.1
                                                :min-region-samples 5 :n-trial 10
                                                :max-depth 10)))
                       (test-forest forest datamatrix-test target-test)))))
          (ok (approximately-equal acc 84.07d0)
              (format nil "a9a forest accuracy ~,4F (expected 84.07 +/- 1.0)" acc)))))))

(deftest letter-forest-accuracy
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let ((forest (make-forest +letter-n-class+ datamatrix target
                                                :n-tree 500 :bagging-ratio 0.1
                                                :min-region-samples 5 :n-trial 10
                                                :max-depth 10)))
                       (test-forest forest datamatrix-test target-test)))))
          (ok (approximately-equal acc 89.05402374267578d0)
              (format nil "letter forest accuracy ~,4F (expected 89.0540 +/- 1.0)" acc)))))))
```

- [ ] **Step 2: 構造チェック**

```lisp
(asdf:load-system :cl-random-forest-test/forest :force t)

(mapcar #'rove/core/suite/package:suite-name
        (rove/core/suite/package:system-suites
         (asdf:find-system "cl-random-forest-test/forest")))
```

Expected: `("cl-random-forest-test/forest")`

```lisp
(rove/core/suite/package:suite-tests
 (rove:find-suite :cl-random-forest-test/forest))
```

Expected: `(A9A-FOREST-ACCURACY LETTER-FOREST-ACCURACY)`

- [ ] **Step 3: カーネルが漏れていないことを確認**

```lisp
(setf lparallel:*kernel* nil)
(rove:run :cl-random-forest-test/forest)
lparallel:*kernel*
```

Expected: 2テストとも pass。実行後 `lparallel:*kernel*` が `NIL` のまま
（`with-serial-kernel` が `let` 束縛であることの確認）。500本フォレスト×5回×2データセットで
5〜10分かかる。

- [ ] **Step 4: コミット**

```bash
git add t/forest.lisp
git commit -m "Port random forest accuracy tests to rove"
```

---

### Task 4: refinement スイート

**Files:**
- Modify: `t/refinement.lisp`（スタブに2つの `deftest` を追加）

**Interfaces:**
- Consumes: Task 1 の fixture
- Produces: なし

- [ ] **Step 1: `t/refinement.lisp` に2つの deftest を追加**

移植元は `t/cl-random-forest.lisp` の 171 行目と 285 行目の `is`。
`train-refine-learner-process` は第1引数を `setf` するマクロなので、`let*` 束縛の変数を渡す。

```lisp
(deftest a9a-refinement-accuracy
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let* ((forest (make-forest 2 datamatrix target
                                                 :n-tree 500 :bagging-ratio 0.1
                                                 :min-region-samples 5 :n-trial 10
                                                 :max-depth 10))
                            (refine-dataset (make-refine-dataset forest datamatrix))
                            (refine-test (make-refine-dataset forest datamatrix-test))
                            (refine-learner (make-refine-learner forest)))
                       (train-refine-learner-process refine-learner
                                                     refine-dataset target
                                                     refine-test target-test)
                       (test-refine-learner refine-learner refine-test target-test)))))
          (ok (approximately-equal acc 80.98789)
              (format nil "a9a refinement accuracy ~,4F (expected 80.9879 +/- 1.0)" acc)))))))

(deftest letter-refinement-accuracy
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (with-serial-kernel
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let* ((forest (make-forest +letter-n-class+ datamatrix target
                                                 :n-tree 500 :bagging-ratio 0.1
                                                 :min-region-samples 5 :n-trial 10
                                                 :max-depth 10))
                            (refine-dataset (make-refine-dataset forest datamatrix))
                            (refine-test (make-refine-dataset forest datamatrix-test))
                            (refine-learner (make-refine-learner forest)))
                       (train-refine-learner-process refine-learner
                                                     refine-dataset target
                                                     refine-test target-test)
                       (test-refine-learner refine-learner refine-test target-test)))))
          (ok (approximately-equal acc 97.06802368164063d0)
              (format nil "letter refinement accuracy ~,4F (expected 97.0680 +/- 1.0)" acc)))))))
```

- [ ] **Step 2: 構造チェック**

```lisp
(asdf:load-system :cl-random-forest-test/refinement :force t)

(mapcar #'rove/core/suite/package:suite-name
        (rove/core/suite/package:system-suites
         (asdf:find-system "cl-random-forest-test/refinement")))
```

Expected: `("cl-random-forest-test/refinement")`

```lisp
(rove/core/suite/package:suite-tests
 (rove:find-suite :cl-random-forest-test/refinement))
```

Expected: `(A9A-REFINEMENT-ACCURACY LETTER-REFINEMENT-ACCURACY)`

- [ ] **Step 3: スイートを実行**

```lisp
(rove:run :cl-random-forest-test/refinement)
```

Expected: 2テストとも pass。フォレスト構築に加えて refine 学習が入るので10分前後かかる。

- [ ] **Step 4: コミット**

```bash
git add t/refinement.lisp
git commit -m "Port global refinement accuracy tests to rove"
```

---

### Task 5: parallel スイート

**Files:**
- Modify: `t/parallel.lisp`（スタブに4つの `deftest` を追加）

**Interfaces:**
- Consumes: Task 1 の fixture（特に `with-parallel-kernel`）
- Produces: なし

- [ ] **Step 1: `t/parallel.lisp` に4つの deftest を追加**

移植元は `t/cl-random-forest.lisp` の 157, 194, 271, 307 行目の `is`（いずれも
`#+sbcl (progn ...)` の中）。期待値は直列版と同一。

旧コードは `(setf lparallel:*kernel* (lparallel:make-kernel 4))` するだけで
`end-kernel` を呼んでおらず、4スレッドのカーネルを毎回リークしていた。
`with-parallel-kernel` に置き換えることでこれも直る。

```lisp
#+sbcl
(deftest a9a-forest-accuracy-parallel
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (with-parallel-kernel ()
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let ((forest (make-forest 2 datamatrix target
                                                :n-tree 500 :bagging-ratio 0.1
                                                :min-region-samples 5 :n-trial 10
                                                :max-depth 10)))
                       (test-forest forest datamatrix-test target-test)))))
          (ok (approximately-equal acc 84.07d0)
              (format nil "a9a forest accuracy (parallel) ~,4F (expected 84.07 +/- 1.0)" acc)))))))

#+sbcl
(deftest a9a-refinement-accuracy-parallel
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (with-parallel-kernel ()
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let* ((forest (make-forest 2 datamatrix target
                                                 :n-tree 500 :bagging-ratio 0.1
                                                 :min-region-samples 5 :n-trial 10
                                                 :max-depth 10))
                            (refine-dataset (make-refine-dataset forest datamatrix))
                            (refine-test (make-refine-dataset forest datamatrix-test))
                            (refine-learner (make-refine-learner forest)))
                       (train-refine-learner-process refine-learner
                                                     refine-dataset target
                                                     refine-test target-test)
                       (test-refine-learner refine-learner refine-test target-test)))))
          (ok (approximately-equal acc 80.98789)
              (format nil "a9a refinement accuracy (parallel) ~,4F (expected 80.9879 +/- 1.0)"
                      acc)))))))

#+sbcl
(deftest letter-forest-accuracy-parallel
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (with-parallel-kernel ()
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let ((forest (make-forest +letter-n-class+ datamatrix target
                                                :n-tree 500 :bagging-ratio 0.1
                                                :min-region-samples 5 :n-trial 10
                                                :max-depth 10)))
                       (test-forest forest datamatrix-test target-test)))))
          (ok (approximately-equal acc 89.05402374267578d0)
              (format nil "letter forest accuracy (parallel) ~,4F (expected 89.0540 +/- 1.0)"
                      acc)))))))

#+sbcl
(deftest letter-refinement-accuracy-parallel
  (multiple-value-bind (datamatrix target) (letter-train)
    (multiple-value-bind (datamatrix-test target-test) (letter-test)
      (with-parallel-kernel ()
        (let ((acc (n-times-average 5
                     (trivial-garbage:gc :full t)
                     (let* ((forest (make-forest +letter-n-class+ datamatrix target
                                                 :n-tree 500 :bagging-ratio 0.1
                                                 :min-region-samples 5 :n-trial 10
                                                 :max-depth 10))
                            (refine-dataset (make-refine-dataset forest datamatrix))
                            (refine-test (make-refine-dataset forest datamatrix-test))
                            (refine-learner (make-refine-learner forest)))
                       (train-refine-learner-process refine-learner
                                                     refine-dataset target
                                                     refine-test target-test)
                       (test-refine-learner refine-learner refine-test target-test)))))
          (ok (approximately-equal acc 97.06802368164063d0)
              (format nil "letter refinement accuracy (parallel) ~,4F (expected 97.0680 +/- 1.0)"
                      acc)))))))
```

- [ ] **Step 2: 構造チェック**

```lisp
(asdf:load-system :cl-random-forest-test/parallel :force t)

(mapcar #'rove/core/suite/package:suite-name
        (rove/core/suite/package:system-suites
         (asdf:find-system "cl-random-forest-test/parallel")))
```

Expected: `("cl-random-forest-test/parallel")`

```lisp
(rove/core/suite/package:suite-tests
 (rove:find-suite :cl-random-forest-test/parallel))
```

Expected（SBCL）: 4つのシンボル
`(A9A-FOREST-ACCURACY-PARALLEL A9A-REFINEMENT-ACCURACY-PARALLEL
  LETTER-FOREST-ACCURACY-PARALLEL LETTER-REFINEMENT-ACCURACY-PARALLEL)`

- [ ] **Step 3: スレッドリークが直っていることを確認**

```lisp
(setf lparallel:*kernel* nil)
(length (sb-thread:list-all-threads))   ; 実行前のスレッド数を控える
(rove:run :cl-random-forest-test/parallel)
lparallel:*kernel*
(length (sb-thread:list-all-threads))
```

Expected: 4テストとも pass。実行後 `lparallel:*kernel*` が `NIL`、スレッド数が実行前と同じ
（旧コードなら 4×4 = 16 スレッド増えていた）。15分前後かかる。

- [ ] **Step 4: コミット**

```bash
git add t/parallel.lisp
git commit -m "Port parallel training accuracy tests to rove

Replace the toplevel (setf lparallel:*kernel* (make-kernel 4)) with a
with-parallel-kernel binding that calls end-kernel on unwind. The old code
leaked a 4-worker kernel per parallel test and left the kernel set when the
file finished."
```

---

### Task 6: 集約の検証・旧ファイル削除・CI・ドキュメント

**Files:**
- Delete: `t/cl-random-forest.lisp`
- Modify: `.github/workflows/ci.yml`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: Task 1〜5 のすべて
- Produces: なし（最終タスク）

- [ ] **Step 1: 集約システムが13テスト全部を走らせることを確認**

タスク1で書いた集約の `test-op` を、全機能システムが揃った状態で初めて検証する。

```lisp
(asdf:test-system :cl-random-forest)
```

Expected: SBCL で13テストすべて pass。dataset 3 + decision-tree 2 + forest 2 +
refinement 2 + parallel 4 = 13。全体で15分以上かかる。

集約が0テストで素通りしていないこと（設計書 3.3 の罠）を必ず出力で確認する。テスト数が
13未満なら `:perform` の明示リストか、いずれかの機能ファイルの `deftest` が
登録されていない。

失敗時に非0終了することの根拠: `rove:run` は `with-reporter` → `invoke-reporter` →
`call-with-suite` と辿り、`(values (passedp *stats*) (stats-results *stats*))` を返す。
テストが1つでも失敗すると第1値が `NIL` になるため、`.asd` の `(unless ... (error ...))` が
発火する（実測確認済み）。

- [ ] **Step 2: 旧テストファイルを削除**

```bash
git rm t/cl-random-forest.lisp
```

- [ ] **Step 3: prove が残っていないことを確認**

新しい Lisp イメージ（`ros -Q dynamic-space-size=2048`）で:

```lisp
(asdf:load-system :cl-random-forest-test)
(find-package :prove)
```

Expected: `NIL`

```bash
grep -rn "prove" --include="*.asd" --include="*.lisp" --include="*.yml" . | grep -v "^./docs/"
```

Expected: 出力なし。

- [ ] **Step 4: `.github/workflows/ci.yml` を更新**

`Install Prove and dependencies` と `Run tests` の2ステップを差し替える。

変更前:

```yaml
      - name: Install Prove and dependencies
        run: ros install prove lparallel trivial-garbage masatoi/cl-libsvm-format masatoi/cl-online-learning
      - name: Run tests
        run: |
          PATH="~/.roswell/bin:$PATH"
          run-prove cl-random-forest-test.asd
```

変更後:

```yaml
      - name: Install Rove and dependencies
        run: ros install rove lparallel trivial-garbage masatoi/cl-libsvm-format masatoi/cl-online-learning
      - name: Run tests
        run: |
          PATH="~/.roswell/bin:$PATH"
          rove cl-random-forest-test.asd
```

`rove` の roswell スクリプトは `.asd` を渡すと system 名を導出して `asdf:load-system` →
`asdf:test-system` を呼び、`rove/core/suite:*last-suite-report*` の全 suite が passed か
どうかで終了コードを決める。`run-prove` と同じ使い勝手。

- [ ] **Step 5: `CLAUDE.md` を更新**

`## Commands` 節を差し替える。変更点は3つ。

1. テスト実行コマンドの `run-prove` → `rove`
2. 「Test/example caveats」の1項目目（単一 prove ファイルで per-test selection 不可、
   `run-tests` MCP ツールが使えない）を削除し、機能システム一覧と個別実行方法に置き換える
3. `## Agent Guidelines` の「Both prompts assume **Rove**... This project uses **prove**」の
   但し書きを更新（rove を使うようになったので、この乖離の説明が不要になる）

`## Commands` のテスト実行部分:

```markdown
Run the test suite (rove-based):

```lisp
(asdf:test-system :cl-random-forest)
```

```sh
./t/run-test.ros                   # same, via roswell with dynamic-space-size=2048
rove cl-random-forest-test.asd     # what CI runs
```

Tests are split into feature systems, each of which can be run on its own:

| System | Tests |
|---|---|
| `cl-random-forest-test/dataset` | dataset download and conversion (3) |
| `cl-random-forest-test/decision-tree` | decision tree accuracy (2) |
| `cl-random-forest-test/forest` | random forest accuracy (2) |
| `cl-random-forest-test/refinement` | global refinement accuracy (2) |
| `cl-random-forest-test/parallel` | parallelized training accuracy (4, SBCL only) |

`cl-random-forest-test` is the aggregate that runs all five.
`cl-random-forest-test/fixture` holds the shared dataset loaders and helpers and has no tests.

```lisp
(rove:run :cl-random-forest-test/forest)                              ; one feature
(rove:run-test 'cl-random-forest-test/forest::a9a-forest-accuracy)    ; one test
```
```

「Test/example caveats」の1項目目を以下に差し替える:

```markdown
- Datasets are loaded lazily and memoized in `cl-random-forest-test/fixture`, so any single
  test can be run on its own and will fetch only what it needs.
```

- [ ] **Step 6: `t/run-test.ros` が動くことを確認**

変更していないが、rove 経由でも動くことを確認する。

```sh
./t/run-test.ros
```

Expected: 13テストすべて pass、終了コード 0。

- [ ] **Step 7: コミット**

```bash
git add -A
git commit -m "Switch CI and docs to rove, drop the old prove test file

t/cl-random-forest.lisp has been fully migrated into the six files under t/,
so remove it along with the last references to prove."
```

---

## 完了基準

設計書 4節の検証基準に対応する。すべて実行して出力を確認すること。

- [ ] `(rove:run :cl-random-forest-test/decision-tree)` が **2テストだけ**走る
- [ ] `(rove:run-test 'cl-random-forest-test/decision-tree::a9a-dtree-accuracy)` が **1テストだけ**走る
- [ ] 上記2つで fixture や他機能のテストが混入しない
- [ ] `(asdf:test-system :cl-random-forest)` が13テスト全部を実行し、失敗時に非0終了する
- [ ] 各機能システムを単独実行したとき（Task 2〜5 で実施）と集約実行したとき（Task 6 Step 1）で
      pass/fail が一致する（順序依存が残っていないことの確認）
- [ ] 新イメージで `cl-random-forest-test` をロードした後 `(find-package :prove)` が `NIL`
- [ ] `.asd` / `t/` / `.github/` に `prove` の文字列が残らない
- [ ] 並列スイート実行後に `lparallel:*kernel*` が `NIL`、スレッド数が実行前と同じ
