# テストフレームワーク prove → rove 移行 設計

日付: 2026-08-01
対象リポジトリ: masatoi/cl-random-forest (master @ b4e5ec5)

## 1. 背景と目的

### 目的

cl-mcp の `run-tests` ツールから**個々のテストを選んで実行できるようにする**。

### 現状の問題

`t/cl-random-forest.lisp`（324行）は prove ベースの**平坦なスクリプト**である。

```
(plan nil)
  → 13個のトップレベル ok / is フォーム
(finalize)
```

個別実行できない原因は3つある。

1. **名前付きテスト単位が存在しない。** `subtest` すら使われておらず、選択の単位がない。
2. **データセット準備自体がアサーションになっており、後続テストがその副作用に依存している。**
   `(ok (let ((a9a-train (clol.utils:read-data ...))) ... (setf a9a-datamatrix datamat) ...))`
   という形で、`defvar` されたグローバル変数に `setf` している。3番目以降のテストだけを
   実行してもデータが `unbound` のまま失敗する。
3. **`lparallel:*kernel*` がトップレベルで `setf` されている。**
   テストの実行順序に暗黙に依存している。

したがって「prove を rove に置換する」だけでは目的を達成できない。(2) と (3) の解消が必須である。

## 2. スコープ

### やること

- prove → rove への移行（`.asd`、CI、テストファイル、ドキュメント）
- 13アサーションを **13個の `deftest` に 1:1 で移植**
- 機能軸での ASDF システム分割（5機能 + fixture + 集約）
- データセット読み込みの遅延メモ化（上記 (2) の解消）
- `lparallel:*kernel*` のスコープ化（上記 (3) の解消。既存のスレッドリーク修正を含む）

### やらないこと

- **新規テストの追加。** 回帰、Global Pruning、feature-importance、reconstruction には
  現在テストが無いが、今回は書かない。機能分割の結果、これらが空白であることが
  構造上見えるようになる（将来の作業の足がかり）。
- 期待精度・反復回数・許容誤差（`1d0`）の見直し。すべて現状維持。
- Issue #14 / #15 / #16 の修正。

## 3. 設計

### 3.1 システム構成

すべて `cl-random-forest-test.asd` の中に定義する（ASDF の副システム命名規則
`primary/secondary` を primary の .asd に置く形式）。

```
cl-random-forest-test                 集約（全機能に依存）
cl-random-forest-test/fixture         t/fixture.lisp        テストを持たない共有基盤
cl-random-forest-test/dataset         t/dataset.lisp        3 tests
cl-random-forest-test/decision-tree   t/decision-tree.lisp  2 tests
cl-random-forest-test/forest          t/forest.lisp         2 tests
cl-random-forest-test/refinement      t/refinement.lisp     2 tests
cl-random-forest-test/parallel        t/parallel.lisp       4 tests
```

1ファイル = 1パッケージ = 1 rove スイート。パッケージ名はシステム名と一致させる。

### 3.2 機能分離が成立する根拠

rove のソース（`core/suite.lisp`, `core/suite/package.lisp`, `core/suite/file.lisp`）を読んで確認した。

`rove:run-system` は system の型で分岐する。

- **plain `asdf:system`**: `(dolist (suite (system-suites system)) ...)`
  `system-suites` → `system-packages` → `system-files` → `component-source-files`。
  `component-source-files` は **そのシステム自身の children だけ**を辿るので、
  依存システムのスイートは走らない。
- **`asdf:package-inferred-system`**: 依存パッケージのスイートも先に走らせる分岐がある。

したがってテスト側のシステムは **plain `defsystem` を維持する**（本体 `cl-random-forest` は
package-inferred-system だが、テストは別）。

ファイルとパッケージの対応は、`deftest` が `set-test` → `package-suite` → `make-new-suite` を
呼んだ時点で `(setf (file-package *load-pathname*) package)` として登録される。

### 3.3 集約システムの注意点

集約 `cl-random-forest-test` は自身のコンポーネントを持たない。そのため
`component-source-files` が空 → `system-suites` が空 → **`(rove:run c)` と書くと
0テストで「成功」してしまう**。

これを避けるため、集約の `test-op` では**システム名のリストを明示的に渡す**。
`rove:run` は `run-system-tests` に委譲し、リストを受け付ける。

```lisp
(defsystem "cl-random-forest-test"
  :depends-on ("rove"
               "cl-random-forest-test/dataset"
               "cl-random-forest-test/decision-tree"
               "cl-random-forest-test/forest"
               "cl-random-forest-test/refinement"
               "cl-random-forest-test/parallel")
  :description "Test system for cl-random-forest"
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

`test-op` は `:depends-on` を伝播しない（伝播するのは `load-op`）ので、この明示リストは必須。

各機能システムは同型。

```lisp
(defsystem "cl-random-forest-test/forest"
  :depends-on ("rove" "cl-random-forest-test/fixture")
  :components ((:module "t" :components ((:file "forest"))))
  :perform (test-op (o c) (declare (ignore o)) (symbol-call :rove :run c)))
```

`:rove` は各機能システムの `:depends-on` に**直接**書く。cl-mcp の `run-tests` は
テストシステム自身の `:depends-on` からフレームワークを判定するため。

fixture はアサーションを持たないので rove に依存しない。

```lisp
(defsystem "cl-random-forest-test/fixture"
  :depends-on ("cl-random-forest" "cl-online-learning" "uiop" "trivial-garbage" "lparallel")
  :components ((:module "t" :components ((:file "fixture")))))
```

`cl-random-forest.asd` の `:in-order-to ((test-op (test-op cl-random-forest-test)))` は変更不要。

### 3.4 fixture の API

`t/fixture.lisp` / パッケージ `cl-random-forest-test/fixture`。

| シンボル | 内容 |
|---|---|
| `*dataset-dir*` | 現状のまま（`dataset/` under system source directory） |
| `+a9a-dim+` / `+letter-dim+` / `+letter-n-class+` | 123 / 16 / 26。旧 `defparameter` から定数に |
| `fetch-a9a` / `fetch-letter` | 現状のまま移設（`wget` によるDL） |
| `a9a-train` / `a9a-test` | `(values datamatrix target)` を返す。遅延メモ化 |
| `letter-train` / `letter-test` | 同上 |
| `approximately-equal` | 現状のまま（`delta` 既定 `1d0`） |
| `n-times-average` | 現状のまま（マクロ） |
| `with-serial-kernel` | 3.6 参照 |
| `with-parallel-kernel` | 3.6 参照 |

遅延メモ化アクセサの形（4つとも同型）。キャッシュには `(datamatrix . target)` の cons を持ち、
`values` に展開して返す。

```lisp
(defvar *a9a-train-cache* nil)

(defun a9a-train ()
  "Return (values datamatrix target) for the a9a training set, loading it on first call."
  (unless *a9a-train-cache*
    (fetch-a9a)
    (let ((data (clol.utils:read-data (merge-pathnames "a9a" *dataset-dir*) +a9a-dim+)))
      (dolist (datum data)                         ; ラベルを 0/1 に変換
        (setf (car datum) (if (> (car datum) 0d0) 0 1)))
      (multiple-value-bind (datamatrix target)
          (clol-dataset->datamatrix/target data)
        (setf *a9a-train-cache* (cons datamatrix target)))))
  (values (car *a9a-train-cache*) (cdr *a9a-train-cache*)))
```

`letter-train` / `letter-test` はラベル変換が無く、`read-data` に `:multiclass-p t` を渡す点が
異なる（`+letter-dim+` = 16、`+letter-n-class+` = 26）。`a9a-test` は `"a9a.t"`、
`letter-test` は `"letter.scale.t"` を読む。

旧テストファイルの `defparameter a9a-dim` 等は fixture の定数（`+a9a-dim+` 形式）に移す。
`cat` / `format-directory` / `format-filename` / `format-pathname`（`wget` 用のパス整形ヘルパ）も
fixture に移設する。

**これが本移行の中核。** 個別テスト実行時に、そのテストが必要とするデータだけが
自動的に揃うようになる。

### 3.5 アサーション変換

rove に `is` は無い。**10個の `is`**（t/cl-random-forest.lisp の 130, 141, 157, 171, 194,
244, 255, 271, 285, 307 行）を `ok` に書き換える。期待値・反復回数・許容誤差は変更しない。

`ok` は成否しか記録しないため、素で書くと確率的テストの失敗時に実測値が分からない。
説明文字列に実測値を埋めて診断性を確保する。

```lisp
;; before (prove)
(is (n-times-average 10 ...) 82.23217010498047d0 :test #'approximately-equal)

;; after (rove)
(deftest a9a-dtree-accuracy
  (multiple-value-bind (datamatrix target) (a9a-train)
    (multiple-value-bind (datamatrix-test target-test) (a9a-test)
      (let ((acc (n-times-average 10
                   (let ((dtree (make-dtree 2 datamatrix target :max-depth 20)))
                     (test-dtree dtree datamatrix-test target-test)))))
        (ok (approximately-equal acc 82.23217010498047d0)
            (format nil "a9a dtree accuracy ~,4F (expected 82.2322 +/- 1.0)" acc))))))
```

既存の**3個の `ok`**（87 行のファイル存在確認、101 行の a9a 準備、222 行の letter 準備）は
`ok` のまま `deftest` に包んで移植する。10 + 3 = 13 で移植前後のアサーション数は一致する。

### 3.6 lparallel カーネルのスコープ化（既存の不具合修正）

現状のコードは以下を繰り返している。

```lisp
(setf lparallel:*kernel* (lparallel:make-kernel 4))
...
(setf lparallel:*kernel* nil)   ; end-kernel を呼んでいない
```

問題:

- `lparallel:end-kernel` を呼ばずに参照を捨てているため、**4スレッドのカーネルが毎回リークする**
  （並列テストは4箇所ある）。
- 最後の並列テストは `*kernel*` を設定したまま終了する。
- トップレベルの `setf` なので、テストの実行順序に暗黙に依存する。

機能ごとに独立実行できるようにする以上、(3) は必ず解消しなければならない。fixture に
マクロを追加し、各テストが自分のスコープでカーネルを持つようにする。

```lisp
(defmacro with-parallel-kernel ((&optional (n 4)) &body body)
  "Bind lparallel:*kernel* to a fresh N-worker kernel for BODY, then shut it down."
  `(let ((lparallel:*kernel* (lparallel:make-kernel ,n)))
     (unwind-protect (progn ,@body)
       (lparallel:end-kernel :wait t))))

(defmacro with-serial-kernel (&body body)
  "Bind lparallel:*kernel* to NIL for BODY, forcing serial execution."
  `(let ((lparallel:*kernel* nil))
     ,@body))
```

`let` 束縛なので `unwind-protect` を抜ければ元の値に戻る。グローバル `setf` は全廃する。

### 3.7 テスト移植の対応表（13 → 13）

| # | 旧（トップレベルフォーム） | 新システム | 新 deftest 名 | 期待値 | 反復 |
|---|---|---|---|---|---|
| 1 | `ok` ファイル存在確認 | `/dataset` | `dataset-files-exist` | — | — |
| 2 | `ok` a9a 準備 | `/dataset` | `a9a-dataset-load` | — | — |
| 3 | `ok` letter 準備 | `/dataset` | `letter-dataset-load` | — | — |
| 4 | `is` a9a dtree | `/decision-tree` | `a9a-dtree-accuracy` | 82.23217010498047d0 | 10 |
| 5 | `is` letter dtree | `/decision-tree` | `letter-dtree-accuracy` | 83.9534912109375d0 | 10 |
| 6 | `is` a9a forest | `/forest` | `a9a-forest-accuracy` | 84.07d0 | 5 |
| 7 | `is` letter forest | `/forest` | `letter-forest-accuracy` | 89.05402374267578d0 | 5 |
| 8 | `is` a9a refine | `/refinement` | `a9a-refinement-accuracy` | 80.98789 | 5 |
| 9 | `is` letter refine | `/refinement` | `letter-refinement-accuracy` | 97.06802368164063d0 | 5 |
| 10 | `is` a9a forest 並列 | `/parallel` | `a9a-forest-accuracy-parallel` | 84.07d0 | 5 |
| 11 | `is` a9a refine 並列 | `/parallel` | `a9a-refinement-accuracy-parallel` | 80.98789 | 5 |
| 12 | `is` letter forest 並列 | `/parallel` | `letter-forest-accuracy-parallel` | 89.05402374267578d0 | 5 |
| 13 | `is` letter refine 並列 | `/parallel` | `letter-refinement-accuracy-parallel` | 97.06802368164063d0 | 5 |

旧ファイルの並び（a9a 全部 → letter 全部）から、機能軸の並びに再編する。
フォレスト系のパラメータは全て `:n-tree 500 :bagging-ratio 0.1 :min-region-samples 5
:n-trial 10 :max-depth 10` で共通。各フォレストテストは現状どおり
`(trivial-garbage:gc :full t)` を反復の先頭で呼ぶ。

`/parallel` の4テストは `#+sbcl` を維持する（CCL では並列テストを走らせない現状の挙動を保つ）。

**`dataset-files-exist` の注意点**: 旧コードはトップレベルで `(fetch-letter)` `(fetch-a9a)` を
呼んでからファイル存在を確認していた。遅延化後は誰も先に fetch しないので、この deftest 自身が
先頭で `(fetch-a9a)` `(fetch-letter)` を呼ぶ。他の deftest は `a9a-train` 等のアクセサ経由で
暗黙に fetch されるため、`/dataset` を実行しなくても単独で成立する。

### 3.8 ビルド・CI・ドキュメント

| 対象 | 変更 |
|---|---|
| `cl-random-forest-test.asd` | 全面書き換え。`:defsystem-depends-on (:prove-asdf)` と `prove` 依存を削除 |
| `t/cl-random-forest.lisp` | 削除 |
| `t/fixture.lisp` ほか5ファイル | 新規 |
| `t/run-test.ros` | 変更なし（`(asdf:test-system :cl-random-forest)` のまま。`dynamic-space-size=2048` も維持） |
| `.github/workflows/ci.yml` | `ros install prove ...` → `ros install rove ...`、`run-prove cl-random-forest-test.asd` → `rove cl-random-forest-test.asd` |
| `CLAUDE.md` | Commands 節を更新。prove/`run-prove` の記述、「`run-tests` MCPツールと個別実行は使えない」の記述を書き換え |

`rove` roswell スクリプトは導入済み（確認済み）。`.asd` を渡すと system 名を導出して
`asdf:load-system` → `asdf:test-system` を呼び、`rove/core/suite:*last-suite-report*` の
全 suite が passed かどうかで終了コードを決める。`run-prove` と同じ使い勝手。

## 4. 完了の検証基準

実装完了と判断する前に、以下をすべて実行して出力を確認する。

1. `run-tests system=cl-random-forest-test/decision-tree` が **2テストだけ**走る
2. `run-tests system=cl-random-forest-test/decision-tree test=cl-random-forest-test/decision-tree::a9a-dtree-accuracy`
   が **1テストだけ**走る
3. 上記 1, 2 で fixture や他機能のテストが混入しない（3.2 の設計が効いていることの確認）
4. `asdf:test-system :cl-random-forest` が13テスト全部を実行し、失敗時に非0終了する
5. 各機能システムを**単独で**実行した結果が、集約実行時と一致する
   （3.6 のカーネルスコープ化が効いていることの確認）
6. 新しい Lisp イメージで `(asdf:load-system :cl-random-forest-test)` した後、
   `(find-package :prove)` が `NIL` を返す（prove が残っていないことの確認）
7. リポジトリ全体で `prove` の文字列が残るのはドキュメントの移行経緯の記述のみ
   （`.asd`、`t/`、`.github/` には残らない）

テストはネットワーク（`wget`）とデータセットのDLを要し、全体で15分以上かかる。
検証時はこの所要時間を見込む。

## 5. リスクと留意点

- **`file-package` 登録の前提**: rove はファイルとパッケージの対応を、`deftest` 評価時の
  `*load-pathname*` / `*compile-file-pathname*` から記録する。1ファイル1パッケージを
  厳守すること。2ファイルが同じパッケージを使うと機能分離が壊れる。
- **精度テストは確率的**: 許容誤差 `1d0` で平均を取る設計は現状のまま引き継ぐ。移行によって
  期待値が変わってはならない。移行後に値がずれた場合、それは移行のバグである。
- **`n-times-average` は `double-float` に coerce する**が、期待値のうち `80.98789` は
  single-float リテラル。`approximately-equal` は `etypecase x` で actual 側（double-float）を
  見るので現状動作する。ここは変更しない。
- **CCL**: `/parallel` は `#+sbcl` で丸ごと無効化されるため、CCL では 9 テストになる。
  CI マトリクスは {sbcl-bin, ccl-bin} × {ubuntu, macOS} のまま。
