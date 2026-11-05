# Qwen3.8 Flash-Next-IQ3_S-384Exp

[English](README.md)

**384Exp = 各層の専門家（experts）を 512 から 384 に減らした版**です。1 トークンで使う専門家の数（10）は変えていません。

元は ISTA-DASLab の [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) の IQ3_S です。量子化はそのままで、専門家の重みとルーターの行をバイト単位で切り出しています（数値を計算し直す処理はしていません）。

VRAM 8GB の GPU と RAM 64GB の PC で、Strata の RAM の上限を 24GB にして普段使いと並べて動かすために作りました。

- 重み（1 本目、42.2GB）：https://huggingface.co/kaban-baku/Qwen3.8-Flash-Next-IQ3_S-384Exp-GGUF
- 2 本目（28.8GB）は、元の IQ3_S の 2 本目と同じ中身です。ISTA-DASLab から落として名前を変えて使います。

## 結果（RTX 3070 Ti Laptop 8GB・RAM 64GB・Windows 11、Strata 0.2.10 を自前でビルド＋下の修正、RAM の上限 24GB、文脈 128K、MTP）

| 項目 | 元の IQ3_S（512） | 384Exp |
|---|---|---|
| 本体（1 本目） | 54.8GB | 42.2GB |
| 書く速さ（tok/s、2 回） | 22.6／22.0 | 38.9／36.5 |
| VRAM の専門家キャッシュの当たり率 | 68% | 67〜72% |
| JamC-QA 748 問（日本の知識、4 択） | 65.9% | 63.1% |
| HumanEval 164 問 | 92.7% | 92.7% |
| MBPP 257 問 | 87.9% | 88.3% |
| ELYZA-tasks-100（3 つの版を伏せて Claude が採点、5 点満点） | 4.09 | 3.64 |
| tool-eval-bench 完全版（69 場面） | 89 | 87 |
| top-1 一致／KL（元の IQ3_S の出力が基準） | − | 80.7%／0.440 |

- 速さは、同じ短い問い 2 つ（1,200 トークンずつ）を 4 回ずつ送った平均です。
- Strata 0.2.11 を同じように作っても速さは同じでした（384Exp 36.8／37.6、IQ3_S 22.6／20.1）。`--pcie-frac 0.37` を外して 0.2.11 の自動の割り振りにしても、この PC では変わりませんでした。RAM の上限をかけない場合は、元の IQ3_S との差が小さくなります（0.2.10、最初の 384 で 47.9、元の IQ3_S で 47.2。384Exp では未計測）。
- 落ちたのは主に推理・ひっかけ（ELYZA）です。一般知識は、最初の選び方では JamC-QA が 55.3% まで落ちましたが、記録を足して選び直して 63.1% まで戻りました。

## 速くするには Strata の修正が要ります

Strata 0.2.11 までは、経路の偏り（STRATA_ROUTE_PRIOR・STRATA_ROUTE_RESIDENT）が専門家 512 のモデルでしか動きません。修正なしだと、最初の 384 では当たり率が 42% 前後、書く速さが 26.5 tok/s 程度でした（384Exp の修正なしは未計測）。修正は issue https://github.com/Niko1221/Strata/issues/5287 で提案中で、差分は `strata-patch/route_bias_any_ne.patch` です（v0.2.10 と v0.2.11 にそのまま当たります）。

## 使い方

1. 重みをそろえる：Hugging Face から 1 本目を、ISTA-DASLab から IQ3_S の 2 本目を落とし、2 本目の名前を `Qwen3.8-Flash-Next-IQ3_S-384Exp-00002-of-00002.gguf` にします（同じ場所に置く）。
2. Strata に修正を当ててビルドします（`git apply strata-patch/route_bias_any_ne.patch`、CUDA 12.4、sm_86 で確認）。
3. pack とプロファイルを作ります。
   - `python <Strata>/tools/iq_pack.py --gguf <1 本目> --out <PACK_DIR>`
   - `python tools/remap_profile.py selection/keep_384Exp.json <Strata>/data/expert-profile-384Exp.bin`（元の付属プロファイルを 384 の番号に写し替え）
4. `config-example-8gb-24g-128k.json` の `<…>` を埋めて、`python <Strata>/serve/server.py --engine strata --config <設定> --port 8080` で起動します。
5. 長文の注意：Strata 0.2.10・0.2.11 は、専門家 512 以外のモデルに約 3,300 トークンを超える文を読ませるとエンジンが落ちます（issue #5104）。設定の例には回避策の `STRATA_PREFILL_LAYER_MAJOR=0` を入れてあり、3.9 万トークンを読ませて完走しました。

## 自分で切る

```
python tools/prune_gguf.py --src <IQ3_S の 1 本目> --keep selection/keep_384Exp.json --out <出力>
python tools/prune_gguf.py --verify --src <IQ3_S の 1 本目> --keep selection/keep_384Exp.json --out <出力>
```

## 専門家の選び方

- Strata の `--dump-routing` で、書いている間にどの専門家が選ばれたかを記録しました（299 本、各 400 トークン）。
- 群は 8 つ：コード・日本語を読む・日本語を書く・道具を使う・技術の小ネタの説明・一般知識・推理・計画と判断。群ごとに回数を正規化し、重みを掛けて足し、各層の上位 384 を残しました（各群の上位 16 は必ず残す）。重みは `selection/keep_384Exp.json` にあります。
- 足した 130 本の問い（一般知識・推理・計画）は `selection/added_prompts.py` で作れます。JCommonsenseQA・JEMHopQA の train から 14 問ずつ使っています。
- 残りの 169 本は、作者の手元の作業記録から作った問いなので公開していません。
- 評価に使った問題（JamC-QA・ELYZA-tasks-100・tool-eval-bench・HumanEval・MBPP）は、記録に入れていません。

## 評価のやり方

- `eval/eval_suite.py`：温度 0・思考なし。JamC-QA は分野ごとに最大 100 問（乱数 42）、HumanEval と MBPP は元のテストを手元で実行。
- ELYZA-tasks-100：元の IQ3_S・最初の 384・384Exp の回答を、問題ごとに順番を変えて伏せ、Claude（Sonnet）4 人が 25 問ずつ、ELYZA の採点基準で採点しました。
- top-1／KL：Strata の `tools/quant_quality.py`（5 種類の文、6,495 位置）。

## 規約

- 重み：Qwen Community License 1.0（元の Qwen3.8-Flash-Next の規約。ISTA-DASLab の量子化版も「元の規約を引き継ぐ」としています）。LICENSE の本文は Hugging Face の置き場に同梱します。
- このリポジトリの台本と一覧：MIT（`LICENSE`）。ただし `strata-patch/` の差分は Strata（MIT）への変更で、取り込まれたら Strata 側の規約に従います。

## 謝辞

Qwen（Qwen3.8-Flash-Next）、ISTA-DASLab（GSQ-RCO の量子化）、Strata、JCommonsenseQA・JEMHopQA・JamC-QA・ELYZA-tasks-100・tool-eval-bench の作者に感謝します。
