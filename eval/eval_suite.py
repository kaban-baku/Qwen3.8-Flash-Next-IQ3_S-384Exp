"""刈り込み版の質の評価（11-04 夜）。サーバー（:8080、キー無し）が起きている前提で、1 モデル分を回す。
usage: python eval_suite.py <label> [jamc humaneval mbpp elyza]
- 温度 0・思考なし（chat_template_kwargs.enable_thinking=False）。同じ IQ3_S を 2 回測ると top-1 100%・KL 0 だったので、結果は決定的とみなす
- 結果は out/<label>/<task>.jsonl（1 問 1 行、途中から再開できる）と out/<label>/summary.json
- jamc：JamC-QA test を区分ごとに 100 問（種 42、healthcare は 48 問全部）、番号 1 文字で答えさせ完全一致
- humaneval：OpenAI HumanEval 164 問、元のテスト（check）を subprocess で実行（10 秒で打ち切り）
- mbpp：MBPP sanitized test 257 問、assert を subprocess で実行
- elyza：ELYZA-tasks-100 の回答を保存するだけ（採点は別途、審判は Claude）
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
label = sys.argv[1]
tasks = sys.argv[2:] or ["jamc", "humaneval", "mbpp", "elyza"]
OUT = HERE / "out" / label
OUT.mkdir(parents=True, exist_ok=True)


def chat(content, max_tokens):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                j = json.load(r)
            break
        except Exception as e:  # サーバーの一時的な失敗は 3 回まで
            if attempt == 2:
                return {"text": "", "error": str(e), "sec": round(time.time() - t, 1), "ctoks": 0}
            time.sleep(10)
    return {"text": j["choices"][0]["message"].get("content") or "", "sec": round(time.time() - t, 1),
            "ctoks": j.get("usage", {}).get("completion_tokens", 0)}


def done_ids(path):
    if not path.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def run_task(name, items, make_prompt, max_tokens, score):
    path = OUT / f"{name}.jsonl"
    done = done_ids(path)
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            if it["id"] in done:
                continue
            if os.environ.get("MOCK"):   # 採点の確認用：模範解答を回答とみなす（全問正解になるはず）
                r = {"text": "```python\n" + it["ref"] + "\n```" if name != "jamc" else str(it["a"] + 1), "sec": 0, "ctoks": 0}
            else:
                r = chat(make_prompt(it), max_tokens)
            rec = {"id": it["id"], **r}
            rec.update(score(it, r["text"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            done[it["id"]] = rec
    recs = list(done.values())
    s = {"n": len(recs), "sec": round(sum(r["sec"] for r in recs)), "ctoks": sum(r["ctoks"] for r in recs)}
    if recs and "ok" in recs[0]:
        s["ok"] = sum(1 for r in recs if r["ok"]); s["acc"] = round(s["ok"] / len(recs) * 100, 1)
    print(name, json.dumps(s, ensure_ascii=False), flush=True)
    return s


def code_block(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(m, key=len) if m else text


def run_py(src, timeout=10):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.py"; p.write_text(src, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            return r.returncode == 0, r.stderr.decode("utf-8", "replace")[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8")) if (OUT / "summary.json").exists() else {}

if "jamc" in tasks:
    d = pd.read_parquet(DATA / "jamc_test.parquet")
    d = pd.concat([g.sample(n=min(100, len(g)), random_state=42) for _, g in d.groupby("category")])
    items = [{"id": r.qid, "cat": r.category, "q": r.question, "c": [r.choice0, r.choice1, r.choice2, r.choice3],
              "a": int(r.answer_index)} for r in d.itertuples()]

    def p_jamc(it):
        ch = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(it["c"]))
        return f"次の問題に答えてください。\n\n問題：{it['q']}\n\n選択肢：\n{ch}\n\n正しい選択肢の番号（1〜4）だけを半角数字 1 文字で答えてください。"

    def s_jamc(it, text):
        m = re.search(r"[1-4１-４]", text)
        pred = ("1234１２３４".index(m.group()) % 4) if m else -1
        return {"pred": pred, "ok": pred == it["a"], "cat": it["cat"]}

    summary["jamc"] = run_task("jamc", items, p_jamc, 16, s_jamc)
    recs = done_ids(OUT / "jamc.jsonl").values()
    summary["jamc"]["by_cat"] = {c: round(sum(r["ok"] for r in recs if r["cat"] == c) / max(1, sum(1 for r in recs if r["cat"] == c)) * 100, 1)
                                 for c in sorted({r["cat"] for r in recs})}

if "humaneval" in tasks:
    d = pd.read_parquet(DATA / "humaneval.parquet")
    items = [{"id": r.task_id, "prompt": r.prompt, "test": r.test, "entry": r.entry_point,
              "ref": r.prompt + r.canonical_solution} for r in d.itertuples()]

    def p_he(it):
        return ("次の Python 関数を完成させてください。import を含めた関数全体を 1 つの ```python コードブロックで出力してください。\n\n```python\n"
                + it["prompt"] + "```")

    def s_he(it, text):
        code = code_block(text)
        if f"def {it['entry']}" not in code:
            code = it["prompt"] + code
        src = it["prompt"].split("def ")[0] + "\n" + code + "\n\n" + it["test"] + f"\n\ncheck({it['entry']})\n"
        ok, err = run_py(src)
        return {"ok": ok, "err": err if not ok else ""}

    summary["humaneval"] = run_task("humaneval", items, p_he, 1024, s_he)

if "mbpp" in tasks:
    d = pd.read_parquet(DATA / "mbpp_san_test.parquet")
    items = [{"id": str(r.task_id), "prompt": r.prompt, "imports": list(r.test_imports), "tests": list(r.test_list), "ref": r.code}
             for r in d.itertuples()]

    def p_mbpp(it):
        return (f"{it['prompt']}\n\nYour code should pass these tests:\n" + "\n".join(it["tests"])
                + "\n\n関数の定義だけを 1 つの ```python コードブロックで出力してください（テストは書かない）。")

    def s_mbpp(it, text):
        src = "\n".join(it["imports"]) + "\n" + code_block(text) + "\n\n" + "\n".join(it["tests"]) + "\n"
        ok, err = run_py(src)
        return {"ok": ok, "err": err if not ok else ""}

    summary["mbpp"] = run_task("mbpp", items, p_mbpp, 768, s_mbpp)

if "elyza" in tasks:
    d = pd.read_csv(DATA / "elyza.csv")
    items = [{"id": f"elyza-{i:03d}", "input": r.input} for i, r in enumerate(d.itertuples())]
    summary["elyza"] = run_task("elyza", items, lambda it: it["input"], 1536, lambda it, t: {})

(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print("summary", json.dumps({k: {kk: v.get(kk) for kk in ("acc", "n", "sec", "ctoks")} for k, v in summary.items()}, ensure_ascii=False))
