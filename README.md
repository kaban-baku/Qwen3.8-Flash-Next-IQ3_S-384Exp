# Qwen3.8 Flash-Next-IQ3_S-384Exp

[日本語](README.ja.md)

**384Exp = 384 experts per layer, down from 512.** The number of experts used per token (10) is unchanged.

This repository has the tools, the list of kept experts and the evaluation scripts for a pruned Qwen3.8-Flash-Next. The model is the IQ3_S file from ISTA-DASLab's [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) with 128 experts removed from every layer. The quantized expert weights and router rows that are kept are copied byte for byte; nothing is re-quantized.

I made it to run Qwen3.8-Flash-Next with [Strata](https://github.com/Niko1221/Strata) on an 8 GB GPU and 64 GB of RAM, with Strata's RAM budget at 24 GB so the PC stays usable for other work.

- Weights (first file, 42.2 GB): https://huggingface.co/kaban-baku/Qwen3.8-Flash-Next-IQ3_S-384Exp-GGUF
- The second file (28.8 GB) is identical to the second file of ISTA-DASLab's IQ3_S. Download it from there and rename it.

## Results

RTX 3070 Ti Laptop 8 GB, 64 GB RAM, Windows 11. Strata 0.2.10 built locally with the patch below, `--resident-budget-gib 24`, 128K context, MTP, STRATA_ROUTE_PRIOR=1.5.

| | IQ3_S (512 experts) | 384Exp |
|---|---|---|
| First file | 54.8 GB | 42.2 GB |
| Decode, tokens/s (2 runs) | 22.6 / 22.0 | 38.9 / 36.5 |
| Expert cache hit rate | 68% | 67-72% |
| JamC-QA, 748 questions (Japanese knowledge, 4 choices) | 65.9% | 63.1% |
| HumanEval, 164 | 92.7% | 92.7% |
| MBPP, 257 | 87.9% | 88.3% |
| ELYZA-tasks-100 (blind, graded by Claude, out of 5) | 4.09 | 3.64 |
| tool-eval-bench, full (69 scenarios) | 89 | 87 |
| top-1 / KL against IQ3_S | - | 80.7% / 0.440 |

- Decode speed is the mean of 4 requests (two short prompts, 1,200 output tokens each).
- Strata 0.2.11 built the same way gave the same speeds (384Exp 36.8 / 37.6, IQ3_S 22.6 / 20.1). Removing `--pcie-frac 0.37` to use 0.2.11's automatic CPU / upload share did not change them on this PC.
- Without the RAM budget, my first 384-expert file and IQ3_S were close (47.9 and 47.2 tokens/s on 0.2.10; not measured for this file).
- The main loss is in reasoning and trick questions (ELYZA). General knowledge dropped to 55.3% on JamC-QA with my first expert selection; adding knowledge prompts to the selection brought it back to 63.1%.

## You need a Strata patch for the speed

Up to Strata 0.2.11, the two routing biases (STRATA_ROUTE_PRIOR and STRATA_ROUTE_RESIDENT) only run on models with 512 experts. Without them, my first 384-expert file had a hit rate of about 42% and decoded at about 26.5 tokens/s (not measured for this file). The patch is proposed in https://github.com/Niko1221/Strata/issues/5287 and is in `strata-patch/route_bias_any_ne.patch`. It applies to v0.2.10 and v0.2.11 as is.

## How to use it

1. Get the weights: the first file from Hugging Face and the second file of ISTA-DASLab's IQ3_S. Rename the second one to `Qwen3.8-Flash-Next-IQ3_S-384Exp-00002-of-00002.gguf` and put both in the same folder.
2. Apply the patch to Strata and build it (`git apply strata-patch/route_bias_any_ne.patch`; checked with CUDA 12.4 for sm_86).
3. Make the pack and the expert profile:
   - `python <Strata>/tools/iq_pack.py --gguf <first file> --out <PACK_DIR>`
   - `python tools/remap_profile.py selection/keep_384Exp.json <Strata>/data/expert-profile-384Exp.bin` (maps Strata's bundled profile to the 384 expert numbers; set `STRATA_DIR` to your Strata folder)
4. Fill in the `<...>` in `config-example-8gb-24g-128k.json` and start it with `python <Strata>/serve/server.py --engine strata --config <config> --port 8080`.
5. Long prompts: Strata 0.2.10 and 0.2.11 stop with 0xC0000409 when a model with fewer than 512 experts reads a prompt longer than about 3,300 tokens (Strata issue #5104). The example config sets the workaround `STRATA_PREFILL_LAYER_MAJOR=0`; with it, this file read a 38,520-token prompt in 79 s and answered correctly.

## Make it yourself

```
python tools/prune_gguf.py --src <first file of IQ3_S> --keep selection/keep_384Exp.json --out <output>
python tools/prune_gguf.py --verify --src <first file of IQ3_S> --keep selection/keep_384Exp.json --out <output>
```

## How the experts were chosen

- Strata's `--dump-routing` recorded which experts the router picked while writing (299 prompts, 400 tokens each).
- Eight groups: code, reading Japanese, writing Japanese, tool use, short technical explanations, general knowledge, reasoning, and planning. Counts were normalized per group, weighted, summed, and the top 384 per layer were kept (the top 16 of every group are always kept). The weights are in `selection/keep_384Exp.json`.
- The 130 added prompts (general knowledge, reasoning, planning) can be regenerated with `selection/added_prompts.py`. It uses 14 questions each from the train splits of JCommonsenseQA and JEMHopQA.
- The other 169 prompts came from my own work notes and are not published.
- None of the evaluation questions (JamC-QA, ELYZA-tasks-100, tool-eval-bench, HumanEval, MBPP) were used for the selection.

## How it was evaluated

- `eval/eval_suite.py`: temperature 0, no thinking. JamC-QA takes up to 100 questions per category (seed 42); HumanEval and MBPP run their original tests locally.
- ELYZA-tasks-100: the answers of IQ3_S, my first 384-expert file and 384Exp were shuffled per question and hidden, and four Claude (Sonnet) graders scored 25 questions each with ELYZA's grading guideline.
- top-1 / KL: Strata's `tools/quant_quality.py` (five kinds of text, 6,495 positions).

## License

- Scripts and the expert list in this repository: MIT (`LICENSE`). `strata-patch/` is a change to Strata (MIT) and follows Strata's license once merged.
- Weights: Qwen Community License 1.0 of Qwen3.8-Flash-Next. ISTA-DASLab's quantized files state that they inherit the license of the base model. The license text is included in the Hugging Face repository.

## Credits

Qwen (Qwen3.8-Flash-Next), ISTA-DASLab (GSQ-RCO quantization), Strata, and the authors of JCommonsenseQA, JEMHopQA, JamC-QA, ELYZA-tasks-100 and tool-eval-bench.
