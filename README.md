# Compute — LLM training planner

A responsive, animated React application that estimates GPU hours, elapsed time, GPU memory requirements and USD rental costs for dense transformer full-parameter training. Calculations run locally in the browser. No backend, API keys, or GPU is required.

## Run

Use Node.js 20.19+ or 22.12+ (validated with Node 24).

```sh
npm ci --cache /tmp/compute-npm-cache
npm run dev -- --port 5173
```

```sh
npm test
npm run build
npm run preview -- --port 4173
```

Deploy `dist/` on any static host. The browser does not require external fonts or APIs.

## Model and hardware coverage

Presets include Llama 3.1, Qwen 2.5, Mistral, Gemma 2 and Phi-3 mini, plus a custom dense architecture with editable parameter count, layer count and hidden width. This is not an exhaustive model registry. Open-weight licenses vary; review each model's license. GPU presets include H100 SXM, A100 SXM, L40S and RTX 4090. GPU counts are directly editable; the slider covers 1–64.

## Method

For parameters N, layers L, hidden width H, sequence length S and processed tokens D:

- FLOPs = (6N + 12LHS) × D
- D = tokens per epoch × epochs; record mode uses records × average tokenized length
- GPU hours = FLOPs / (dense GPU peak FLOPs/sec × MFU × cluster efficiency) / 3600 × (1 + overhead)
- Elapsed hours = GPU hours / GPU count
- USD cost = GPU hours × price per GPU-hour

Single-GPU runs use 100% cluster efficiency. Defaults assume 40% model FLOPs utilization, 90% multi-GPU efficiency and 10% run overhead. All are editable. The 0.75–1.5× range is a planning heuristic, not a confidence interval. A pilot run should calibrate throughput.

Memory approximates 16 bytes per parameter for mixed-precision Adam model/gradient/master-weight/optimizer states. We reserve 20% aggregate capacity; actual activation memory and buffers may exceed that. Assumes supported sharded training (e.g. FSDP/ZeRO); aggregate capacity is not a fit guarantee. Offloading requires recalibrating throughput.

LoRA/QLoRA, mixture-of-experts, inference, FP8 and sparse acceleration are outside this estimator. Rate presets are illustrative, not live quotes. Confirm GPU variant specifications and enter your provider's current price in Advanced assumptions. Storage, networking, taxes and minimum billing units are excluded.

## Research

Reviewed October 10, 2026:

- [nanoGPT FLOPs implementation](https://github.com/karpathy/nanoGPT/blob/master/model.py#L289): verified source uses `6N + 12LHS` and 312 TFLOPS for A100 BF16.
- [PaLM Appendix B](https://arxiv.org/abs/2204.02311): reference cited by nanoGPT; direct paper access was blocked in this environment.
- [Hugging Face GPU training guide v4.46.3](https://huggingface.co/docs/transformers/v4.46.3/en/perf_train_gpu_one): source reviewed for training memory and performance considerations.
- NVIDIA and Lambda vendor pages were inaccessible (HTTP 403); hardware presets and rental rates are not represented as newly verified live data.

## Validation

`npm test` runs five numerical tests covering the formula, ideal and inefficient GPU scaling, overhead, epochs, optimizer memory and invalid inputs. The production build and Chromium smoke checks cover model selection, price updates, reset, token inputs, JSON export, methodology navigation and mobile overflow.

## Self-contained HTML edition

```sh
npm run build:standalone
```

Open `dist/compute.html` directly in a browser. It bundles React, icons, styles and calculations into one file and works offline, with no server or installation for the viewer. External reference links still require Internet access. This same file can be uploaded to an HTML-capable viewer or static host; embedding support depends on the viewer.
