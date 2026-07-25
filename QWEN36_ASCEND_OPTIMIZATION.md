# Qwen3.6 Ascend Graph-Mode Optimization Guide

This document describes the Qwen3.6 optimizations and experimental switches in
the `optimize-qwen3_6-long-context-v1` branch. The branch targets graph-mode
serving on Ascend and is based on vLLM 0.13.0.

> Status: the graph-replay stability fixes and Mamba/GDN slot-mapping skip are
> enabled by default. NPU slot mapping, fused decode GDN, mixed-batch GDN
> splitting, and MoE GroupedMatmul tuning remain experimental and are disabled
> by default.

## Validated scope

The latest controlled validation used the following scope. Other combinations
require separate correctness and performance testing.

| Item | Validated configuration |
|---|---|
| Models | Qwen3.6-35B-A3B for the current worktree validation |
| Hardware | 4 x Ascend 910B, tensor parallel size 4 |
| Software | vLLM 0.13.0, torch-npu 2.8.0.post2, CANN 8.5.0 |
| Precision | BF16 |
| Serving | vLLM V1, chunked prefill, no prefix caching |
| Graph request | `cudagraph_mode=FULL` |
| Effective graph mode | `FULL_AND_PIECEWISE`: FULL decode and PIECEWISE mixed/prefill |
| Main benchmark | concurrency 64, `max_num_seqs=64`, `max_model_len=8192`, TP=4 |

`GDNAttentionBackend` reports `AttentionCGSupport.UNIFORM_BATCH`, so a requested
`FULL` mode is expected to produce this warning:

```text
CUDAGraphMode.FULL is not supported with GDNAttentionBackend backend
(support: AttentionCGSupport.UNIFORM_BATCH); setting
cudagraph_mode=FULL_AND_PIECEWISE
```

This is not a startup failure. Decode still uses a FULL graph; mixed and
prefill batches use PIECEWISE capture.

## Implemented changes

### ACL graph replay

- Updates captured paged-attention task parameters with runtime sequence
  lengths before replay.
- Resolves graph update hooks from all initialized attention backend groups and
  fails early if captured mutable attention tasks have no updater.
- Captures decode with a sufficiently large sequence length to reserve the
  maximum paged-attention workspace used by the validated context range.
- Synchronizes a FULL graph replay only when its captured entry has mutable
  attention or conv1d task parameters. `VLLM_FL_FORCE_ACLGRAPH_REPLAY_SYNC=1`
  remains an emergency compatibility switch.
- Preserves graph-capture state around capture success and failure paths.

Before this branch, the Ascend graph path synchronized every eligible FULL
replay. The current default is automatic conditional synchronization, which
keeps the required ordering barrier while avoiding it for graph entries without
runtime task updates.

### Qwen3.6 GDN path

- Caches the AscendC conv1d transposed weight layout after checkpoint loading
  instead of rebuilding it on each forward.
- Uses sync-free CPU metadata for the fresh-prefill PTO decision.
- Keeps the fused `RMSNormGated` path enabled, with an independent rollback
  switch.
- Fixes fused recurrent decode replay for padded graph batches:
  - zero-length padded sequences return before reading a state index;
  - padded `-1` state indices are retained through replay;
  - scalar state selection replaces an Ascend Triton `vsel` pattern that could
    generate an invalid device-memory access;
  - state-index length and in-place state contiguity are validated.
- Keeps fused decode GDN opt-in because stability is fixed but throughput has
  not improved in the current A/B.
- Keeps mixed decode/prefill GDN splitting opt-in because the first screening
  run regressed significantly.

### Slot mapping

- Skips CPU slot-mapping computation and H2D commit for Mamba/GDN KV-cache
  groups, which do not consume paged-attention slot mapping. The validated
  35B-A3B layout consequently computes slot mapping for 1 of 4 cache groups.
- Includes an Ascend Triton slot-mapping kernel adapted from vllm-ascend PR
  #12096. It reuses the runner's existing device `query_start_loc` and
  `seq_lens` buffers and falls back to CPU for context parallelism.
- Keeps the NPU kernel disabled by default. Screening results were inconsistent
  across short and long contexts, including regressions on the long-context
  case.

### MoE path

- Uses native `torch_npu.npu_swiglu` in the AscendC and torch-npu fused expert
  paths.
- Avoids an unnecessary int64 round trip for routing IDs before the Ascend MoE
  routing operator.
- Adds an experimental GroupedMatmul `tuning_config` hint for static small-token
  TP-only decode shapes. It is disabled by default and is excluded for expert
  parallelism, symbolic token counts, and more than 256 tokens.

## Feature flags

The simple boolean switches below use exact `0` and `1` values unless noted.

| Variable | Default | Effect | Recommendation |
|---|---:|---|---|
| `VLLM_FL_FORCE_ACLGRAPH_REPLAY_SYNC` | `0` | `0`: synchronize only graph entries with mutable runtime tasks. `1`: force the replay barrier for every eligible FULL graph entry. | Keep `0`; use `1` only to diagnose ordering-sensitive graph failures. |
| `VLLM_FL_SKIP_MAMBA_SLOT_MAPPING` | `1` | Skip slot mapping for Mamba/GDN cache groups. Only exact `0` restores all-group mapping. | Keep `1` for the validated Qwen3.6 hybrid-cache layout. |
| `VLLM_FL_ENABLE_NPU_SLOT_MAPPING` | `0` | Run selected-group slot mapping with the Ascend Triton kernel. | Keep `0` until a multi-run A/B and CPU-vs-NPU numerical test pass. |
| `VLLM_FL_NPU_SLOT_MAPPING_DECODE_ONLY` | `1` | When NPU slot mapping is enabled, restrict it to pure one-token decode. Exact `0` also enables prefill/mixed batches. | Keep `1` during experiments. |
| `VLLM_FL_ENABLE_FUSED_DECODE_GDN` | `0` | Enable the fused Triton decode GDN update. | Keep `0`; current A/B is neutral. |
| `VLLM_FL_DISABLE_FUSED_DECODE_GDN` | `0` | Legacy force-disable switch; `1` overrides the enable switch. | Set `1` in controlled baseline scripts. |
| `VLLM_FL_ENABLE_MIXED_GDN_SPLIT` | `0` | Split a mixed batch into recurrent decode and chunked prefill GDN work. | Keep `0`; the first screening run regressed. |
| `VLLM_FL_DISABLE_FUSED_RMSNORM_GATED` | `0` | Set `1` to restore the decomposed RMSNormGated path. | Keep `0` unless isolating a correctness or performance issue. |
| `VLLM_FL_DISABLE_PTO_GDN` | `0` | Set `1` to disable the PTO fresh-prefill megakernel. | Leave at its existing default for normal serving; set `1` when isolating decode-only changes. |
| `VLLM_FL_DISABLE_ASCENDC_GDN` | `0` | Set `1` to fall back from AscendC GDN operators. | Keep `0` on the validated stack. |
| `VLLM_FL_DISABLE_ASCENDC_MOE` | `0` | Set `1` to fall back from the AscendC MoE path. | Keep `0` on the validated stack. |
| `VLLM_FL_MOE_GMM_TUNING` | off | Accepts `auto` or a positive integer; empty, `0`, `off`, `false`, and `none` disable it. | Keep off until the tuning sweep, reverse-order rerun, and output-parity checks complete. |

## Recommended stable configuration

The branch defaults already select the conservative paths. The following
explicit values make a decode-throughput baseline reproducible and prevent
inherited shell variables from enabling experimental work:

```bash
export VLLM_FL_FORCE_ACLGRAPH_REPLAY_SYNC=0
export VLLM_FL_SKIP_MAMBA_SLOT_MAPPING=1
export VLLM_FL_ENABLE_NPU_SLOT_MAPPING=0
export VLLM_FL_NPU_SLOT_MAPPING_DECODE_ONLY=1

export VLLM_FL_ENABLE_FUSED_DECODE_GDN=0
export VLLM_FL_DISABLE_FUSED_DECODE_GDN=1
export VLLM_FL_ENABLE_MIXED_GDN_SPLIT=0
export VLLM_FL_DISABLE_FUSED_RMSNORM_GATED=0

unset VLLM_FL_MOE_GMM_TUNING
```

For a decode-isolation benchmark, also use:

```bash
export VLLM_FL_DISABLE_PTO_GDN=1
```

That last setting is a benchmark control, not a general claim that PTO should
be disabled in production.

## ATB ABI requirement

Qwen3.6-35B-A3B startup performs multimodal dummy profiling. On the validated
image, loading the non-ABI1 `libmki.so` caused a symbol/ABI failure. Preload the
ATB C++ ABI1 library before launching the server:

```bash
export VLLM_FL_ATB_MKI_LIB=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libmki.so
test -r "${VLLM_FL_ATB_MKI_LIB}"
export LD_PRELOAD="${VLLM_FL_ATB_MKI_LIB}${LD_PRELOAD:+:${LD_PRELOAD}}"
```

If jemalloc is also preloaded, retain both libraries. The validated effective
value was:

```text
/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib/libmki.so
```

## Example server launch

Install this checkout in editable mode or place it first on `PYTHONPATH`, then
start vLLM with the FL plugin selected:

```bash
export MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export GEMS_VENDOR=ascend
export TRITON_ALL_BLOCKS_PARALLEL=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

vllm serve "${MODEL_PATH}" \
  --served-model-name qwen3.6 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.55 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --no-async-scheduling \
  --compilation-config '{
    "cudagraph_mode": "FULL",
    "backend": "eager",
    "custom_ops": ["all"],
    "pass_config": {
      "fuse_norm_quant": false,
      "fuse_act_quant": false,
      "fuse_attn_quant": false,
      "enable_sp": false,
      "fuse_gemm_comms": false,
      "fuse_allreduce_rms": false
    }
  }'
```

The generic Ascend quick-start section still recommends eager execution for
released configurations. The graph command above is specific to this branch
and the validated vLLM 0.13.0 stack.

## Benchmark method

Use an exclusive four-device allocation. Start a fresh server for every switch
that is parsed at module import, warm it up once, and run comparison cases
serially. Do not run another model on devices 0-3 during an A/B.

Example warmup:

```bash
vllm bench serve \
  --backend openai-chat \
  --model qwen3.6 \
  --tokenizer "${MODEL_PATH}" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --max-concurrency 64 \
  --num-prompts 64 \
  --ignore-eos
```

Example formal case:

```bash
vllm bench serve \
  --backend openai-chat \
  --model qwen3.6 \
  --tokenizer "${MODEL_PATH}" \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 1024 \
  --max-concurrency 64 \
  --num-prompts 256 \
  --ignore-eos
```

Use output token throughput as the primary objective, but retain TTFT, TPOT,
ITL, successful/failed request counts, source commit, environment variables,
and server logs. Repeat promising changes in both A/B and B/A order; a single
fixed-order run is only a screen.

## Validation results

These runs used `d3865fa` plus the then-uncommitted functional changes
described in this guide; they must not be attributed to `d3865fa` alone. The
final branch commit only adds review/documentation adjustments on top of the
same functional paths.

### Fused decode GDN state-selection fix

The same server replayed graph batches at 64, 63, and 33 requests without a
507035/MPU/AICore fault or Python traceback:

| Case | Successful / failed | Output tokens/s |
|---|---:|---:|
| input 128, output 128, prompts 64 | 64 / 0 | 1148.80 |
| input 128, output 128, prompts 63 | 63 / 0 | 1129.97 |
| input 128, output 128, prompts 33 | 33 / 0 | 767.27 |

This validates the padded graph-replay boundary path. It does not yet prove
token-by-token and final SSM-state numerical parity against the unfused path.

### Fused decode GDN throughput A/B

Both variants used TP=4, concurrency 64, 256 prompts, PTO disabled, NPU slot
mapping disabled, and MoE GMM tuning disabled.

| Case | Fused off | Fused on | Change |
|---|---:|---:|---:|
| input 1024, output 1024 | 1210.24 | 1211.96 | +0.14% |
| input 4096, output 1024 | 668.79 | 665.14 | -0.55% |

Both deltas are below 1% and are treated as run noise. Fused decode GDN remains
disabled by default.

### Mamba/GDN slot-mapping skip screen

A same-order single-run screen compared all-group CPU mapping with selected
attention-group CPU mapping:

| Case | All groups | Skip Mamba/GDN | Change |
|---|---:|---:|---:|
| input 1024, output 1024 | 1028.91 | 1073.21 | +4.31% |
| input 4096, output 1024 | 585.32 | 629.55 | +7.56% |

This supports keeping the Mamba/GDN skip enabled, but the measurements should
still be repeated in reverse order. The optional NPU kernel produced mixed
results, including a decode-only short-case regression, and remains disabled.

## Known limitations

- `FULL` is currently resolved to `FULL_AND_PIECEWISE` by the GDN backend;
  mixed/prefill batches are not FULL graph captures.
- The fused decode GDN smoke verifies stability, not complete numerical parity,
  and its throughput result is neutral.
- Mixed GDN splitting has no validated benefit and is known to regress its
  first screening case.
- NPU slot mapping lacks a full CPU-vs-NPU numerical suite for standard,
  hybrid-block, graph-padding, speculative, and context-parallel cases.
- MoE GMM tuning is an unfinished experiment. A promising single screening run
  is insufficient to select a default.
- Older Qwen3.6-27B exploratory runs did not consistently record the current
  source hash and are not treated as validation of this worktree.
- The slot-mapping skip changes behavior for hybrid-cache models globally;
  models beyond the validated Qwen3.6 layout require correctness coverage.
- The fused recurrent dummy-program workaround is specific to the observed
  Ascend 910B/CANN 8.5 Triton compilation behavior.
- Results from non-exclusive cards, different warmup histories, or old runs
  without a source hash must not be combined into a claimed performance gain.

## Troubleshooting and rollback

1. Confirm the server logs show the intended branch and feature flags.
2. Treat the `FULL_AND_PIECEWISE` downgrade warning as expected.
3. For an ATB symbol or multimodal dummy-profile startup failure, verify that
   the ABI1 `libmki.so` appears in the effective `LD_PRELOAD`.
4. For a graph replay hang or ordering fault, retry once with
   `VLLM_FL_FORCE_ACLGRAPH_REPLAY_SYNC=1`. A throughput decrease is expected;
   this is a diagnostic/compatibility mode.
5. For a fused GDN fault, set `VLLM_FL_DISABLE_FUSED_DECODE_GDN=1` and keep
   `VLLM_FL_ENABLE_MIXED_GDN_SPLIT=0`.
6. For a slot-mapping issue, set `VLLM_FL_ENABLE_NPU_SLOT_MAPPING=0`; if needed,
   also set `VLLM_FL_SKIP_MAMBA_SLOT_MAPPING=0` to restore all-group CPU mapping.
7. For a MoE issue, unset `VLLM_FL_MOE_GMM_TUNING` and, if required, set
   `VLLM_FL_DISABLE_ASCENDC_MOE=1` to isolate the fallback path.

## Next validation priorities

1. Complete the MoE GMM `off/1/2/4` screen, select at most one candidate, then
   repeat in reverse order and run the formal input-1024/input-4096 cases.
2. Add fused-vs-unfused token output and final SSM-state parity tests for graph
   batches 64, 63, and 33.
3. Add CPU-vs-NPU slot-mapping numerical tests before reconsidering the NPU
   kernel default.
4. Repeat the Mamba/GDN slot-mapping skip A/B in B/A order on both 27B and
   35B-A3B.
5. Profile the remaining decode bottleneck only after each candidate passes
   correctness and multi-run throughput gates.
