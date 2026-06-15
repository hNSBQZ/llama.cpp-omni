# Duplex Perf Test

This directory contains a duplex performance harness for `tools/omni`.

The harness drives the existing duplex session path and relies on internal
profiling hooks in the encoder thread, duplex LLM thread, TTS thread, and T2W
thread. Those hooks emit `[DUPLEX_PERF]` / `[DUPLEX_GPU]` lines through
`tools/omni/omni-perf.cpp`.

## Build

```sh
cmake --build build --target llama-omni-perf-duplex
```

Use the build directory name used by your local configuration.

## Run

```sh
./build/bin/llama-omni-perf-duplex \
  -m ./models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf \
  --omni \
  --test tools/omni/assets/test_case/omni_test_case/omni_test_case_ 9 \
  --stream-interval 1000 \
  -o ./tools/omni/output \
  2>&1 | tee duplex-perf.log
```

Useful modes:

- `--stream-interval 0`: back-to-back pressure test.
- `--stream-interval 1000`: approximate realtime streaming input.
- `--no-tts`: measure duplex frame decisions without waiting for TTS/T2W audio.
- `--timeout-ms <ms>`: per-frame wait timeout.
- `--llama-perf`: also print llama context perf counters.

The binary emits stable lines such as:

```text
[DUPLEX_PERF] seq=0 stage=duplex.llm.prefill event=start chunk=1 t_ms=1.000 dur_ms=-1.000 rss_mb=100.00 gpu_used_mb=NA gpu_total_mb=NA n_past=123 detail="frame_id=1,user_seq=1"
[DUPLEX_FRAME] seq=1 frame_id=1 ok=1 is_speak=0 prefill_submit_ms=1.234 decode_ms=10.000 e2e_ms=20.000 n_past=123 text=""
[DUPLEX_PERF_CASE] event=end chunks=9 pushed=9 completed=9 total_ms=9000.000 avg_prefill_submit_ms=1.000 avg_decode_ms=10.000 avg_e2e_ms=20.000 speak=3 listen=6
```

## Analyze

```sh
python3 tools/omni/test/perf-test/analyze_duplex_perf.py \
  --log duplex-perf.log \
  --out-dir duplex-perf-report
```

The analyzer is the full duplex perf parser. It understands `[DUPLEX_PERF]`,
`[DUPLEX_GPU]`, and frame summary lines.
