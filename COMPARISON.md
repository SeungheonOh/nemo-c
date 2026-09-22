# C + Metal vs MLX: Nemotron 3.5 ASR Streaming 0.6B on Apple M4 Max

Same checkpoint (bf16), same streaming algorithm (cache-aware chunks, greedy RNNT), same clips. Both runtimes run the encoder on the GPU. Numbers are from `bench/run_bench.py`; each row is one run.

## Throughput, file mode, as fast as possible

`enc/chunk` is the mean wall time of one encoder chunk (subsampling + 24 layers + prompt), `joint step` the mean time of one RNNT joint evaluation (prediction net when needed + joint + argmax). `speed` is audio seconds per wall second for the whole streaming loop.

| clip | latency | impl | speed | enc/chunk | enc max | joint step | dec/chunk max | first token | tokens |
|---|---|---|---|---|---|---|---|---|---|
| fox | 80 ms | c | 15.2x | 4.7 ms | 5.6 ms | 0.24 ms | 1.5 ms | 34 ms | 86 |
| fox | 80 ms | mlx | 4.0x | 18.5 ms | 42.2 ms | 0.72 ms | 5.6 ms | 127 ms | 86 |
| fox | 320 ms | c | 46.5x | 5.3 ms | 5.9 ms | 0.25 ms | 2.4 ms | 13 ms | 83 |
| fox | 320 ms | mlx | 11.9x | 21.7 ms | 43.4 ms | 0.63 ms | 5.9 ms | 46 ms | 83 |
| fox | 560 ms | c | 69.9x | 5.5 ms | 5.9 ms | 0.26 ms | 2.9 ms | 8 ms | 87 |
| fox | 560 ms | mlx | 18.1x | 22.3 ms | 43.8 ms | 0.61 ms | 9.6 ms | 23 ms | 87 |
| fox | 1120 ms | c | 96.2x | 6.7 ms | 7.2 ms | 0.29 ms | 4.4 ms | 12 ms | 84 |
| fox | 1120 ms | mlx | 30.0x | 22.5 ms | 45.3 ms | 0.57 ms | 15.3 ms | 24 ms | 84 |
| mixed | 80 ms | c | 15.4x | 4.7 ms | 5.5 ms | 0.22 ms | 0.9 ms | 34 ms | 92 |
| mixed | 80 ms | mlx | 4.1x | 18.2 ms | 42.0 ms | 0.66 ms | 2.6 ms | 127 ms | 92 |
| mixed | 320 ms | c | 49.2x | 5.4 ms | 6.3 ms | 0.23 ms | 1.5 ms | 12 ms | 91 |
| mixed | 320 ms | mlx | 12.3x | 22.0 ms | 43.8 ms | 0.63 ms | 6.4 ms | 45 ms | 91 |
| mixed | 560 ms | c | 72.9x | 5.6 ms | 6.1 ms | 0.27 ms | 2.6 ms | 8 ms | 91 |
| mixed | 560 ms | mlx | 19.6x | 21.8 ms | 44.1 ms | 0.58 ms | 9.8 ms | 22 ms | 91 |
| mixed | 1120 ms | c | 104.3x | 6.9 ms | 7.3 ms | 0.28 ms | 3.8 ms | 12 ms | 91 |
| mixed | 1120 ms | mlx | 30.8x | 23.0 ms | 45.4 ms | 0.60 ms | 15.5 ms | 23 ms | 91 |
| pauses | 80 ms | c | 15.2x | 4.8 ms | 6.0 ms | 0.23 ms | 0.9 ms | 25 ms | 99 |
| pauses | 80 ms | mlx | 4.1x | 18.3 ms | 36.5 ms | 0.66 ms | 3.3 ms | 87 ms | 99 |
| pauses | 320 ms | c | 50.1x | 5.4 ms | 6.2 ms | 0.23 ms | 1.7 ms | 13 ms | 98 |
| pauses | 320 ms | mlx | 12.5x | 22.1 ms | 47.0 ms | 0.63 ms | 6.8 ms | 48 ms | 98 |
| pauses | 560 ms | c | 74.7x | 5.7 ms | 6.3 ms | 0.29 ms | 3.2 ms | 9 ms | 104 |
| pauses | 560 ms | mlx | 19.8x | 22.3 ms | 43.8 ms | 0.61 ms | 9.2 ms | 22 ms | 104 |
| pauses | 1120 ms | c | 117.9x | 6.6 ms | 7.2 ms | 0.24 ms | 2.5 ms | 12 ms | 104 |
| pauses | 1120 ms | mlx | 32.2x | 23.3 ms | 45.4 ms | 0.60 ms | 14.0 ms | 24 ms | 104 |
| narrate | 80 ms | c | 15.0x | 4.9 ms | 6.2 ms | 0.23 ms | 1.6 ms | 31 ms | 182 |
| narrate | 80 ms | mlx | 4.1x | 18.2 ms | 42.1 ms | 0.67 ms | 3.8 ms | 112 ms | 182 |
| narrate | 320 ms | c | 48.2x | 5.5 ms | 6.5 ms | 0.26 ms | 2.2 ms | 14 ms | 178 |
| narrate | 320 ms | mlx | 12.3x | 22.1 ms | 46.4 ms | 0.64 ms | 6.5 ms | 44 ms | 178 |
| narrate | 560 ms | c | 75.0x | 5.7 ms | 6.3 ms | 0.27 ms | 2.4 ms | 9 ms | 179 |
| narrate | 560 ms | mlx | 19.5x | 22.4 ms | 44.6 ms | 0.62 ms | 10.3 ms | 24 ms | 179 |
| narrate | 1120 ms | c | 109.2x | 6.9 ms | 7.5 ms | 0.29 ms | 3.8 ms | 13 ms | 178 |
| narrate | 1120 ms | mlx | 31.9x | 23.4 ms | 47.0 ms | 0.59 ms | 18.6 ms | 25 ms | 178 |

Across all 16 settings: C encoder chunk takes 0.26x the MLX time (geometric mean 0.26x), C joint step 0.41x, and the C streaming loop is 3.72x faster end to end (>1 means C faster).

## Live-rate load (file paced at real time, like the microphone)

CPU is percent of one core (M4 Max has 16), sampled every 0.5 s after start-up. RSS is resident memory. `compute RTF` is (encoder + decoder time) / audio time. Per-chunk times are higher than in the throughput table for both runtimes because the GPU idles between chunks and clocks down; what matters here is that both stay far below real time.

| clip | latency | impl | CPU avg | CPU peak | RSS | compute RTF | enc/chunk | joint step |
|---|---|---|---|---|---|---|---|---|
| narrate | 560 ms | c | 0.7% | 1.6% | 1350 MB | 0.029 | 14.6 ms | 0.45 ms |
| narrate | 560 ms | mlx | 4.9% | 10.6% | 1338 MB | 0.080 | 37.1 ms | 0.79 ms |
| narrate | 80 ms | c | 2.1% | 3.3% | 1348 MB | 0.126 | 9.4 ms | 0.44 ms |
| narrate | 80 ms | mlx | 18.3% | 25.7% | 1340 MB | 0.283 | 21.6 ms | 0.76 ms |

The live-rate penalty is GPU clock ramp: the GPU idles between chunks and starts each one at low clocks. `nemoasr-c --gpu-warm` keeps it busy while waiting for audio and brings live chunks back to the fast-path time (measured 14.2 -> 6.3 ms at 560 ms, 10.7 -> 5.5 ms at 80 ms) at a power cost, so it is off by default. Short warm-up bursts just before a chunk were measured and do not help; the governor needs sustained activity.


## Start-up

| impl | interpreter / imports | model load | warm-up (kernel compile) | total to first audio |
|---|---|---|---|---|
| c | 0 ms | 209 ms | 26 ms | 235 ms |
| mlx | 567 ms | 161 ms | 177 ms | 905 ms |

## Output parity

`tools/compare.py` checks the C dumps against MLX dumps (`tools/mlx_reference.py`): log-mel frames, every post-prompt encoder frame, and the emitted token ids. On fox, mixed, pauses and narrate at 560 ms, fox at 80 ms and narrate at 80 and 1120 ms, in en-US and auto, the token sequences are identical and encoder outputs agree to about 3e-5 absolute (bf16 weights, f32 activations in both). The parity checks were re-run after every kernel change.

## Optimisation history (encoder chunk, fast path, M4 Max)

| step | 80 ms | 560 ms | 1120 ms |
|---|---|---|---|
| first working version (one thread per output column) | 32 ms | 34 ms | 36 ms |
| SIMD-group cooperative GEMM, attention, joint | 7.1 | 12.3 | 20.3 |
| specialised + split-K MMA GEMM | 5.4 | 7.4 | 8.6 |
| append-only caches, fused QKV, batched joint | 5.2 | 7.0 | 8.2 |
| unified MMA policy, GLU + paired LN fusion | 5.6 | 6.4 | 7.8 |
| MMA for many-row GEMMs (subsampling) | 4.9 | 5.8 | 6.9 |
| attention: cooperative lanes, parallel softmax | 4.7 | 5.7 | 6.5 |

Remaining time at 560 ms: about 3.9 ms of GEMM against a DRAM floor near 2.9 ms (1.1 GB of bf16 weights per chunk), and about 1.7 ms of everything else, most of it the fixed cost of ~340 dispatches at ~3 us each.

