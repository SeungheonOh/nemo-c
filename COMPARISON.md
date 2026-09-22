# C + Metal vs MLX: Nemotron 3.5 ASR Streaming 0.6B on Apple M4 Max

Same checkpoint (bf16), same streaming algorithm (cache-aware chunks, greedy RNNT), same clips. Both runtimes run the encoder on the GPU. Numbers are from `bench/run_bench.py`; each row is one run.

## Throughput, file mode, as fast as possible

`enc/chunk` is the mean wall time of one encoder chunk (subsampling + 24 layers + prompt), `joint step` the mean time of one RNNT joint evaluation (prediction net when needed + joint + argmax). `speed` is audio seconds per wall second for the whole streaming loop.

| clip | latency | impl | speed | enc/chunk | enc max | joint step | dec/chunk max | first token | tokens |
|---|---|---|---|---|---|---|---|---|---|
| fox | 80 ms | c | 11.1x | 6.7 ms | 7.4 ms | 0.24 ms | 1.6 ms | 45 ms | 86 |
| fox | 80 ms | mlx | 4.0x | 18.6 ms | 40.9 ms | 0.72 ms | 4.5 ms | 132 ms | 86 |
| fox | 320 ms | c | 28.8x | 8.8 ms | 9.7 ms | 0.23 ms | 2.1 ms | 20 ms | 83 |
| fox | 320 ms | mlx | 11.8x | 21.9 ms | 44.1 ms | 0.63 ms | 5.7 ms | 46 ms | 83 |
| fox | 560 ms | c | 36.9x | 11.2 ms | 12.1 ms | 0.22 ms | 3.5 ms | 14 ms | 87 |
| fox | 560 ms | mlx | 18.3x | 22.0 ms | 44.8 ms | 0.61 ms | 9.7 ms | 23 ms | 87 |
| fox | 1120 ms | c | 43.8x | 18.1 ms | 19.1 ms | 0.22 ms | 6.0 ms | 26 ms | 84 |
| fox | 1120 ms | mlx | 29.3x | 22.5 ms | 46.5 ms | 0.60 ms | 16.5 ms | 23 ms | 84 |
| mixed | 80 ms | c | 11.3x | 6.5 ms | 7.6 ms | 0.25 ms | 1.1 ms | 44 ms | 92 |
| mixed | 80 ms | mlx | 4.1x | 18.3 ms | 40.1 ms | 0.70 ms | 3.1 ms | 125 ms | 92 |
| mixed | 320 ms | c | 29.7x | 9.0 ms | 10.0 ms | 0.22 ms | 2.2 ms | 20 ms | 91 |
| mixed | 320 ms | mlx | 12.3x | 21.9 ms | 44.5 ms | 0.63 ms | 5.9 ms | 46 ms | 91 |
| mixed | 560 ms | c | 38.1x | 11.5 ms | 12.2 ms | 0.21 ms | 3.5 ms | 14 ms | 91 |
| mixed | 560 ms | mlx | 19.2x | 22.1 ms | 44.5 ms | 0.62 ms | 10.6 ms | 23 ms | 91 |
| mixed | 1120 ms | c | 45.3x | 18.4 ms | 19.2 ms | 0.21 ms | 5.7 ms | 26 ms | 91 |
| mixed | 1120 ms | mlx | 31.1x | 22.7 ms | 46.1 ms | 0.60 ms | 17.6 ms | 23 ms | 91 |
| pauses | 80 ms | c | 11.3x | 6.7 ms | 7.4 ms | 0.24 ms | 1.2 ms | 32 ms | 99 |
| pauses | 80 ms | mlx | 4.1x | 18.3 ms | 36.0 ms | 0.71 ms | 3.9 ms | 91 ms | 99 |
| pauses | 320 ms | c | 30.4x | 8.9 ms | 9.6 ms | 0.21 ms | 2.4 ms | 20 ms | 98 |
| pauses | 320 ms | mlx | 12.6x | 21.8 ms | 43.7 ms | 0.63 ms | 6.2 ms | 46 ms | 98 |
| pauses | 560 ms | c | 39.0x | 11.5 ms | 12.2 ms | 0.21 ms | 3.6 ms | 14 ms | 104 |
| pauses | 560 ms | mlx | 20.2x | 21.9 ms | 44.5 ms | 0.60 ms | 9.3 ms | 23 ms | 104 |
| pauses | 1120 ms | c | 47.5x | 18.4 ms | 19.2 ms | 0.19 ms | 4.8 ms | 25 ms | 104 |
| pauses | 1120 ms | mlx | 32.8x | 22.8 ms | 46.3 ms | 0.59 ms | 13.9 ms | 23 ms | 104 |
| narrate | 80 ms | c | 11.2x | 6.7 ms | 7.3 ms | 0.24 ms | 1.4 ms | 39 ms | 182 |
| narrate | 80 ms | mlx | 4.1x | 18.3 ms | 41.0 ms | 0.68 ms | 3.6 ms | 107 ms | 182 |
| narrate | 320 ms | c | 29.2x | 9.3 ms | 9.9 ms | 0.22 ms | 2.5 ms | 22 ms | 178 |
| narrate | 320 ms | mlx | 12.5x | 22.1 ms | 45.6 ms | 0.60 ms | 6.2 ms | 46 ms | 178 |
| narrate | 560 ms | c | 37.6x | 12.1 ms | 12.8 ms | 0.21 ms | 3.8 ms | 16 ms | 179 |
| narrate | 560 ms | mlx | 19.3x | 22.6 ms | 45.7 ms | 0.62 ms | 11.1 ms | 24 ms | 179 |
| narrate | 1120 ms | c | 45.1x | 19.2 ms | 19.8 ms | 0.21 ms | 6.0 ms | 27 ms | 178 |
| narrate | 1120 ms | mlx | 31.0x | 23.2 ms | 47.3 ms | 0.65 ms | 25.5 ms | 23 ms | 178 |

Across all 16 settings: C encoder chunk takes 0.53x the MLX time (geometric mean 0.50x), C joint step 0.35x, and the C streaming loop is 2.09x faster end to end (>1 means C faster).

## Live-rate load (file paced at real time, like the microphone)

CPU is percent of one core (M4 Max has 16), sampled every 0.5 s after start-up. RSS is resident memory. `compute RTF` is (encoder + decoder time) / audio time. Per-chunk times are higher than in the throughput table for both runtimes because the GPU idles between chunks and clocks down; what matters here is that both stay far below real time.

| clip | latency | impl | CPU avg | CPU peak | RSS | compute RTF | enc/chunk | joint step |
|---|---|---|---|---|---|---|---|---|
| narrate | 560 ms | c | 0.9% | 1.9% | 1295 MB | 0.040 | 18.8 ms | 0.35 ms |
| narrate | 560 ms | mlx | 5.1% | 13.6% | 1336 MB | 0.077 | 34.5 ms | 0.91 ms |
| narrate | 80 ms | c | 1.5% | 3.1% | 1290 MB | 0.161 | 12.3 ms | 0.41 ms |
| narrate | 80 ms | mlx | 18.4% | 24.4% | 1339 MB | 0.286 | 21.8 ms | 0.79 ms |

## Start-up

| impl | interpreter / imports | model load | warm-up (kernel compile) | total to first audio |
|---|---|---|---|---|
| c | 0 ms | 225 ms | 41 ms | 266 ms |
| mlx | 593 ms | 212 ms | 175 ms | 979 ms |

## Output parity

`tools/compare.py` checks the C dumps against MLX dumps (`tools/mlx_reference.py`): log-mel frames, every post-prompt encoder frame, and the emitted token ids. On fox, mixed, pauses and narrate at 560 ms, fox at 80 ms and narrate at 1120 ms, in en-US and auto, the token sequences are identical and encoder outputs agree to about 3e-5 absolute (bf16 weights, f32 activations in both).

