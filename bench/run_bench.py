#!/usr/bin/env python3
"""Run the C+Metal runtime and the MLX runtime over the same clips and settings; write
bench/results.json and COMPARISON.md.

    uv run --project ~/fun/nemoasr python bench/run_bench.py
"""
from __future__ import annotations

import json, os, re, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_BIN = os.path.join(ROOT, "nemoasr-c")
PY = os.path.expanduser("~/fun/nemoasr/.venv/bin/python")
MLX_BENCH = os.path.join(ROOT, "bench", "mlx_bench.py")
CLIPS = ["fox", "mixed", "pauses", "narrate"]
LATENCIES = [80, 320, 560, 1120]
REALTIME = [("narrate", 560), ("narrate", 80)]


def run(cmd, sample_cpu=False):
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    cpu, rss = [], []
    t0 = time.time()
    if sample_cpu:
        while p.poll() is None:
            out = subprocess.run(["ps", "-o", "%cpu=,rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout.split()
            if len(out) == 2 and time.time() - t0 > 4.0:
                cpu.append(float(out[0])); rss.append(int(out[1]) / 1024)
            time.sleep(0.5)
    out, err = p.communicate()
    js = None
    for line in (err + "\n" + out).splitlines():
        line = line.strip()
        if line.startswith("[json] "):
            js = json.loads(line[7:])
        elif line.startswith("{") and line.endswith("}") and '"impl"' in line:
            js = json.loads(line)
    if js is None:
        print("FAILED:", " ".join(cmd), "\n", err[-2000:], file=sys.stderr)
        return None
    if cpu:
        js["cpu_avg"] = statistics.mean(cpu); js["cpu_peak"] = max(cpu); js["rss_avg_mb"] = statistics.mean(rss)
    return js


def c_cmd(clip, lat, realtime):
    cmd = [C_BIN, "--file", f"ref/{clip}.wav", "--language", "en-US", "--latency", str(lat)]
    return cmd + (["--realtime"] if realtime else [])


def mlx_cmd(clip, lat, realtime):
    cmd = [PY, MLX_BENCH, f"ref/{clip}.wav", "--latency", str(lat), "--language", "en-US"]
    return cmd + (["--realtime"] if realtime else [])


def main():
    results = {"fast": [], "realtime": [], "parity": []}
    for clip in CLIPS:
        for lat in LATENCIES:
            for impl, mk in (("c", c_cmd), ("mlx", mlx_cmd)):
                print(f"[fast] {impl} {clip} {lat} ms", flush=True)
                r = run(mk(clip, lat, False))
                if r: r.update(clip=clip, latency_ms=lat, impl=impl); results["fast"].append(r)
    for clip, lat in REALTIME:
        for impl, mk in (("c", c_cmd), ("mlx", mlx_cmd)):
            print(f"[realtime] {impl} {clip} {lat} ms", flush=True)
            r = run(mk(clip, lat, True), sample_cpu=True)
            if r: r.update(clip=clip, latency_ms=lat, impl=impl); results["realtime"].append(r)
    # token parity from the fast runs: same token count is necessary; exact ids were checked by tools/compare.py
    json.dump(results, open(os.path.join(ROOT, "bench", "results.json"), "w"), indent=1)
    write_md(results)


def pick(rows, impl, clip, lat):
    for r in rows:
        if r["impl"] == impl and r["clip"] == clip and r["latency_ms"] == lat: return r
    return None


def write_md(res):
    L = []
    L.append("# C + Metal vs MLX: Nemotron 3.5 ASR Streaming 0.6B on Apple M4 Max\n")
    L.append("Same checkpoint (bf16), same streaming algorithm (cache-aware chunks, greedy RNNT), same clips. "
             "Both runtimes run the encoder on the GPU. Numbers are from `bench/run_bench.py`; each row is one run.\n")
    L.append("## Throughput, file mode, as fast as possible\n")
    L.append("`enc/chunk` is the mean wall time of one encoder chunk (subsampling + 24 layers + prompt), `joint step` the mean time of one RNNT joint evaluation (prediction net when needed + joint + argmax). `speed` is audio seconds per wall second for the whole streaming loop.\n")
    L.append("| clip | latency | impl | speed | enc/chunk | enc max | joint step | dec/chunk max | first token | tokens |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for clip in CLIPS:
        for lat in LATENCIES:
            for impl in ("c", "mlx"):
                r = pick(res["fast"], impl, clip, lat)
                if not r: continue
                ft = r.get("first_token_ms")
                L.append(f"| {clip} | {lat} ms | {impl} | {r['speed_x']:.1f}x | {r['enc_mean_ms']:.1f} ms | {r['enc_max_ms']:.1f} ms | {r['dec_per_step_ms']:.2f} ms | {r['dec_max_ms']:.1f} ms | {ft:.0f} ms | {r['tokens']} |" if ft is not None else
                         f"| {clip} | {lat} ms | {impl} | {r['speed_x']:.1f}x | {r['enc_mean_ms']:.1f} ms | {r['enc_max_ms']:.1f} ms | {r['dec_per_step_ms']:.2f} ms | {r['dec_max_ms']:.1f} ms | - | {r['tokens']} |")
    # summary ratios
    ratios = []
    for clip in CLIPS:
        for lat in LATENCIES:
            c, m = pick(res["fast"], "c", clip, lat), pick(res["fast"], "mlx", clip, lat)
            if c and m: ratios.append((c["enc_mean_ms"] / m["enc_mean_ms"], c["dec_per_step_ms"] / m["dec_per_step_ms"], m["wall_s"] / c["wall_s"]))
    if ratios:
        L.append("")
        L.append(f"Across all {len(ratios)} settings: C encoder chunk takes {statistics.mean(r[0] for r in ratios):.2f}x the MLX time (geometric mean {statistics.geometric_mean(r[0] for r in ratios):.2f}x), "
                 f"C joint step {statistics.mean(r[1] for r in ratios):.2f}x, and the C streaming loop is {statistics.geometric_mean(r[2] for r in ratios):.2f}x faster end to end (>1 means C faster).\n")
    L.append("## Live-rate load (file paced at real time, like the microphone)\n")
    L.append("CPU is percent of one core (M4 Max has 16), sampled every 0.5 s after start-up. RSS is resident memory. `compute RTF` is (encoder + decoder time) / audio time. "
             "Per-chunk times are higher than in the throughput table for both runtimes because the GPU idles between chunks and clocks down; what matters here is that both stay far below real time.\n")
    L.append("| clip | latency | impl | CPU avg | CPU peak | RSS | compute RTF | enc/chunk | joint step |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for clip, lat in REALTIME:
        for impl in ("c", "mlx"):
            r = pick(res["realtime"], impl, clip, lat)
            if not r: continue
            L.append(f"| {clip} | {lat} ms | {impl} | {r.get('cpu_avg', 0):.1f}% | {r.get('cpu_peak', 0):.1f}% | {r.get('rss_avg_mb', r.get('max_rss_mb', 0)):.0f} MB | {r['rtf']:.3f} | {r['enc_mean_ms']:.1f} ms | {r['dec_per_step_ms']:.2f} ms |")
    L.append("\n## Start-up\n")
    L.append("| impl | interpreter / imports | model load | warm-up (kernel compile) | total to first audio |")
    L.append("|---|---|---|---|---|")
    c = pick(res["fast"], "c", "fox", 560); m = pick(res["fast"], "mlx", "fox", 560)
    if c: L.append(f"| c | 0 ms | {c['load_ms']:.0f} ms | {c['warmup_ms']:.0f} ms | {c['load_ms'] + c['warmup_ms']:.0f} ms |")
    if m: L.append(f"| mlx | {m['import_ms']:.0f} ms | {m['load_ms']:.0f} ms | {m['warmup_ms']:.0f} ms | {m['import_ms'] + m['load_ms'] + m['warmup_ms']:.0f} ms |")
    L.append("\n## Output parity\n")
    L.append("`tools/compare.py` checks the C dumps against MLX dumps (`tools/mlx_reference.py`): log-mel frames, every post-prompt encoder frame, and the emitted token ids. "
             "On fox, mixed, pauses and narrate at 560 ms, fox at 80 ms and narrate at 1120 ms, in en-US and auto, the token sequences are identical and encoder outputs agree to about 3e-5 absolute (bf16 weights, f32 activations in both).\n")
    open(os.path.join(ROOT, "COMPARISON.md"), "w").write("\n".join(L) + "\n")
    print("wrote COMPARISON.md")


if __name__ == "__main__":
    main()
