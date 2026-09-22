#!/usr/bin/env python3
"""MLX-side benchmark with the same per-chunk structure as the C runtime.

Feeds a WAV as fast as possible (or paced) through StreamingLogMelSpectrogram ->
ConformerStreamingState -> apply_prompt -> greedy RNNT, timing the encoder chunk and the
decoder separately, exactly like nemoasr-c --file. Prints one JSON line.

uv run --project ~/fun/nemoasr python bench/mlx_bench.py ref/fox.wav --latency 560 [--realtime]
"""
import argparse, json, resource, sys, time

import mlx.core as mx
import numpy as np

LAT = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--latency", type=int, default=560, choices=sorted(LAT))
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b")
    a = ap.parse_args()

    t0 = time.perf_counter()
    from mlx_audio.stt import load
    from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
    from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState
    from mlx_audio.stt.utils import load_audio
    import_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    model = load(a.model)
    load_ms = (time.perf_counter() - t0) * 1000
    att = [56, LAT[a.latency]]
    model.default_att_context_size = att
    sr = model.preprocessor_config.sample_rate
    audio = np.asarray(load_audio(a.wav, sr), dtype=np.float32)
    duration = audio.size / sr

    def run(audio, measure):
        fe = StreamingLogMelSpectrogram(model.preprocessor_config)
        st = ConformerStreamingState(model.encoder, att_context_size=att)
        blank, last, hidden = model.blank_id, model.blank_id, None
        enc_ms, dec_ms, enc_max, dec_max, chunks, steps, tokens = 0.0, 0.0, 0.0, 0.0, 0, 0, 0
        first_token = None
        chunk_samples = st.chunk_mel * model.preprocessor_config.hop_length
        piece = sr // 10 if a.realtime else chunk_samples
        start = time.perf_counter()
        pos = 0
        while pos < audio.size:
            seg = audio[pos:pos + piece]
            pos += piece
            final = pos >= audio.size
            if a.realtime:
                target = start + pos / sr
                while time.perf_counter() < target:
                    time.sleep(0.001)
            mel = fe.push(mx.array(seg), final=final)
            te = time.perf_counter()
            outs = []
            for encoded in st.push(mel, final=final):
                prompted = model.apply_prompt(encoded, a.language)
                mx.eval(prompted)
                outs.append(prompted)
            e_ms = (time.perf_counter() - te) * 1000
            if outs:
                enc_ms += e_ms; enc_max = max(enc_max, e_ms); chunks += len(outs)
            for prompted in outs:
                td = time.perf_counter()
                t = 0; symbols = 0
                while t < prompted.shape[1]:
                    feature = prompted[:, t:t + 1]
                    token = mx.array([[last]], dtype=mx.int32) if last != blank else None
                    output, (h, c) = model.decoder(token, hidden)
                    hid = (h.astype(feature.dtype), c.astype(feature.dtype))
                    pred = int(mx.argmax(model.joint(feature, output.astype(feature.dtype))))
                    steps += 1
                    if pred != blank:
                        last = pred; hidden = hid; tokens += 1; symbols += 1
                        if first_token is None:
                            first_token = (time.perf_counter() - start) * 1000
                    if pred == blank or symbols >= model.max_symbols:
                        t += 1; symbols = 0
                d_ms = (time.perf_counter() - td) * 1000
                dec_ms += d_ms; dec_max = max(dec_max, d_ms)
        wall = time.perf_counter() - start
        return dict(wall_s=wall, enc_ms=enc_ms, enc_mean_ms=enc_ms / max(chunks, 1), enc_max_ms=enc_max,
                    dec_ms=dec_ms, dec_per_step_ms=dec_ms / max(steps, 1), dec_max_ms=dec_max, steps=steps,
                    chunks=chunks, tokens=tokens, first_token_ms=first_token)

    tw = time.perf_counter()
    run(np.zeros(sr * 3, np.float32), measure=False)  # warm-up (kernel compilation)
    warmup_ms = (time.perf_counter() - tw) * 1000
    r = run(audio, measure=True)
    r.update(impl="mlx", wav=a.wav, latency_ms=a.latency, realtime=a.realtime, audio_s=duration, import_ms=import_ms,
             load_ms=load_ms, warmup_ms=warmup_ms, rtf=(r["enc_ms"] + r["dec_ms"]) / 1000 / duration,
             speed_x=duration / r["wall_s"], max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
