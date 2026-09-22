#!/usr/bin/env python3
"""Dump reference artifacts from the MLX implementation for parity testing of the C port.

Run from the nemoasr venv:
    uv run --project ~/fun/nemoasr python tools/mlx_reference.py samples/fox.wav ref/fox --latency 560 --language en-US

Writes into <out_dir>/:
    mel.f32            full-utterance log-mel, (T, 128) float32, via the streaming frontend
    enc_chunk_XXX.f32  post-prompt encoder output per streaming chunk, (c, 1024) float32
    enc_all.f32        all encoder frames concatenated
    tokens.txt         emitted token ids, one per line (blank excluded), then the text
    pre_encode_win0.f32   output of pre_encode for the first mel window fed to the encoder
    meta.json          shapes and settings
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np
from mlx_audio.stt import load
from mlx_audio.stt.models.nemotron_asr import tokenizer
from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState
from mlx_audio.stt.utils import load_audio

LAT = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--latency", type=int, default=560, choices=sorted(LAT))
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--block-ms", type=int, default=100, help="feed granularity, like the mic")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load(args.model)
    att = [56, LAT[args.latency]]
    model.default_att_context_size = att
    sr = model.preprocessor_config.sample_rate
    audio = np.asarray(load_audio(args.wav, sr), dtype=np.float32)

    # --- 1. full mel via the same streaming frontend the session uses ---------------
    fe = StreamingLogMelSpectrogram(model.preprocessor_config)
    block = sr * args.block_ms // 1000
    mels = []
    for s in range(0, audio.size, block):
        m = fe.push(mx.array(audio[s : s + block]))
        if m.shape[1]:
            mels.append(np.asarray(m[0], dtype=np.float32))
    m = fe.flush()
    if m.shape[1]:
        mels.append(np.asarray(m[0], dtype=np.float32))
    mel = np.concatenate(mels, axis=0)
    mel.tofile(os.path.join(args.out_dir, "mel.f32"))

    # --- 2. streaming encoder, chunk by chunk, mirroring NemotronStreamingSession -----
    enc_state = ConformerStreamingState(model.encoder, att_context_size=att)
    fe2 = StreamingLogMelSpectrogram(model.preprocessor_config)
    chunks = []
    first_win = {}

    orig_pre = model.encoder.pre_encode

    class PreHook:
        def __call__(self, x, lengths):
            out, l = orig_pre(x, lengths)
            if "x" not in first_win:
                first_win["x"] = np.asarray(x[0], dtype=np.float32)
                first_win["out"] = np.asarray(out[0], dtype=np.float32)
            return out, l

    model.encoder.pre_encode = PreHook()
    limit = enc_state.chunk_mel * model.preprocessor_config.hop_length  # samples per ingest, like session._ingest
    pos = 0
    while pos < audio.size:
        piece = audio[pos : pos + limit]
        pos += limit
        final = pos >= audio.size
        melc = fe2.push(mx.array(piece), final=final)
        for encoded in enc_state.push(melc, final=final):
            prompted = model.apply_prompt(encoded, args.language)
            mx.eval(prompted)
            chunks.append(np.asarray(prompted[0], dtype=np.float32))
    model.encoder.pre_encode = orig_pre
    for i, c in enumerate(chunks):
        c.tofile(os.path.join(args.out_dir, f"enc_chunk_{i:03d}.f32"))
    enc_all = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1024), np.float32)
    enc_all.tofile(os.path.join(args.out_dir, "enc_all.f32"))
    first_win["x"].tofile(os.path.join(args.out_dir, "pre_encode_win0_in.f32"))
    first_win["out"].tofile(os.path.join(args.out_dir, "pre_encode_win0_out.f32"))

    # --- 3. greedy RNNT over the chunks, exactly like session.step ---------------------
    blank = model.blank_id
    last = blank
    hidden = None
    ids = []
    for prompted in chunks:
        feats = mx.array(prompted)[None]
        t = 0
        symbols = 0
        while t < feats.shape[1]:
            feature = feats[:, t : t + 1]
            token = mx.array([[last]], dtype=mx.int32) if last != blank else None
            output, (h, c) = model.decoder(token, hidden)
            hid = (h.astype(feature.dtype), c.astype(feature.dtype))
            pred = int(mx.argmax(model.joint(feature, output.astype(feature.dtype))))
            if pred != blank:
                last = pred
                hidden = hid
                ids.append(pred)
                symbols += 1
            if pred == blank or (model.max_symbols is not None and symbols >= model.max_symbols):
                t += 1
                symbols = 0
    text = tokenizer.decode(ids, model.vocabulary)
    with open(os.path.join(args.out_dir, "tokens.txt"), "w") as fh:
        fh.write(" ".join(map(str, ids)) + "\n")
        fh.write(text.strip() + "\n")

    meta = {
        "wav": args.wav, "latency_ms": args.latency, "att_context_size": att, "language": args.language,
        "samples": int(audio.size), "mel_frames": int(mel.shape[0]),
        "chunks": [int(c.shape[0]) for c in chunks], "encoder_frames": int(enc_all.shape[0]),
        "pre_encode_win0_in": list(first_win["x"].shape), "pre_encode_win0_out": list(first_win["out"].shape),
        "num_tokens": len(ids), "text": text.strip(),
        "prompt_index": model._resolve_prompt_index(args.language), "blank_id": blank,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("chunks",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
