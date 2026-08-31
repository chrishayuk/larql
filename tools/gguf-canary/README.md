# Cross-runtime logit canary

Two gates, deliberately narrow, on one frozen token sequence:

    TOKENIZATION  both runtimes tokenize the fixed prompt to identical ids
    EXECUTION     full logits at every position, compared per position

    same VINDEX3 representation (stored NVFP4 pack)
            ↓ frozen token ids
            ├── larql vindex3 exec --backend production-nvfp4
            │       --representation-source stored --logit-dump A.f32
            └── llama_logits (libllama) --ids ... --out B.f32
                  ↓
            canary.py A.f32 B.f32 --ids ... --vocab N --record out.json

`llama_logits.cpp` is a tiny libllama program: token ids in, raw
[positions, vocab] f32 out — no sampling, no chat template, no CLI
presentation. Build:

    clang++ -O2 -std=c++17 -I $LLAMA/include -I $LLAMA/ggml/include \
        llama_logits.cpp -L $LLAMA/build-nvfp4/bin -lllama -lggml \
        -lggml-base -Wl,-rpath,$LLAMA/build-nvfp4/bin -o llama_logits

Bit-identical logits are NOT the bar — different kernels and
accumulation orders move low bits legitimately. The bar (predeclared in
canary.py::TOL): top-1 at every position, top-5 overlap >= 80%,
KL <= 0.05 nats, cosine >= 0.995, and no systematic divergence.

`canary-2026-08-31.json` is the recorded first run: 12 positions,
100% top-1, 100% top-5 everywhere, KL max 0.00028 nats.
