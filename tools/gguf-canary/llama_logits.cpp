// Full-logit dump from llama.cpp, for the cross-runtime canary.
//
// Deliberately narrow: no sampling, no chat template, no CLI
// presentation. Token ids in, raw logits out — the same
// [positions, vocab] f32 layout `larql vindex3 exec --logit-dump`
// writes, so the two runtimes are compared on identical terms.
//
//   llama_logits MODEL --tokenize "text"     print ids, comma-separated
//                                            (add_special=false,
//                                             parse_special=false — the
//                                             raw encoding, matching the
//                                             container tokenizer's)
//   llama_logits MODEL --ids 1,2,3 --out F   decode the fixed sequence,
//                                            dump every position's
//                                            logits to F
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> parse_ids(const char *s) {
    std::vector<llama_token> ids;
    std::string cur;
    for (const char *p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) ids.push_back((llama_token) atol(cur.c_str()));
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur += *p;
        }
    }
    return ids;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL (--tokenize TEXT | --ids I,I,... --out FILE)\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    const char *tokenize_text = nullptr;
    const char *ids_arg = nullptr;
    const char *out_path = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--tokenize") && i + 1 < argc) tokenize_text = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_arg = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
    }

    llama_log_set([](ggml_log_level, const char *text, void *) { fputs(text, stderr); }, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    // The tokenize gate needs only the vocabulary — skip 18 GB of weights.
    mparams.vocab_only = tokenize_text != nullptr;
    llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "FAIL: model load\n"); return 1; }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (tokenize_text) {
        std::vector<llama_token> toks(strlen(tokenize_text) + 16);
        int n = llama_tokenize(vocab, tokenize_text, (int) strlen(tokenize_text),
                               toks.data(), (int) toks.size(),
                               /*add_special*/ false, /*parse_special*/ false);
        if (n < 0) { fprintf(stderr, "FAIL: tokenize\n"); return 1; }
        for (int i = 0; i < n; ++i) printf("%s%d", i ? "," : "", toks[i]);
        printf("\n");
        llama_model_free(model);
        return 0;
    }

    if (!ids_arg || !out_path) { fprintf(stderr, "FAIL: need --ids and --out\n"); return 2; }
    std::vector<llama_token> ids = parse_ids(ids_arg);
    const int n = (int) ids.size();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n + 8;
    cparams.n_batch = n;
    cparams.n_ubatch = n;
    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "FAIL: context\n"); return 1; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // every position's logits, not just the last
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: decode\n"); return 1; }

    FILE *f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "FAIL: open %s\n", out_path); return 1; }
    for (int i = 0; i < n; ++i) {
        const float *logits = llama_get_logits_ith(ctx, i);
        if (!logits) { fprintf(stderr, "FAIL: logits at %d\n", i); return 1; }
        fwrite(logits, sizeof(float), (size_t) n_vocab, f);
    }
    fclose(f);
    fprintf(stderr, "wrote [%d, %d] f32 to %s\n", n, n_vocab, out_path);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
