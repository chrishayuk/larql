// Emit a foreign-reference K-quant fixture using ggml's OWN encoder and
// decoder. Nothing in this file is transcribed from LARQL: the bytes and
// the decoded values are whatever llama.cpp produces.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ggml.h"

#define N 512   // two 256-element super-blocks, so multi-block layout is exercised

static void emit(FILE *f, enum ggml_type t, const float *src, int first) {
    const struct ggml_type_traits *tr = ggml_get_type_traits(t);
    size_t nbytes = ggml_row_size(t, N);
    void *q = malloc(nbytes);
    memset(q, 0, nbytes);
    // ggml's own quantizer, one row of N elements.
    size_t wrote = ggml_quantize_chunk(t, src, q, 0, 1, N, NULL);
    float *back = malloc(sizeof(float) * N);
    tr->to_float(q, back, N);

    if (!first) fprintf(f, ",\n");
    fprintf(f, "  {\n");
    fprintf(f, "   \"encoding\": \"%s\",\n", tr->type_name);
    // ggml's OWN numeric id for this type. Carried because a type id is
    // as much a shared contract as a byte layout: LARQL had Q8_0 and Q5_0
    // transposed against upstream and nothing noticed, because the id
    // never left the workspace until it crossed this FFI.
    fprintf(f, "   \"ggml_type\": %d,\n", (int) t);
    fprintf(f, "   \"blck_size\": %lld,\n", (long long) tr->blck_size);
    fprintf(f, "   \"type_size\": %zu,\n", tr->type_size);
    fprintf(f, "   \"elements\": %d,\n", N);
    fprintf(f, "   \"bytes_written\": %zu,\n", wrote);
    fprintf(f, "   \"quantised_hex\": \"");
    for (size_t i = 0; i < nbytes; i++) fprintf(f, "%02x", ((unsigned char *) q)[i]);
    fprintf(f, "\",\n");
    fprintf(f, "   \"dequantised_bits\": [");
    for (int i = 0; i < N; i++) {
        uint32_t b; memcpy(&b, &back[i], 4);
        fprintf(f, "%s%u", i ? "," : "", b);
    }
    fprintf(f, "]\n  }");
    free(q); free(back);
}

int main(int argc, char **argv) {
    // Deterministic, asymmetric, and not a ramp: a symmetric or monotone
    // input lets a wrong nibble order or a swapped scale/min still look
    // right. Integer arithmetic scaled by a power of two, so the values
    // are bit-exact and portable.
    float *src = malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        int a = (i * 37) % 251 - 125;
        int b = (i * 101) % 67;
        src[i] = (float) a / 32.0f + (float) b / 512.0f;
    }

    FILE *f = fopen(argv[1], "w");
    fprintf(f, "{\n \"source\": \"llama.cpp ggml, generated independently of LARQL\",\n");
    fprintf(f, " \"input_bits\": [");
    for (int i = 0; i < N; i++) {
        uint32_t b; memcpy(&b, &src[i], 4);
        fprintf(f, "%s%u", i ? "," : "", b);
    }
    fprintf(f, "],\n \"cases\": [\n");
    emit(f, GGML_TYPE_Q8_0, src, 1);
    emit(f, GGML_TYPE_Q6_K, src, 0);
    emit(f, GGML_TYPE_Q4_K, src, 0);
    fprintf(f, "\n ]\n}\n");
    fclose(f);
    printf("wrote %s\n", argv[1]);
    return 0;
}
