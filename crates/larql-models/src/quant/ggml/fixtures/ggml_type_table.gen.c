// Dump ggml's ENTIRE type table — id, name, block geometry — from
// `ggml_get_type_traits` itself.
//
// A type id is a shared ABI contract exactly like a byte layout, and it
// is the half that internal round-trip tests cannot check: every caller
// inside this workspace passes the same constant in both directions, so
// a transposed id cancels out and looks correct. It only fails when the
// value crosses to another implementation.
//
// It did. `TYPE_Q8_0` was 6 (upstream's Q5_0) and `TYPE_Q5_0` was 8
// (upstream's Q8_0), so GGUF ingest decoded either as the other — wrong
// values and a wrong block stride. This fixture is the foreign reference
// that makes that class of error impossible to reintroduce silently.
//
// Build:
//   clang -O2 -o gen ggml_type_table.gen.c -I<llama.cpp>/ggml/include \
//     -L<llama.cpp>/build/bin -lggml -lggml-base -lggml-cpu
#include <stdio.h>
#include "ggml.h"

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "w");
    fprintf(f, "{\n \"source\": \"ggml_get_type_traits, llama.cpp\",\n \"types\": [\n");
    int first = 1;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const struct ggml_type_traits *tr = ggml_get_type_traits((enum ggml_type) t);
        if (!tr || !tr->type_name) continue;
        if (!first) fprintf(f, ",\n");
        first = 0;
        fprintf(f,
                "  {\"id\": %d, \"name\": \"%s\", \"blck_size\": %lld, \"type_size\": %zu, "
                "\"is_quantized\": %s}",
                t, tr->type_name, (long long) tr->blck_size, tr->type_size,
                tr->is_quantized ? "true" : "false");
    }
    fprintf(f, "\n ]\n}\n");
    fclose(f);
    printf("wrote %s\n", argv[1]);
    return 0;
}
