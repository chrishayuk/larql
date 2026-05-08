#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

typedef half ggml_half;

#define QK_TURBO4 128
typedef struct {
    ggml_half norm;
    ggml_half rnorm;
    uint8_t qs[QK_TURBO4 / 2];
} block_turbo4_0;
static_assert(sizeof(block_turbo4_0) == 68, "wrong turbo4_0 block size");

#define QK_PLANAR3 128
#define NL_PLANAR3 (QK_PLANAR3 / 16)
#define NL_PLANAR3_VEC (QK_PLANAR3 / 4)
typedef struct {
    ggml_half norm;
    uint8_t qs[QK_PLANAR3 / 4];
    uint8_t signs[QK_PLANAR3 / 8];
} block_planar3_0;
static_assert(
    sizeof(block_planar3_0) == sizeof(ggml_half) + QK_PLANAR3 / 4 + QK_PLANAR3 / 8,
    "wrong planar3_0 block size/padding"
);

#define QK_ISO3 128
#define NL_ISO3 (QK_ISO3 / 16)
#define NL_ISO3_VEC (QK_ISO3 / 4)
typedef struct {
    ggml_half norm;
    uint8_t qs[QK_ISO3 / 4];
    uint8_t signs[QK_ISO3 / 8];
} block_iso3_0;
static_assert(
    sizeof(block_iso3_0) == sizeof(ggml_half) + QK_ISO3 / 4 + QK_ISO3 / 8,
    "wrong iso3_0 block size/padding"
);

#define QK_PLANAR4 128
#define NL_PLANAR4 8
#define NL_PLANAR4_VEC 32
typedef block_turbo4_0 block_planar4_0;

#define QK_ISO4 128
#define NL_ISO4 8
#define NL_ISO4_VEC 32
typedef block_turbo4_0 block_iso4_0;
