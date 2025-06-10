// Copyright (c) 2025, Vibe Inc

#include "thumbhash/thumbhash.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CH_R 0
#define CH_G 1
#define CH_B 2
#define CH_A 3
#define CH_L 0
#define CH_P 1
#define CH_Q 2
#define TB_MAX_AC_DIM 6

typedef float data_channel[TB_DATA_DIM * TB_DATA_DIM];

static float CT[TB_MAX_AC_DIM][TB_DATA_DIM];

struct context_t {
  data_channel lpqa[4];
  bool has_alpha;
};

static void init_cos_table() {
  if (CT[0][0] != 0) {
    return;
  }
  float pi_n = M_PI / TB_DATA_DIM;
  for (size_t u = 0; u < TB_MAX_AC_DIM; ++u) {
    for (size_t x = 0; x < TB_DATA_DIM; ++x) {
      CT[u][x] = cosf(pi_n * (x + 0.5f) * u);
    }
  }
}

static inline float C(int k) {
  if (k == 0) {
    return 1;
  }
  return sqrt(2.0F);
}

// Convert value from [-1,1] range to [0,1] range
static inline float to_unit(float v) {
  return v * 0.5F + 0.5F;
}

// Convert value from [0,1] range to [-1,1] range
static inline float from_unit(float v) {
  return v * 2.0F - 1.0F;
}

static inline uint32_t getChannelI(uint32_t pixel, uint8_t channel) {
  return (pixel >> (channel * 8)) & 0xFF;
}

static inline float getChannel(uint32_t pixel, uint8_t channel) {
  return getChannelI(pixel, channel) / 255.0F;
}

static inline void setChannel(float value, uint32_t* pixel, uint8_t channel) {
  uint8_t clamped = (uint8_t)(fminf(fmaxf(0.0F, value), 1.0F) * 255.0F);
  *pixel |= (clamped << (channel * 8));
}

static inline uint8_t quantize(float in, size_t bits, size_t shift) {
  assert(shift + bits <= 8);
  return (uint8_t)(in * (float)((1 << bits) - 1)) << shift;
}

static inline float dequantize(uint8_t in, size_t bits, size_t shift) {
  assert(shift + bits <= 8);
  uint8_t mask = ((1 << bits) - 1);
  uint8_t value = (in >> shift) & mask;
  return (float)value / (float)mask;
}

static int thumbhashp_encode_to_context(struct context_t* restrict ctx,
                                        const uint32_t* restrict data) {
  uint32_t avg[4] = {0};
  const size_t data_size = TB_DATA_DIM * TB_DATA_DIM;
  const size_t data_size_shift = 14;

  for (size_t i = 0; i < data_size; ++i) {
    uint32_t r = getChannelI(data[i], CH_R);
    uint32_t g = getChannelI(data[i], CH_G);
    uint32_t b = getChannelI(data[i], CH_B);
    uint32_t a = getChannelI(data[i], CH_A);
    avg[CH_R] += (r * a) >> 8;
    avg[CH_G] += (g * a) >> 8;
    avg[CH_B] += (b * a) >> 8;
    avg[CH_A] += a;
  }
  avg[CH_A] >>= data_size_shift;  // [0 - 255]
  avg[CH_R] >>= data_size_shift;  // [0 - 255]
  avg[CH_G] >>= data_size_shift;  // [0 - 255]
  avg[CH_B] >>= data_size_shift;  // [0 - 255]

  ctx->has_alpha = avg[CH_A] < 255;

  for (size_t i = 0; i < data_size; ++i) {
    float r = getChannel(data[i], CH_R);
    float g = getChannel(data[i], CH_G);
    float b = getChannel(data[i], CH_B);
    float a = getChannel(data[i], CH_A);
    // do a ATOP mix over avg
    float mix_r = (float)(avg[CH_R] / 255.0F) * (1 - a) + r * a;
    float mix_g = (float)(avg[CH_G] / 255.0F) * (1 - a) + g * a;
    float mix_b = (float)(avg[CH_B] / 255.0F) * (1 - a) + b * a;
    ctx->lpqa[CH_L][i] = (mix_r + mix_g + mix_b) / 3.0F;
    ctx->lpqa[CH_P][i] = to_unit((mix_r + mix_g) * 0.5F - mix_b);
    ctx->lpqa[CH_Q][i] = to_unit(mix_r - mix_g);
    ctx->lpqa[CH_A][i] = a;
  }
  return 0;
}

/**
 * @brief Performs a 2D DCT-II transform on image channel data and stores
 * compressed coefficients.
 *
 * The function computes the Discrete Cosine Transform (type II) of the input
 * channel data, quantizes the coefficients, and stores them in a compact format
 * suitable for thumbhashp.
 *
 * @param channel Input image channel data (TB_DATA_DIM x TB_DATA_DIM
 * array)
 * @param dim Dimension of the AC coefficients block to store (must be <=
 * TB_MAX_AC_DIM)
 * @param[out] dc Output DC coefficient (quantized to TB_DC_BITS bits)
 * @param[out] scale Output scale factor for AC coefficients (quantized to
 * TB_SCALE_BITS bits)
 * @param[out] acs Output array for packed AC coefficients (each stored in
 * TB_AC_BITS bits)
 *
 * @note The function stores AC coefficients in a triangular pattern (u + v <
 * dim)
 * @note A 15% scale boost is applied to compensate for quantization errors
 */
static void channelDCT(const data_channel channel, size_t dim,
                       uint8_t* restrict dc, uint8_t* restrict scale,
                       uint8_t* restrict acs) {
  const size_t N = TB_DATA_DIM;

  float coeff[dim * dim];

  int coeff_size = 0;
  float s = 0.0F;
  for (size_t v = 0; v < dim; ++v) {
    for (size_t u = 0; u + v < dim; ++u) {
      float sum = 0.0F;
      for (size_t y = 0; y < N; ++y) {
        for (size_t x = 0; x < N; ++x) {
          sum += C(u) * C(v) * CT[u][x] * CT[v][y] * channel[y * N + x];
        }
      }
      sum /= N * N;
      if (v == 0 && u == 0) {
        *dc = quantize(sum, TB_DC_BITS, 0);
      } else {
        float ac = sum;
        coeff[coeff_size++] = ac;
        s = fmaxf(s, fabsf(ac));
      }
    }
  }

  for (size_t i = 0; i < coeff_size; ++i) {
    // scale to [0, 1] for storage
    acs[i / 2] |=
        quantize(to_unit(coeff[i] / s), TB_AC_BITS, TB_AC_BITS * (i & 1));
  }
  // compensate for quantization error
  s *= 1.15F;
  *scale = quantize(s, TB_SCALE_BITS, 0);
}

/**
 * @brief Performs a 2D Inverse Discrete Cosine Transform (type III) using
 * stored coefficients.
 *
 * Reconstructs spatial domain data from quantized DCT coefficients stored in
 * thumbhashp format. The function handles both DC and AC coefficients, applying
 * proper scaling and dequantization.
 *
 * @param[out] channel Output spatial data (TB_DATA_DIM x TB_DATA_DIM
 * array)
 * @param encoded_acs Packed AC coefficients array (each stored in TB_AC_BITS
 * bits)
 * @param dim Dimension of the AC coefficients block (must match encoding
 * dimension)
 * @param dc Quantized DC coefficient (stored in TB_DC_BITS bits)
 * @param scale Quantized scale factor for AC coefficients (stored in
 * TB_SCALE_BITS bits)
 *
 * @note The function reconstructs coefficients in a triangular pattern (u + v <
 * dim)
 */
static void channelIDCT(data_channel channel,
                        const uint8_t* restrict encoded_acs, size_t dim,
                        uint8_t dc, uint8_t scale) {
  const size_t N = TB_DATA_DIM;

  float coeff[dim * dim];

  // Unscale and place the AC coefficients into their correct positions
  int ac_idx = 0;
  float s = dequantize(scale, TB_SCALE_BITS, 0);
  for (size_t v = 0; v < dim; ++v) {
    for (size_t u = 0; u + v < dim; ++u) {
      if (u == 0 && v == 0) {
        coeff[0] = dequantize(dc, TB_DC_BITS, 0) * N;
      } else {
        float ac_val = from_unit(dequantize(encoded_acs[ac_idx / 2], TB_AC_BITS,
                                            (ac_idx % 2) * TB_AC_BITS));
        // Unscale from [0, 1] to [-1, 1], then multiply by scale
        coeff[v * dim + u] = ac_val * s * N;
        ac_idx++;
      }
    }
  }

  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      float sum = 0.0f;
      for (size_t v = 0; v < dim; ++v) {
        for (size_t u = 0; u < dim; ++u) {
          if (u + v < dim) {
            float cu = 1.0F / C(u);
            float cv = 1.0F / C(v);
            sum += cu * cv * coeff[v * dim + u] * CT[u][x] * CT[v][y];
          }
        }
      }
      channel[y * N + x] = sum / N;
    }
  }
}

void thumbhashp_init(struct thumbhashp_t* restrict hash) {
  memset(hash, 0, sizeof(struct thumbhashp_t));
}

uint32_t* thumbhashp_allocate_data() {
  return malloc(sizeof(uint32_t) * TB_DATA_DIM * TB_DATA_DIM);
}

int thumbhashp_encode(struct thumbhashp_t* restrict hash,
                      const uint32_t* restrict data) {
  init_cos_table();
  struct context_t* ctx = calloc(1, sizeof(struct context_t));
  if (ctx == NULL) {
    return -1;
  }
  if (thumbhashp_encode_to_context(ctx, data) != 0) {
    return -1;
  }
  {  // L channel
    uint8_t dc = 0;
    uint8_t scale = 0;
    channelDCT(ctx->lpqa[CH_L], TB_L_AC_DIM, &dc, &scale, hash->l_ac);
    hash->l_dc = dc;
    hash->l_scale = scale;
  }
  {  // P channel
    uint8_t dc = 0;
    uint8_t scale = 0;
    channelDCT(ctx->lpqa[CH_P], TB_P_AC_DIM, &dc, &scale, hash->p_ac);
    hash->p_dc = dc;
    hash->p_scale = scale;
  }
  {  // Q channel
    uint8_t dc = 0;
    uint8_t scale = 0;
    channelDCT(ctx->lpqa[CH_Q], TB_Q_AC_DIM, &dc, &scale, hash->q_ac);
    hash->q_dc = dc;
    hash->q_scale = scale;
  }
  if (ctx->has_alpha) {  // A channel
    uint8_t dc = 0;
    uint8_t scale = 0;
    channelDCT(ctx->lpqa[CH_A], TB_A_AC_DIM, &dc, &scale, hash->a_ac);
    hash->a_dc = dc;
    hash->a_scale = scale;
  }
  return 0;
}

int thumbhashp_decode(const struct thumbhashp_t* restrict hash,
                      uint32_t* restrict data) {
  init_cos_table();
  struct context_t* ctx = calloc(1, sizeof(struct context_t));
  if (ctx == NULL) {
    return -1;
  }
  ctx->has_alpha = thumbhashp_has_alpha(hash);

  // Reconstruct the LPQA channels

  channelIDCT(ctx->lpqa[CH_L], hash->l_ac, TB_L_AC_DIM, hash->l_dc,
              hash->l_scale);
  channelIDCT(ctx->lpqa[CH_P], hash->p_ac, TB_P_AC_DIM, hash->p_dc,
              hash->p_scale);
  channelIDCT(ctx->lpqa[CH_Q], hash->q_ac, TB_Q_AC_DIM, hash->q_dc,
              hash->q_scale);
  if (ctx->has_alpha) {
    channelIDCT(ctx->lpqa[CH_A], hash->a_ac, TB_A_AC_DIM, hash->a_dc,
                hash->a_scale);
  }

  // Convert LPQA back to RGBA
  for (size_t i = 0; i < TB_DATA_DIM * TB_DATA_DIM; i++) {
    float l = ctx->lpqa[CH_L][i];
    float p = from_unit(ctx->lpqa[CH_P][i]);
    float q = from_unit(ctx->lpqa[CH_Q][i]);
    float a = ctx->has_alpha ? ctx->lpqa[CH_A][i] : 1.0F;

    // Convert LPQ to RGB
    float b = l - 2.0F / 3.0F * p;
    float r = (3.0F * l - b + q) / 2.0F;
    float g = r - q;

    setChannel(r, &data[i], CH_R);
    setChannel(g, &data[i], CH_G);
    setChannel(b, &data[i], CH_B);
    setChannel(a, &data[i], CH_A);
  }

  free(ctx);
  return 0;
}

bool thumbhashp_has_alpha(const struct thumbhashp_t* RESTRICT hash) {
  return hash->a_scale != 0 || hash->a_dc != 0;
}

void* thumbhashp_to_bytes(const struct thumbhashp_t* restrict hash) {
  size_t len = thumbhashp_bytes_len(hash);
  void* bytes = malloc(len);
  memcpy(bytes, hash, len);
  return bytes;
}

size_t thumbhashp_bytes_len(const struct thumbhashp_t* hash) {
  if (thumbhashp_has_alpha(hash)) {
    return sizeof(struct thumbhashp_t);
  }
  return sizeof(struct thumbhashp_t) - 10;
}

void thumbhashp_from_bytes(struct thumbhashp_t* RESTRICT hash,
                           const void* restrict bytes) {
  const struct thumbhashp_t* data = bytes;
  if (thumbhashp_has_alpha(data)) {  // has_alpha
    memcpy(hash, data, sizeof(struct thumbhashp_t));
  } else {
    memcpy(hash, data, sizeof(struct thumbhashp_t) - 10);
  }
}
