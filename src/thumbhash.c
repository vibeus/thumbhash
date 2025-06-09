// Copyright (c) 2025, Vibe Inc

#include "thumbhash/thumbhash.h"
#include <assert.h>
#include <math.h>
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

typedef float data_channel[TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM];

// Helper for the DCT/IDCT normalization factor
static inline float C(int k) {
  return k == 0 ? 1.0F / sqrtf(2.0F) : 1.0F;
}

static uint32_t getChannelI(uint32_t pixel, uint8_t channel) {
  return (pixel >> (channel * 8)) & 0xFF;
}

static float getChannel(uint32_t pixel, uint8_t channel) {
  return getChannelI(pixel, channel) / 255.0F;
}

static void setChannel(float value, uint32_t* pixel, uint8_t channel) {
  uint8_t clamped = (uint8_t)(fminf(fmaxf(0.0F, value), 1.0F) * 255.0F);
  *pixel |= (clamped << (channel * 8));
}

struct context_t {
  _Bool has_alpha;
  data_channel* lpqa;
};

static void thumbhash_float_to_bitfield(float in, size_t bits, uint8_t* out,
                                        size_t shift) {
  assert(shift + bits <= 8);
  *out = *out | (uint8_t)(in * (float)((1 << bits) - 1)) << shift;
}

static void thumbhash_bitfield_to_float(uint8_t in, size_t bits, float* out,
                                        size_t shift) {
  assert(shift + bits <= 8);
  uint8_t mask = ((1 << bits) - 1);
  uint8_t value = (in >> shift) & mask;
  *out = (float)value / (float)mask;
}

static int thumbhash_encode_to_context(struct context_t* restrict ctx,
                                       const uint32_t* restrict data) {
  uint32_t avg[4] = {0};
  const size_t data_size = TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM;
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

  ctx->lpqa = calloc(4, sizeof(data_channel));
  if (ctx->lpqa == NULL) {
    return -1;
  }
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
    ctx->lpqa[CH_P][i] =
        ((mix_r + mix_g) * 0.5F - mix_b) * 0.5F + 0.5F;  // scale to [0, 1]
    ctx->lpqa[CH_Q][i] = (mix_r - mix_g) * 0.5F + 0.5F;  // scale to [0, 1]
    ctx->lpqa[CH_A][i] = a;
  }
  return 0;
}

/**
 * @brief Performs a standard 2D DCT-II on a channel and stores a limited
 * set of coefficients.
 *
 * @param channel The input spatial data (TB_SIZE_DATA_DIM x TB_SIZE_DATA_DIM).
 * @param dim The dimension of the square block of AC coefficients to store
 * (e.g., TB_L_AC_DIM).
 * @param dc The output DC coefficient, scaled to [0, 1].
 * @param scale The output scale factor for the AC coefficients.
 * @param acs The output array of selected AC coefficients, scaled to [0, 1].
 * @return The number of AC coefficients written, or -1 on error.
 */
static int thumbhash_DCT(data_channel channel, size_t dim, float* restrict dc,
                         float* restrict scale, float** restrict acs) {
  const size_t N = TB_SIZE_DATA_DIM;

  // Temporary matrix to hold the full set of DCT coefficients
  float* F = calloc(N * N, sizeof(float));
  if (F == NULL) {
    return -1;
  }

  // --- Perform standard 2D DCT (row-column decomposition) ---
  // 1. 1D DCT on each row
  for (size_t v = 0; v < N; ++v) {
    for (size_t u = 0; u < N; ++u) {
      float sum = 0.0f;
      for (size_t x = 0; x < N; ++x) {
        sum += channel[v * N + x] * cosf((float)M_PI / N * (x + 0.5f) * u);
      }
      F[v * N + u] = sum;
    }
  }

  // 2. 1D DCT on each column of the result from step 1
  float* temp_col = malloc(N * sizeof(float));
  if (temp_col == NULL) {
    free(F);
    return -1;
  }

  for (size_t u = 0; u < N; ++u) {
    // copy column u into a temporary buffer
    for (size_t v = 0; v < N; v++) {
      temp_col[v] = F[v * N + u];
    }

    for (size_t v = 0; v < N; v++) {
      float sum = 0.0f;
      for (size_t y = 0; y < N; ++y) {
        sum += temp_col[y] * cosf((float)M_PI / N * (y + 0.5f) * v);
      }
      // Apply normalization factors and store final coefficient
      F[v * N + u] = (2.0f / N) * C(u) * C(v) * sum;
    }
  }
  free(temp_col);

  // --- Extract, scale, and store the desired coefficients ---
  // DC coefficient (u=0, v=0)
  // The original range is [-1, 1] for luminance/chrominance, scale to [0, 1]
  // for storage.
  *dc = 0.5f + 0.5f * F[0];

  // AC coefficients
  // Count how many AC coeffs we will store based on the triangular region
  int count = 0;
  for (size_t v = 0; v < dim; ++v) {
    for (size_t u = 0; u < dim; ++u) {
      if (u == 0 && v == 0)
        continue;  // Skip DC
      // Use `u + v < dim` to select a triangular region of low-frequency
      // coefficients
      if (u + v < dim) {
        count++;
      }
    }
  }

  *acs = calloc(count, sizeof(float));
  if (*acs == NULL) {
    free(F);
    return -1;
  }

  *scale = 0.0f;
  int current_ac = 0;
  for (size_t v = 0; v < dim; ++v) {
    for (size_t u = 0; u < dim; ++u) {
      if (u == 0 && v == 0)
        continue;
      if (u + v < dim) {
        float val = F[v * N + u];
        *scale = fmaxf(*scale, fabsf(val));
        (*acs)[current_ac++] = val;
      }
    }
  }

  // Normalize the selected AC coefficients
  if (*scale > 0) {
    for (int i = 0; i < count; ++i) {
      // Scale to [-1, 1] and then to [0, 1] for storage
      (*acs)[i] = 0.5f + 0.5f * ((*acs)[i] / *scale);
    }
  }

  free(F);
  return count;
}

/**
 * @brief Performs a standard 2D IDCT-II using a limited set of stored
 * coefficients.
 *
 * @param channel The output spatial data (TB_SIZE_DATA_DIM x TB_SIZE_DATA_DIM).
 * @param encoded_acs The input array of stored AC coefficients.
 * @param dim The dimension of the square block of AC coefficients that were
 * stored (e.g., TB_L_AC_DIM).
 * @param dc The stored DC coefficient, in [0, 1] range.
 * @param scale The stored scale factor for the AC coefficients.
 * @return 0 on success, or -1 on error.
 */
static int thumbhash_IDCT(data_channel channel,
                          const uint8_t* restrict encoded_acs, size_t dim,
                          float dc, float scale) {
  const size_t N = TB_SIZE_DATA_DIM;

  // --- Reconstruct the coefficient matrix from the stored hash ---
  // Create a full coefficient matrix, initialized to zero
  float* F = calloc(N * N, sizeof(float));
  if (F == NULL) {
    return -1;
  }

  // Unscale and place DC coefficient
  F[0] = dc * 2.0f - 1.0f;  // from [0, 1] back to [-1, 1]

  // Unscale and place the AC coefficients into their correct positions
  int ac_idx = 0;
  for (size_t v = 0; v < dim; ++v) {
    for (size_t u = 0; u < dim; ++u) {
      if (u == 0 && v == 0)
        continue;
      if (u + v < dim) {
        float ac_val;
        // Unpack from bitfield
        thumbhash_bitfield_to_float(encoded_acs[ac_idx / 2], TB_AC_BITS,
                                    &ac_val, (ac_idx % 2) * TB_AC_BITS);
        // Unscale from [0, 1] to [-1, 1], then multiply by scale
        F[v * N + u] = (ac_val * 2.0f - 1.0f) * scale;
        ac_idx++;
      }
    }
  }

  // --- Perform standard 2D IDCT (row-column decomposition) ---
  // 1. 1D IDCT on each column
  float* temp_matrix = calloc(N * N, sizeof(float));
  if (temp_matrix == NULL) {
    free(F);
    return -1;
  }

  for (size_t u = 0; u < N; ++u) {
    for (size_t y = 0; y < N; ++y) {
      float sum = 0.0f;
      for (size_t v = 0; v < N; ++v) {
        sum += C(v) * F[v * N + u] * cosf((float)M_PI / N * (y + 0.5f) * v);
      }
      temp_matrix[y * N + u] = sum;
    }
  }

  // 2. 1D IDCT on each row of the result from step 1
  for (size_t y = 0; y < N; ++y) {
    for (size_t x = 0; x < N; ++x) {
      float sum = 0.0f;
      for (size_t u = 0; u < N; ++u) {
        sum += C(u) * temp_matrix[y * N + u] *
               cosf((float)M_PI / N * (x + 0.5f) * u);
      }
      float final_val = sum * (2.0f / N);
      channel[y * N + x] = final_val;
    }
  }

  free(temp_matrix);
  free(F);
  return 0;
}

void thumbhash_init(struct thumbhash_t* restrict hash) {
  memset(hash, 0, sizeof(struct thumbhash_t));
}

uint32_t* thumbhash_allocate_data() {
  return malloc(sizeof(uint32_t) * TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM);
}

int thumbhash_encode(struct thumbhash_t* restrict hash,
                     const uint32_t* restrict data) {
  struct context_t ctx;
  if (thumbhash_encode_to_context(&ctx, data) != 0) {
    return -1;
  }
  {  // L channel
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    int count = thumbhash_DCT(ctx.lpqa[CH_L], TB_L_AC_DIM, &dc, &scale, &acs);
    if (count < 0) {
      goto DCT_err;
    }
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_DC_BITS, &value, 0);
    hash->l_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_SCALE_BITS, &value, 0);
    hash->l_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i], TB_AC_BITS, &hash->l_ac[i / 2],
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  {  // P channel
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    int count = thumbhash_DCT(ctx.lpqa[CH_P], TB_P_AC_DIM, &dc, &scale, &acs);
    if (count < 0) {
      goto DCT_err;
    }
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_DC_BITS, &value, 0);
    hash->p_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_SCALE_BITS, &value, 0);
    hash->p_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i], TB_AC_BITS, &hash->p_ac[i / 2],
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  {  // Q channel
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    int count = thumbhash_DCT(ctx.lpqa[CH_Q], TB_Q_AC_DIM, &dc, &scale, &acs);
    if (count < 0) {
      goto DCT_err;
    }
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_DC_BITS, &value, 0);
    hash->q_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_SCALE_BITS, &value, 0);
    hash->q_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i], TB_AC_BITS, &hash->q_ac[i / 2],
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  if (ctx.has_alpha) {  // A channel
    hash->has_alpha = 1;
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    int count = thumbhash_DCT(ctx.lpqa[CH_A], TB_A_AC_DIM, &dc, &scale, &acs);
    if (count < 0) {
      goto DCT_err;
    }
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_A_DC_BITS, &value, 0);
    hash->a_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_A_SCALE_BITS, &value, 0);
    hash->a_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i], TB_AC_BITS, &hash->a_ac[i / 2],
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  free(ctx.lpqa);
  return 0;
DCT_err:
  free(ctx.lpqa);
  return TB_DCT_ERROR;
}

int thumbhash_decode(const struct thumbhash_t* restrict hash,
                     uint32_t* restrict data) {
  struct context_t ctx;
  ctx.has_alpha = hash->has_alpha;
  ctx.lpqa = calloc(4, sizeof(data_channel));
  if (ctx.lpqa == NULL) {
    return -1;
  }

  // Reconstruct the LPQA channels
  float l_dc, p_dc, q_dc, a_dc;
  float l_scale, p_scale, q_scale, a_scale;
  thumbhash_bitfield_to_float(hash->l_dc, TB_DC_BITS, &l_dc, 0);
  thumbhash_bitfield_to_float(hash->p_dc, TB_DC_BITS, &p_dc, 0);
  thumbhash_bitfield_to_float(hash->q_dc, TB_DC_BITS, &q_dc, 0);
  thumbhash_bitfield_to_float(hash->l_scale, TB_SCALE_BITS, &l_scale, 0);
  thumbhash_bitfield_to_float(hash->p_scale, TB_SCALE_BITS, &p_scale, 0);
  thumbhash_bitfield_to_float(hash->q_scale, TB_SCALE_BITS, &q_scale, 0);

  // Decode luminance channel (6x6 DCT)
  if (thumbhash_IDCT(ctx.lpqa[CH_L], hash->l_ac, TB_L_AC_DIM, l_dc, l_scale) !=
      0) {
    goto IDCT_err;
  }

  // Decode P channel (5x5 DCT)
  if (thumbhash_IDCT(ctx.lpqa[CH_P], hash->p_ac, TB_P_AC_DIM, p_dc, p_scale) !=
      0) {
    goto IDCT_err;
  }

  // Decode Q channel (5x5 DCT)
  if (thumbhash_IDCT(ctx.lpqa[CH_Q], hash->q_ac, TB_Q_AC_DIM, q_dc, q_scale) !=
      0) {
    goto IDCT_err;
  }

  // Decode alpha channel if present (6x6 DCT)
  if (ctx.has_alpha) {
    thumbhash_bitfield_to_float(hash->a_dc, TB_A_DC_BITS, &a_dc, 0);
    thumbhash_bitfield_to_float(hash->a_scale, TB_A_SCALE_BITS, &a_scale, 0);

    if (thumbhash_IDCT(ctx.lpqa[CH_A], hash->a_ac, TB_A_AC_DIM, a_dc,
                       a_scale) != 0) {
      goto IDCT_err;
    }
  } else {
    // Fill with opaque alpha if no alpha channel
    for (size_t i = 0; i < TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM; i++) {
      ctx.lpqa[CH_A][i] = 1.0F;
    }
  }

  // Convert LPQA back to RGBA
  for (size_t i = 0; i < TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM; i++) {
    float l = ctx.lpqa[CH_L][i];
    float p = ctx.lpqa[CH_P][i] * 2.0F - 1.0F;
    float q = ctx.lpqa[CH_Q][i] * 2.0F - 1.0F;
    float a = ctx.lpqa[CH_A][i];

    // Convert LPQ to RGB
    float b = l - 2.0F / 3.0F * p;
    float r = (3.0F * l - b + q) / 2.0F;
    float g = r - q;

    setChannel(r, &data[i], CH_R);
    setChannel(g, &data[i], CH_G);
    setChannel(b, &data[i], CH_B);
    setChannel(a, &data[i], CH_A);
  }

  free(ctx.lpqa);
  return 0;
IDCT_err:
  free(ctx.lpqa);
  return TB_IDCT_ERROR;
}

void thumbhash_bytes(uint8_t* restrict bytes,
                     const struct thumbhash_t* restrict hash) {
  memcpy(bytes, hash, sizeof(struct thumbhash_t));
}
