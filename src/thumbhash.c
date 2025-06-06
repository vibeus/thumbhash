// Copyright (c) 2025, Vibe Inc

#include "thumbhash/thumbhash.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef float data_channel[TB_SIZE_DATA_DIM];

static float getChannel(uint32_t pixel, uint8_t channel) {
  return ((pixel >> (channel * 8)) & 0xFF) / 255.0F;
}

static void setChannel(float value, uint32_t* pixel, uint8_t channel) {
  uint8_t clamped = fmin(fmax(0.0F, value), 1.0F) * 255.0F;
  *pixel |= (clamped << (channel * 8));
}

struct context_t {
  _Bool has_alpha;
  data_channel* lpqa;
};

static void thumbhash_float_to_bitfield(float in, size_t bits, uint8_t* out,
                                        size_t shift) {
  assert(shift + bits <= 8);
  *out = *out | lroundf(in * (float)((1 << bits) - 1)) << shift;
}

static void thumbhash_bitfield_to_float(uint8_t in, size_t bits, float* out,
                                        size_t shift) {
  assert(shift + bits <= 8);
  uint8_t mask = ((1 << bits) - 1);
  uint8_t value = (in >> shift) & mask;
  *out = (float)value / mask;
}

static void thumbhash_encode_to_context(struct context_t* restrict ctx,
                                        const uint32_t* restrict data) {
  float avg[4] = {0.0F};
  const size_t data_size = TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM;
  const size_t R = 0, G = 1, B = 2, A = 3;
  const size_t L = 0, P = 1, Q = 2;
  const float rdata_size = 1.0 / data_size;

  for (size_t i = 0; i < data_size; ++i) {
    float r = getChannel(data[i], 0);
    float g = getChannel(data[i], 1);
    float b = getChannel(data[i], 2);
    float a = getChannel(data[i], 3);
    avg[R] += r * a;
    avg[G] += g * a;
    avg[B] += b * a;
    avg[A] += a;
  }

  if (avg[A] > 0.0F) {
    avg[R] /= avg[A];
    avg[G] /= avg[A];
    avg[B] /= avg[A];
  }
  avg[R] *= rdata_size;
  avg[G] *= rdata_size;
  avg[B] *= rdata_size;
  avg[A] *= rdata_size;

  ctx->has_alpha = avg[A] < 1.0F;

  ctx->lpqa = calloc(4, sizeof(data_channel));
  for (size_t i = 0; i < data_size; ++i) {
    float r = getChannel(data[i], 0);
    float g = getChannel(data[i], 1);
    float b = getChannel(data[i], 2);
    float a = getChannel(data[i], 3);
    // do a ATOP mix over avg
    float mix_r = avg[R] * (1 - a) + r * a;
    float mix_g = avg[G] * (1 - a) + g * a;
    float mix_b = avg[B] * (1 - a) + b * a;
    ctx->lpqa[L][i] = (mix_r + mix_g + mix_b) / 3.0f;
    ctx->lpqa[P][i] = (mix_r + mix_g) * 0.5 - mix_b;
    ctx->lpqa[Q][i] = (mix_r - mix_g);
    ctx->lpqa[A][i] = a;
  }
}

// allocate memory for AC coefficients and returns the size of the allocated
// memory
static size_t thumbhash_DCT(data_channel channel, size_t dim, float* dc,
                            float* scale, float** acs) {
  float fx[TB_SIZE_DATA_DIM] = {0.0F};
  *acs = calloc(dim * dim, sizeof(float));
  float* aacs = *acs;
  size_t count = 0;
  for (size_t y = 0; y < dim; ++y) {
    for (size_t x = 0; x < (dim - y); ++x) {
      float f = 0.0F;
      for (size_t i = 0; i < TB_SIZE_DATA_DIM; ++i) {
        fx[i] = cosf(M_PI / TB_SIZE_DATA_DIM * x * ((float)i + 0.5));
      }
      for (size_t j = 0; j < TB_SIZE_DATA_DIM; ++j) {
        float fy = cosf(M_PI / TB_SIZE_DATA_DIM * y * ((float)j + 0.5));
        for (size_t i = 0; i < TB_SIZE_DATA_DIM; ++i) {
          f += channel[i + j * TB_SIZE_DATA_DIM] * fx[i] * fy;
        }
      }
      f /= (TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM);
      if (x != 0 || y != 0) {
        aacs[count++] = f;
        *scale = fmax(*scale, fabs(f));
      } else {
        *dc = f;
      }
    }
  }

  // Scale AC coefficients to [0, 1] for bitfield encoding
  if (*scale > 0) {
    for (size_t i = 0; i < count; ++i) {
      aacs[i] = 0.5F + 0.5F / *scale * aacs[i];
    }
  }
  return count;
}

static void thumbhash_IDCT(data_channel channel,
                           const uint8_t* restrict encoded_acs, size_t dim,
                           float dc, float scale) {
  float* coeffs = calloc(dim * dim, sizeof(float));
  size_t count = 0;

  // Set DC coefficient
  coeffs[0] = dc;

  // Decode AC coefficients
  for (size_t y = 0; y < dim; ++y) {
    for (size_t x = 0; x < (dim - y); ++x) {
      if (x != 0 || y != 0) {
        float ac;
        thumbhash_bitfield_to_float(encoded_acs[count / 2], TB_AC_BITS, &ac,
                                    (count % 2) * TB_AC_BITS);
        ac = ac * 2.0F - 1.0F;  // Scale from [0,1] to [-1,1]
        coeffs[y * dim + x] = scale * ac;
        count++;
      }
    }
  }

  // Reconstruct channel data using inverse DCT
  for (size_t y = 0; y < TB_SIZE_DATA_DIM; ++y) {
    for (size_t x = 0; x < TB_SIZE_DATA_DIM; ++x) {
      float value = 0.0F;
      for (size_t j = 0; j < dim; ++j) {
        for (size_t i = 0; i < dim; ++i) {
          float basis = cosf(M_PI / TB_SIZE_DATA_DIM * i * (x + 0.5F)) *
                        cosf(M_PI / TB_SIZE_DATA_DIM * j * (y + 0.5F));
          value += coeffs[j * dim + i] * basis;
        }
      }
      channel[y * TB_SIZE_DATA_DIM + x] = value;
    }
  }

  free(coeffs);
}

void thumbhash_init(struct thumbhash_t* restrict hash) {
  memset(hash, 0, sizeof(struct thumbhash_t));
}

uint32_t* thumbhash_allocate_data() {
  return malloc(sizeof(uint32_t) * TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM);
}

void thumbhash_encode(struct thumbhash_t* restrict hash,
                      const uint32_t* restrict data) {
  struct context_t ctx;
  thumbhash_encode_to_context(&ctx, data);
  {  // L channel
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    size_t count = thumbhash_DCT(ctx.lpqa[0], TB_L_AC_DIM, &dc, &scale, &acs);
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_DC_BITS, &value, 0);
    hash->l_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_SCALE_BITS, &value, 0);
    hash->l_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->l_ac,
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  if (ctx.has_alpha) {  // A channel
    hash->has_alpha = 1;
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    size_t count = thumbhash_DCT(ctx.lpqa[3], TB_A_AC_DIM, &dc, &scale, &acs);
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_A_DC_BITS, &value, 0);
    hash->a_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_A_SCALE_BITS, &value, 0);
    hash->a_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->a_ac,
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  {  // P channel
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    size_t count = thumbhash_DCT(ctx.lpqa[1], TB_P_AC_DIM, &dc, &scale, &acs);
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_DC_BITS, &value, 0);
    hash->p_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_SCALE_BITS, &value, 0);
    hash->p_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->p_ac,
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  {  // Q channel
    float dc = 0.0F;
    float scale = 0.0F;
    float* acs;
    size_t count = thumbhash_DCT(ctx.lpqa[2], TB_Q_AC_DIM, &dc, &scale, &acs);
    uint8_t value = 0;
    thumbhash_float_to_bitfield(dc, TB_DC_BITS, &value, 0);
    hash->q_dc = value;
    value = 0;
    thumbhash_float_to_bitfield(scale, TB_SCALE_BITS, &value, 0);
    hash->q_scale = value;

    for (size_t i = 0; i < count; i++) {
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->q_ac,
                                  (i % 2) * TB_AC_BITS);
    }
    free(acs);
  }
  free(ctx.lpqa);
}

void thumbhash_decode(const struct thumbhash_t* restrict hash,
                      uint32_t* restrict data) {
  struct context_t ctx;
  ctx.has_alpha = hash->has_alpha;
  ctx.lpqa = calloc(4, sizeof(data_channel));

  // Reconstruct the LPQA channels
  float l_dc, p_dc, q_dc, a_dc;
  float l_scale, p_scale, q_scale, a_scale;
  thumbhash_bitfield_to_float(hash->l_dc, TB_DC_BITS, &l_dc, 0);
  thumbhash_bitfield_to_float(hash->p_dc, TB_DC_BITS, &p_dc, 0);
  thumbhash_bitfield_to_float(hash->q_dc, TB_DC_BITS, &q_dc, 0);
  p_dc = p_dc * 2.0F - 1.0F;  // scale back to [-1, 1]
  q_dc = q_dc * 2.0F - 1.0F;  // scale back to [-1, 1]
  thumbhash_bitfield_to_float(hash->l_scale, TB_SCALE_BITS, &l_scale, 0);
  thumbhash_bitfield_to_float(hash->p_scale, TB_SCALE_BITS, &p_scale, 0);
  thumbhash_bitfield_to_float(hash->q_scale, TB_SCALE_BITS, &q_scale, 0);

  // Decode luminance channel (6x6 DCT)
  thumbhash_IDCT(ctx.lpqa[0], hash->l_ac, TB_L_AC_DIM, l_dc, l_scale);

  // Decode P channel (5x5 DCT)
  thumbhash_IDCT(ctx.lpqa[1], hash->p_ac, TB_P_AC_DIM, p_dc, p_scale);

  // Decode Q channel (5x5 DCT)
  thumbhash_IDCT(ctx.lpqa[2], hash->q_ac, TB_Q_AC_DIM, q_dc, q_scale);

  // Decode alpha channel if present (6x6 DCT)
  if (ctx.has_alpha) {
    thumbhash_bitfield_to_float(hash->a_dc, 4, &a_dc, 0);
    thumbhash_bitfield_to_float(hash->a_scale, 4, &a_scale, 0);
    thumbhash_IDCT(ctx.lpqa[3], hash->a_ac, TB_A_AC_DIM, a_dc, a_scale);
  } else {
    // Fill with opaque alpha if no alpha channel
    for (size_t i = 0; i < TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM; i++) {
      ctx.lpqa[3][i] = 1.0F;
    }
  }

  // Convert LPQA back to RGBA
  for (size_t i = 0; i < TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM; i++) {
    float l = ctx.lpqa[0][i];
    float p = ctx.lpqa[1][i];
    float q = ctx.lpqa[2][i];
    float a = ctx.lpqa[3][i];

    // Convert LPQ to RGB
    float b = l - 2.0F / 3.0F * p;
    float r = (3.0F * l - b + q) / 2.0F;
    float g = r - q;

    setChannel(r, &data[i], 0);
    setChannel(g, &data[i], 1);
    setChannel(b, &data[i], 2);
    setChannel(a, &data[i], 3);
  }

  free(ctx.lpqa);
}

void thumbhash_bytes(uint8_t* restrict bytes,
                     const struct thumbhash_t* restrict hash) {
  memcpy(bytes, hash, sizeof(struct thumbhash_t));
}
