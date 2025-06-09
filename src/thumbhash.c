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

// allocate memory for AC coefficients and returns the size of the allocated
// memory
static int thumbhash_DCT(data_channel channel, size_t dim, float* restrict dc,
                         float* restrict scale, float** restrict acs) {
  float* fx = calloc(TB_SIZE_DATA_DIM, sizeof(float));
  if (fx == NULL) {
    return -1;
  }

  *acs = calloc(dim * dim, sizeof(float));
  if (*acs == NULL) {
    free(fx);
    return -1;
  }
  float* aacs = *acs;
  int count = 0;
  for (size_t y = 0; y < dim; ++y) {
    for (size_t x = 0; x < (dim - y); ++x) {
      float f = 0.0F;
      for (size_t i = 0; i < TB_SIZE_DATA_DIM; ++i) {
        fx[i] = cosf((float)M_PI / TB_SIZE_DATA_DIM * x * ((float)i + 0.5F));
      }
      for (size_t j = 0; j < TB_SIZE_DATA_DIM; ++j) {
        float fy = cosf((float)M_PI / TB_SIZE_DATA_DIM * y * ((float)j + 0.5F));
        for (size_t i = 0; i < TB_SIZE_DATA_DIM; ++i) {
          f += channel[i + (j * TB_SIZE_DATA_DIM)] * fx[i] * fy;
        }
      }
      f /= (TB_SIZE_DATA_DIM * TB_SIZE_DATA_DIM);
      if (x != 0 || y != 0) {
        aacs[count++] = f;
        *scale = fmaxf(*scale, fabsf(f));
      } else {
        *dc = 0.5F + 0.5F * f;  // scale to [0, 1]
      }
    }
  }
  free(fx);

  if (*scale > 0) {
    for (size_t i = 0; i < count; ++i) {
      // scale to [0, 1]
      aacs[i] = 0.5F + 0.5F * aacs[i] / *scale;
    }
  }
  return count;
}

static int thumbhash_IDCT(data_channel channel,
                          const uint8_t* restrict encoded_acs, size_t dim,
                          float dc, float scale, int normalize) {
  float* coeffs = calloc(dim * dim, sizeof(float));
  if (coeffs == NULL) {
    return -1;
  }
  size_t count = 0;

  // Set DC coefficient
  coeffs[0] = dc * 2.0F - 1.0F;

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
      float* fx = calloc(dim, sizeof(float));
      if (fx == NULL) {
        free(coeffs);
        return -1;
      }
      float* fy = calloc(dim, sizeof(float));
      if (fx == NULL) {
        free(fx);
        free(coeffs);
        return -1;
      }
      for (size_t i = 0; i < dim; ++i) {
        fx[i] = cosf((float)M_PI / TB_SIZE_DATA_DIM * (x + 0.5F) * i);
      }
      for (size_t j = 0; j < dim; ++j) {
        fy[j] = cosf((float)M_PI / TB_SIZE_DATA_DIM * (y + 0.5F) * j);
      }
      float value = 0.0F;
      for (size_t j = 0; j < dim; ++j) {
        for (size_t i = 0; i < (dim - j); ++i) {
          if (j == 0 && i == 0) {
            value = coeffs[0];
          } else {
            value += coeffs[j * dim + i] * fx[i] * fy[j];
          }
        }
      }
      if (normalize) {
        channel[y * TB_SIZE_DATA_DIM + x] = value * 0.5F + 0.5F;
      } else {
        channel[y * TB_SIZE_DATA_DIM + x] = value;
      }
      free(fy);
      free(fx);
    }
  }

  free(coeffs);
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
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->l_ac,
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
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->p_ac,
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
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->q_ac,
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
      thumbhash_float_to_bitfield(acs[i / 2], TB_AC_BITS, hash->a_ac,
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
  if (thumbhash_IDCT(ctx.lpqa[CH_L], hash->l_ac, TB_L_AC_DIM, l_dc, l_scale,
                     1) != 0) {
    goto IDCT_err;
  }

  // Decode P channel (5x5 DCT)
  if (thumbhash_IDCT(ctx.lpqa[CH_P], hash->p_ac, TB_P_AC_DIM, p_dc, p_scale,
                     0) != 0) {
    goto IDCT_err;
  }

  // Decode Q channel (5x5 DCT)
  if (thumbhash_IDCT(ctx.lpqa[CH_Q], hash->q_ac, TB_Q_AC_DIM, q_dc, q_scale,
                     0) != 0) {
    goto IDCT_err;
  }

  // Decode alpha channel if present (6x6 DCT)
  if (ctx.has_alpha) {
    thumbhash_bitfield_to_float(hash->a_dc, TB_A_DC_BITS, &a_dc, 0);
    thumbhash_bitfield_to_float(hash->a_scale, TB_A_SCALE_BITS, &a_scale, 0);

    if (thumbhash_IDCT(ctx.lpqa[CH_A], hash->a_ac, TB_A_AC_DIM, a_dc, a_scale,
                       1) != 0) {
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
    float p = ctx.lpqa[CH_P][i];
    float q = ctx.lpqa[CH_Q][i];
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
