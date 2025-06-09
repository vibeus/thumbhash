// Copyright (c) 2025, Vibe Inc

#pragma once

#ifdef __cplusplus
#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
extern "C" {
#endif
#else
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */
#define RESTRICT restrict
#endif
#endif

#include <stddef.h>
#include <stdint.h>
#define TB_SIZE_DATA_DIM 128
#define TB_DC_BITS 6
#define TB_SCALE_BITS 6
#define TB_A_DC_BITS 4
#define TB_A_SCALE_BITS 4
#define TB_AC_BITS 4
#define TB_L_AC_DIM 6
#define TB_P_AC_DIM 5
#define TB_Q_AC_DIM 5
#define TB_A_AC_DIM 6
// we use a different approach to original algorithm
// luminance and alpha channel are encoded with fixed terms
// this will generate a fixed sized hash will a small extra space
// total size of thumbhash_t struct is 40 bytes
#pragma pack(push, 1)
struct thumbhash_t {
  // dc part and flag 6 bytes
  uint8_t l_dc : TB_DC_BITS;
  uint8_t p_dc : TB_DC_BITS;
  uint8_t q_dc : TB_DC_BITS;
  uint8_t l_scale : TB_SCALE_BITS;
  uint8_t p_scale : TB_SCALE_BITS;
  uint8_t q_scale : TB_SCALE_BITS;
  uint8_t a_dc : TB_A_DC_BITS;
  uint8_t a_scale : TB_A_SCALE_BITS;
  uint8_t has_alpha : 1;
  uint8_t reserved : 3;

  uint8_t l_ac[TB_AC_BITS * (5 + 5 + 4 + 3 + 2 + 1) / 8];  // 10 bytes
  uint8_t p_ac[TB_AC_BITS * (4 + 4 + 3 + 2 + 1) / 8];      // 7 bytes
  uint8_t q_ac[TB_AC_BITS * (4 + 4 + 3 + 2 + 1) / 8];      // 7 bytes
  uint8_t a_ac[TB_AC_BITS * (5 + 5 + 4 + 3 + 2 + 1) / 8];  // 10 bytes
};
#pragma pack(pop)

// assume pixel format is 4 bytes per pixel RGBA unpremultiplied in linear space

// Initialize thumbhash_t struct to zero
void thumbhash_init(struct thumbhash_t* RESTRICT hash);
// Encode a RGBA image into a thumbhash_t structure. The function takes a
// pointer to the input image data and write to the output hash structure.
// hash should be initialized before calling this function.
void thumbhash_encode(struct thumbhash_t* RESTRICT hash,
                      const uint32_t* RESTRICT data);
void thumbhash_decode(const struct thumbhash_t* RESTRICT hash,
                      uint32_t* RESTRICT data);

// convert a thumbhash_t structure to a byte array. The function takes a pointer
// to the output byte
// the bytes should be of size sizeof(thumbhash_t)
void thumbhash_bytes(uint8_t* RESTRICT bytes,
                     const struct thumbhash_t* RESTRICT hash);
#ifdef __cplusplus
}
#endif
