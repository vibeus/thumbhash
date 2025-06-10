// Copyright (c) 2025, Vibe Inc

#pragma once

#ifdef __cplusplus
#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif
extern "C" {
#else
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */
#define RESTRICT restrict
#endif
#endif  // __cplusplus

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#define TB_DATA_DIM 128
#define TB_DC_BITS 6
#define TB_SCALE_BITS 6
#define TB_AC_BITS 4
#define TB_L_AC_DIM 6
#define TB_P_AC_DIM 5
#define TB_Q_AC_DIM 5
#define TB_A_AC_DIM 6
/**
 * ThumbHash+ is a image placeholder algorithm that implements a modified
 * version of the ThumbHash algorithm with:
 * - Fixed-size input (TB_DATA_DIM x TB_DATA_DIM) (128x128) pixels
 * - Fixed-size output (40 bytes when alpha present, 30 bytes otherwise)
 * - Luminance and alpha channels encoded with fixed terms
 * - Optimized storage using bit packing
 *
 * The format uses DCT compression with:
 * - 6-bit DC coefficients
 * - 6-bit scale factors
 * - 4-bit AC coefficients
 *
 * When alpha channel is not present, last 10 bytes can be omitted
 */
#pragma pack(push, 1)
struct thumbhashp_t {
  // DC part
  uint8_t l_dc : TB_DC_BITS;
  uint8_t p_dc : TB_DC_BITS;
  uint8_t q_dc : TB_DC_BITS;
  uint8_t l_scale : TB_SCALE_BITS;
  uint8_t p_scale : TB_SCALE_BITS;
  uint8_t q_scale : TB_SCALE_BITS;
  uint8_t a_dc : TB_DC_BITS;
  uint8_t a_scale : TB_SCALE_BITS;

  // AC part
  uint8_t l_ac[TB_AC_BITS * (5 + 5 + 4 + 3 + 2 + 1) / 8];  // 10 bytes
  uint8_t p_ac[TB_AC_BITS * (4 + 4 + 3 + 2 + 1) / 8];      // 7 bytes
  uint8_t q_ac[TB_AC_BITS * (4 + 4 + 3 + 2 + 1) / 8];      // 7 bytes
  uint8_t a_ac[TB_AC_BITS * (5 + 5 + 4 + 3 + 2 + 1) / 8];  // 10 bytes
};
#pragma pack(pop)

// assume pixel format is RGBA8888 non-premultiplied (same as PNG standard) in
// linear (preferably) or sRGB (acceptably) space.
/**
 * @brief Initialize thumbhashp_t struct to zero
 * @param hash Pointer to thumbhashp_t structure to initialize
 */
void thumbhashp_init(struct thumbhashp_t* RESTRICT hash);

/**
 * @brief Encode RGBA image into thumbhashp_t structure
 * @param hash Initialized thumbhashp_t structure to store result
 * @param data Input RGBA image data (32-bit pixels)
 * @return 0 on success, non-zero on error
 */
int thumbhashp_encode(struct thumbhashp_t* RESTRICT hash,
                      const uint32_t* RESTRICT data);

/**
 * @brief Decode thumbhashp into RGBA image
 * @param hash thumbhashp to decode
 * @param data Output buffer for decoded image (must be allocated)
 * @return 0 on success, non-zero on error
 */
int thumbhashp_decode(const struct thumbhashp_t* RESTRICT hash,
                      uint32_t* RESTRICT data);

/**
 * @brief Check if thumbhashp contains alpha channel
 * @param hash thumbhashp to check
 * @return true if alpha channel is present, false otherwise
 */
bool thumbhashp_has_alpha(const struct thumbhashp_t* RESTRICT hash);

/**
 * @brief Convert thumbhashp to byte array
 * @param hash thumbhashp to convert
 * @return Pointer to allocated byte array (must be freed by caller)
 */
void* thumbhashp_to_bytes(const struct thumbhashp_t* RESTRICT hash);

/**
 * @brief Get length of byte array for thumbhashp
 * @param hash thumbhashp to measure
 * @return Size in bytes needed to store the thumbhashp
 */
size_t thumbhashp_bytes_len(const struct thumbhashp_t* hash);

/**
 * @brief Load thumbhashp from byte array
 * @param hash Output thumbhashp structure
 * @param bytes Input byte array containing thumbhashp data
 */
void thumbhashp_from_bytes(struct thumbhashp_t* RESTRICT hash,
                           const void* RESTRICT bytes);
#ifdef __cplusplus
}
#endif
