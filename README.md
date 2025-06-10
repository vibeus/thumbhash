# ThumbHash+

A C/C++ implementation of modified ThumbHash image placeholder algorithm

Original: https://evanw.github.io/thumbhash/

Modified:
 - Fixed-size input (TB_DATA_DIM x TB_DATA_DIM) (128x128) pixels
 - Fixed-size output (40 bytes when alpha present, 30 bytes otherwise)
 - Luminance and alpha channels encoded with fixed terms
 - Optimized storage using bit packing
 
The format uses DCT compression with:
 - 6-bit DC coefficients
 - 6-bit scale factors
 - 4-bit AC coefficients
 
When alpha channel is not present, last 10 bytes can be omitted
