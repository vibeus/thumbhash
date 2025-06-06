# thumbhash
A C/C++ implementation of ThumbHash image placeholder algorithm

https://evanw.github.io/thumbhash/

Note that the hash struct is modified from the original design:
- input size should be 128x128, instead of any size below 100x100
- hash size is fixed to 40 bytes
  - use DCT size 6 for luminance and alpha channels
  - use DCT size 4 for color channels