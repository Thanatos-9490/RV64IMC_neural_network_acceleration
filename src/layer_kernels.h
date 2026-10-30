/* ---------------------------------------------------------------------------
 * layer_kernels.h - quantized GEMM-form layer kernels (conv via im2col, FC)
 * ------------------------------------------------------------------------- */
#ifndef LAYER_KERNELS_H
#define LAYER_KERNELS_H

#include <stdint.h>
#include "layer_t.h"

/* X1T = X1 with cache-blocked loop order (same instructions per MAC, same bytes) */
/* X2  = X1  + ldpi (post-increment load);  X2T = X1T + ldpi */
enum { CFG_B0 = 0, CFG_B1 = 1, CFG_X0 = 2, CFG_X1 = 3, CFG_X1T = 4,
       CFG_X2 = 5, CFG_X2T = 6 };

/* out is [M][N] int8 */
void run_layer(const layer_t *L, int cfg, int8_t *out);

#endif
