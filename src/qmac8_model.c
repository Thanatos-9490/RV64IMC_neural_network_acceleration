/* ---------------------------------------------------------------------------
 * qmac8_model.c - software model of the qmac8 / qreq instructions (host x86).
 * ------------------------------------------------------------------------- */

#include "nnaccel.h"
#include "tfl_requant.h"

int32_t qmac8(int32_t acc, uint64_t packed_a, uint64_t packed_w)
{
    for (int lane = 0; lane < 8; lane++) {
        int8_t a = (int8_t)((packed_a >> (8 * lane)) & 0xFF);
        int8_t w = (int8_t)((packed_w >> (8 * lane)) & 0xFF);
        acc += (int32_t)a * (int32_t)w;
    }
    return acc;
}

int8_t qreq(int32_t acc, uint64_t p)
{
    return tfl_requant_i8(acc,
                          (int32_t)(p & 0xFFFFFFFFull),
                          (int8_t)((p >> 32) & 0xFF),
                          (int8_t)((p >> 40) & 0xFF),
                          (int8_t)((p >> 48) & 0xFF),
                          (int8_t)((p >> 56) & 0xFF));
}
