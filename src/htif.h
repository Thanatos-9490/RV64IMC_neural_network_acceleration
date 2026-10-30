/* ---------------------------------------------------------------------------
 * htif.h - minimal console output + instret for the bare-metal Spike programs.
 * ------------------------------------------------------------------------- */
#ifndef HTIF_H
#define HTIF_H
#include <stdint.h>

extern volatile uint64_t tohost, fromhost;

static inline uint64_t rd_instret(void)
{
    uint64_t x;
    __asm__ volatile ("csrr %0, instret" : "=r"(x));
    return x;
}

static inline void htif_write(const char *s, int n)
{
    static volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
    magic_mem[0] = 64;                        /* SYS_write */
    magic_mem[1] = 1;                         /* fd = stdout */
    magic_mem[2] = (uint64_t)s;
    magic_mem[3] = (uint64_t)n;
    __sync_synchronize();
    tohost = (uint64_t)magic_mem;
    while (fromhost == 0)
        ;
    fromhost = 0;
}

/* tiny formatter: "<tag> v0 v1 ...\n" with unsigned decimal values */
static inline void htif_line(const char *tag, const uint64_t *v, int nv)
{
    char buf[200];
    int n = 0;
    while (*tag)
        buf[n++] = *tag++;
    for (int i = 0; i < nv; i++) {
        char t[24];
        int l = 0;
        uint64_t x = v[i];
        buf[n++] = ' ';
        if (!x)
            t[l++] = '0';
        while (x) {
            t[l++] = (char)('0' + x % 10);
            x /= 10;
        }
        while (l)
            buf[n++] = t[--l];
    }
    buf[n++] = '\n';
    htif_write(buf, n);
}
#endif
