# ---------------------------------------------------------------------------
# Bare-metal startup for RV64IMC on Spike.
#
# HTIF convention (fesvr):
#     tohost = 1              -> PASS (simulator exits with code 0)
#     tohost = (code<<1) | 1  -> FAIL (simulator exits with code `code`)
# ---------------------------------------------------------------------------

    .section .text.init
    .globl _start
_start:
    la   sp, _stack_top
    call main

    # main() returns 0 on success, non-zero on failure.
    slli a1, a0, 1
    ori  a1, a1, 1            # a1 = (return_code << 1) | 1
    la   t0, tohost
    sd   a1, 0(t0)
1:
    j    1b

    .section .tohost,"aw",@progbits
    .align 6
    .globl tohost
tohost:
    .dword 0
    .align 6
    .globl fromhost
fromhost:
    .dword 0