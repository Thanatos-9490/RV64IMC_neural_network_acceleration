OUTPUT_ARCH( "riscv" )
ENTRY( _start )

SECTIONS
{
  /* Spike loads the program at 0x80000000. */
  . = 0x80000000;

  .text : {
    *(.text.init)     /* _start must be first */
    *(.text .text.*)
  }

  .rodata : { *(.rodata .rodata.*) }
  .data   : { *(.data .data.*) }
  .bss    : { *(.bss .bss.*) *(COMMON) }

  /* HTIF tohost/fromhost region. */
  .tohost : ALIGN(8) { *(.tohost) }

  /* Place the stack at the end of the first 1 MB of memory. */
  _stack_top = . + 0x100000;
}