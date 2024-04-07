/* inffast.c -- fast decoding
 * Copyright (C) 1995-2017 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zutil.h"
#include "inftrees.h"
#include "inflate.h"
#include "inffast.h"

#ifndef CHUNKCOPY_CHUNK_SIZE
#define CHUNKCOPY_CHUNK_SIZE 32
#endif

#ifndef ASMINF

#define ASSUME_NOHARDSTOP 3
#define ASSUME_NOOVERLAP 4
#define ASSUME_INPUT_APLENTY 8
#define ASSUME_OUTPUT_APLENTY 16
#define CAN_ASSUME(shortcut) ((~assumptions & ASSUME_##shortcut) == 0)
#define CANNOT_ASSUME(shortcut) ((~assumptions & ASSUME_##shortcut) != 0)

#define INLINE_NEVER __attribute__((noinline))
#define INLINE_ALWAYS inline __attribute__((always_inline))
#if 0
#define INLINE_UNLESS_DEBUG INLINE_NEVER
#else
#define INLINE_UNLESS_DEBUG INLINE_ALWAYS
#endif

static INLINE_ALWAYS int unwind_chunk(uint8_t* out, int d) {
    int d0 = d;
    int avail = d;
    *out = *(out - d);
    ++out;
    ++avail;
    if (d <= 1) d += d;
    for (int seg = 1; seg < CHUNKCOPY_CHUNK_SIZE; seg <<= 1) {
        zmemcpy(out, out - d, seg);
        out += seg;
        d += d;
        avail += seg;
        d -= (d <= avail) ? d0 : 0;
    }
    return d;
}

static INLINE_ALWAYS void omnicopy258(uint8_t* dptr, uint8_t const* sptr, int len, const int assumptions) {
    if (CANNOT_ASSUME(NOHARDSTOP) && len < CHUNKCOPY_CHUNK_SIZE) {
        for (int i = 0; i < len; ++i) *dptr++ = *sptr++;
        return;
    }
    size_t offset = dptr - sptr;
    size_t vl = (len - 1) % CHUNKCOPY_CHUNK_SIZE + 1;
    if (CANNOT_ASSUME(NOOVERLAP) && offset < CHUNKCOPY_CHUNK_SIZE) {
#if 1
        int off0 = offset;
        size_t avail = offset;
        *dptr = *(dptr - offset);
        ++dptr;
        ++avail;
        if (offset <= 1) offset += offset;
        for (int seg = 1; seg < CHUNKCOPY_CHUNK_SIZE; seg <<= 1) {
            zmemcpy(dptr, dptr - offset, seg);
            dptr += seg;
            offset += offset;
            avail += seg;
            offset -= (offset <= avail) ? off0 : 0;
        }
#else
        offset = unwind_chunk(dptr, offset);
#endif
        sptr = dptr - offset;
    } else {
        zmemcpy(dptr, sptr, CHUNKCOPY_CHUNK_SIZE);
    }
    dptr += vl;
    sptr += vl;
    len -= vl;
    while (len > 0) {
        zmemcpy(dptr, sptr, CHUNKCOPY_CHUNK_SIZE);
        dptr += CHUNKCOPY_CHUNK_SIZE;
        sptr += CHUNKCOPY_CHUNK_SIZE;
        len -= CHUNKCOPY_CHUNK_SIZE;
    }
}


#if defined __riscv
#include <riscv_vector.h>

// TODO: reconcile these two functions; one uses m2, one uses m1.  The overlap
// case can probably be trivially expanded to match the non-overlap case, but
// the non-overlap case needs to be evaluated to choose the proper LMUL.
static INLINE_UNLESS_DEBUG void rvv_copy258_nooverlap(uint8_t* dptr, uint8_t const* sptr, int len) {
    size_t vl = __riscv_vsetvl_e8m2(len);
    vuint8m2_t chunk = __riscv_vle8_v_u8m2(sptr, vl);
    __riscv_vse8_v_u8m2(dptr, chunk, vl);

    while (__builtin_expect(len > vl, 0)) {
        len -= vl;
        dptr += vl;
        sptr += vl;
        vl = __riscv_vsetvl_e8m2(len);
        chunk = __riscv_vle8_v_u8m2(sptr, vl);
        __riscv_vse8_v_u8m2(dptr, chunk, vl);
    }
}

static INLINE_UNLESS_DEBUG void rvv_copy258_overlap(uint8_t* dptr, int offset, int len) {
    uint8_t const* sptr = dptr - offset;
    size_t vl = __riscv_vsetvl_e8m1(len);
    vuint8m1_t chunk = __riscv_vle8_v_u8m1(sptr, vl);
    if (offset < vl) {  // TODO: should it be unconditional?
        vuint8m1_t idx = __riscv_vid_v_u8m1(vl);
        idx = __riscv_vremu(idx, offset, vl);
        chunk = __riscv_vrgather(chunk, idx, vl);
        int unroll = vl - vl % offset;
        // if offset >= vl, unroll becomes 0, so no-op below:
        offset += unroll;
        sptr -= unroll;
    }
    __riscv_vse8_v_u8m1(dptr, chunk, vl);
    while (__builtin_expect(len > vl, 0)) {
        len -= vl;
        dptr += vl;
        sptr += vl;
        vl = __riscv_vsetvl_e8m1(len);
        chunk = __riscv_vle8_v_u8m1(sptr, vl);
        __riscv_vse8_v_u8m1(dptr, chunk, vl);
    }
}

#define COPY258_NOOVERLAP_HARDSTOP(d, s, l) rvv_copy258_nooverlap(d, s, l)
#define COPY258_NOOVERLAP_NOHARDSTOP(d, s, l) rvv_copy258_nooverlap(d, s, l)
#define COPY258_OVERLAP_HARDSTOP(d, o, l) rvv_copy258_overlap(d, o, l)
#define COPY258_OVERLAP_NOHARDSTOP(d, o, l) rvv_copy258_overlap(d, o, l)
#else

static INLINE_UNLESS_DEBUG void copy258_nooverlap_nohardstop(uint8_t* dptr, uint8_t const* sptr, int len) {
    omnicopy258(dptr, sptr, len, ASSUME_NOOVERLAP | ASSUME_NOHARDSTOP);
}

static INLINE_UNLESS_DEBUG void copy258_nooverlap_hardstop(uint8_t* dptr, uint8_t const* sptr, int len) {
    omnicopy258(dptr, sptr, len, ASSUME_NOOVERLAP);
}

static INLINE_UNLESS_DEBUG void copy258_overlap_nohardstop(uint8_t* dptr, int offset, int len) {
    omnicopy258(dptr, dptr - offset, len, ASSUME_NOHARDSTOP);
}

static INLINE_UNLESS_DEBUG void copy258_overlap_hardstop(uint8_t* dptr, int offset, int len) {
    omnicopy258(dptr, dptr - offset, len, 0);
}

#define COPY258_NOOVERLAP_HARDSTOP(d, s, l) copy258_nooverlap_hardstop(d, s, l)
#define COPY258_NOOVERLAP_NOHARDSTOP(d, s, l) copy258_nooverlap_nohardstop(d, s, l)
#define COPY258_OVERLAP_HARDSTOP(d, o, l) copy258_overlap_hardstop(d, o, l)
#define COPY258_OVERLAP_NOHARDSTOP(d, o, l) copy258_overlap_nohardstop(d, o, l)
#endif

static INLINE_UNLESS_DEBUG int windowcopy(uint8_t* outptr, uint8_t const* window, int wsize, int whave, int wnext, int offset, int copy) {
    int problem = 0;
    if (__builtin_expect(copy > whave, 0)) {
        problem = 1;
#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
        Trace((stderr, "inflate.c too far\n"));
        int segment = copy - whave;
        zmemzero(outptr, segment);
        outptr += segment;
        offset -= segment;
        copy -= segment;
#else
        return problem;
#endif
    }
    /* if caller consumed > 32k last time then wnext will be zero, so expect
     * false.  Otherwise it's probably even odds.
     */
    if (offset > wnext) {
        int top_offset = offset - wnext;
        int segment = copy;
        if (segment > top_offset) segment = top_offset;
        COPY258_NOOVERLAP_HARDSTOP(outptr, window + wsize - top_offset, segment);
        outptr += segment;
        offset -= segment;
        copy -= segment;
        if (__builtin_expect(copy == 0, 1)) return problem;
    }
    COPY258_NOOVERLAP_HARDSTOP(outptr, window + wnext - offset, copy);

    return problem;
}


/* Clear the input bit accumulator */
#define REPLACEBITS() \
    do { \
        int replace_size = bits >> 3; \
        bits -= replace_size * 8; \
        hold &= ~(~0ull << bits); \
        inptr -= replace_size; \
        avail_in += replace_size; \
    } while (0)
/* Restore state from registers in inflate() */
#define RESTORE() \
    do { \
        /* assert(bits <= 8 * sizeof(state->hold)) */ \
        strm->next_out = outptr; \
        strm->avail_out = avail_out; \
        strm->next_in = inptr; \
        strm->avail_in = avail_in; \
        state->hold = hold; \
        state->bits = bits; \
    } while (0)
/* Check that required number of bits are available, and bail-out otherwise. */
#define NEEDBITS(n, use) \
    do { \
        if (CANNOT_ASSUME(INPUT_APLENTY) && bits < (n)) { \
            Tracevv((stderr, "Still need %d bits for %s (hold:%0*lb (%d bits)).\n", (int)(n), (use), bits, hold, bits)); \
            goto input_empty; \
        } \
    } while (0)

static INLINE_ALWAYS uint64_t read_64le(uint8_t const* ptr) {
    return *(uint64_t const*)ptr;
}

static INLINE_ALWAYS void inflate_fastish_core(struct inflate_state FAR *state, z_streamp strm, unsigned output_size, int const assumptions) {
    unsigned char FAR* outptr = strm->next_out;            /* next output */
    z_const unsigned char FAR* inptr = strm->next_in;    /* next input */
    unsigned int avail_out = strm->avail_out;                /* available output */
    unsigned int avail_in = strm->avail_in;                 /* available input */
    uint64_t hold = state->hold;                        /* bit buffer */
    unsigned int bits = state->bits;                    /* bits in bit buffer */

    state->mode = LEN;

    code const* len_table = state->lencode;
    code const* dist_table = state->distcode;
    int state_was = 0, state_length = 0;
    int state_offset = 0;

    code const* lenptr = state->lencode;
    uint32_t lenpeekmask = ~(~0ull << state->lenbits);
    code const* distptr = state->distcode;
    uint32_t distpeekmask = ~(~0ull << state->distbits);
    uint64_t hold_rewind = hold;
    int bits_rewind = bits;

    while (avail_out > 0) {
        /* Ideally (64-bits)/8, but letting bits==64 causes an illegal shift elsewhere. */
        int chomp_size = (63u - bits) >> 3;
        if (avail_in < sizeof(uint64_t)) {
            if (CAN_ASSUME(INPUT_APLENTY)) goto input_empty;
            chomp_size = chomp_size <= avail_in ? chomp_size : avail_in;
            while (chomp_size > 0) {
                hold |= (uint64_t)*inptr << bits;
                bits += 8;
                inptr++;
                avail_in--;
                chomp_size--;
            }
        } else {
            hold |= read_64le(inptr) << bits;
            bits += chomp_size * 8;
            inptr += chomp_size; 
            avail_in -= chomp_size; 
        }

        // Save for bailout.
        hold_rewind = hold;
        bits_rewind = bits;

        code here = lenptr[hold & lenpeekmask];
        NEEDBITS(here.bits, "length code");

        if (here.op && (here.op & 0xf0) == 0) {
            unsigned long mask = ~(~0ul << (here.bits + here.op));
            code last = here;
            here = lenptr[last.val + ((hold & mask) >> last.bits)];
            here.bits += last.bits;  // TODO: fold into table?
            NEEDBITS(here.bits, "length bounce");
        }
        uint64_t bit_stash_raw = hold;
        uint64_t bit_stash = hold & ~(~0ul << here.bits);

        hold >>= here.bits;
        bits -= here.bits;

        if (here.op == 0) {
            Tracevv((stderr, here.val >= 0x20 && here.val < 0x7f ?
                    "inflate:         literal '%c\t\t! %0*lb\n" :
                    "inflate:         literal %d\t\t! %0*lb\n",
                    here.val, here.bits, (unsigned long)bit_stash));
            *outptr = (uint8_t)here.val;
            outptr++;
            avail_out--;
            continue;
        }

        // do this after literal handling, so we don't contaminate state_length
        int extra = (unsigned)here.op & 15;
        state_length = (unsigned)here.val;
        state_length += (bit_stash << extra) >> here.bits;

        code here2 = distptr[hold & distpeekmask];
        NEEDBITS(here2.bits, "distance code");
        if ((here2.op & 0xf0) == 0) {
            unsigned long mask = ~(~0ul << (here2.bits + here2.op));
            code last = here2;
            here2 = distptr[last.val + ((hold & mask) >> last.bits)];
            here2.bits += last.bits;  // TODO: fold into table?
            NEEDBITS(here2.bits, "distance bounce");
        }

        uint64_t bit_stash2 = hold & ~(~0ul << here2.bits);

        if (here.op >= 32) {
            if (here.op & 32) {
                Tracevv((stderr, "inflate:         end of block\n"));
                state->mode = TYPE;
                break;
            }
            if (here.op & 64) {
                strm->msg = (char *)"invalid literal/length code";
                state->mode = BAD;
                break;
            }
        }
        hold >>= here2.bits;
        bits -= here2.bits;

        state_was = state_length;

        if (here2.op & 64) {
            strm->msg = (char *)"invalid distance code";
            state->mode = BAD;
            break;
        }

        extra = (unsigned)here2.op & 15;
        state_offset = (unsigned)here2.val;
        state_offset += (bit_stash2 << extra) >> here2.bits;
#ifdef INFLATE_STRICT
        if (state_offset > state->dmax) {
            strm->msg = (char *)"invalid distance too far back";
            state->mode = BAD;
            break;
        }
#endif
        Tracevv((stderr, "inflate:         match %u %u\t\t! %0*lb %0*lb\n",
                    state_length, state_offset,
                    here2.bits, (unsigned long)bit_stash2,
                    here.bits, (unsigned long)bit_stash));
        int hardstop = CANNOT_ASSUME(NOHARDSTOP) && (avail_out < CHUNKCOPY_CHUNK_SIZE);
        int output_so_far = (output_size - avail_out);
        int copy = state_length;
        if (copy > avail_out) copy = avail_out;

        /* We try not to rely on the window and hope the output buffer is large enough */
        if (__builtin_expect(state_offset > output_so_far, 0)) {         /* copy from window */
            int windist = state_offset - output_so_far;
            int wcopy = copy < windist ? copy : windist;
            if (windowcopy(outptr, state->window, state->wsize, state->whave, state->wnext, windist, wcopy) != 0 && state->sane) {
                strm->msg = (char *)"invalid distance too far back";
                state->mode = BAD;
                break;
            }
            outptr += wcopy;
            avail_out -= wcopy;
            copy -= wcopy;
            state_length -= wcopy;
            if (copy > 0) {
                /* copy from output */
                if (hardstop) {
                    COPY258_OVERLAP_HARDSTOP(outptr, state_offset, copy);
                } else {
                    COPY258_OVERLAP_NOHARDSTOP(outptr, state_offset, copy);
                }
                outptr += copy;
                avail_out -= copy;
                state_length -= copy;
            }
        }
        else {
            /* copy from output */
            if (hardstop) {
                COPY258_OVERLAP_HARDSTOP(outptr, state_offset, copy);
            } else {
                COPY258_OVERLAP_NOHARDSTOP(outptr, state_offset, copy);
            }
            outptr += copy;
            avail_out -= copy;
            state_length -= copy;
        }
        if (state_length > 0) {
            state->mode = MATCH;
            state->was = state_was;
            state->length = state_length;
            state->offset = state_offset;
            // TODO: check this:
            REPLACEBITS();
            state->back = state->bits - bits;
            RESTORE();
            return;
        }
    }
    state->back = -1;
    REPLACEBITS();
    RESTORE();
    return;

  input_empty:
    if (CAN_ASSUME(INPUT_APLENTY)) {
        Tracev((stderr, "Input low: avail_in:%d, hold:%0*lb (%d bits)\n", avail_in, bits, hold, bits));
    }
    else {
        hold = hold_rewind;
        bits = bits_rewind;
        Tracev((stderr, "Out of input: avail_in:%d, hold:%0*lb (%d bits)\n", avail_in, bits, hold, bits));
        // assert(avail_in == 0);
    }
    RESTORE();
    state->back = 0;
    state->was = state->length = state->offset = 0;
}

static INLINE_NEVER void inflate_fasterish(struct inflate_state FAR *state, z_streamp strm, unsigned output_size) {
    inflate_fastish_core(state, strm, output_size, ASSUME_INPUT_APLENTY);
}

static INLINE_NEVER void inflate_fastish_tail(struct inflate_state FAR *state, z_streamp strm, unsigned output_size) {
    inflate_fastish_core(state, strm, output_size, 0);
}

int ZLIB_INTERNAL inffast_wincopy(uint8_t* outptr, uint8_t const* window, int wsize, int whave, int wnext, int offset, int copy) {
    return windowcopy(outptr, window, wsize, whave, wnext, offset, copy);
}

void ZLIB_INTERNAL inffast_outcopy(uint8_t* outptr, int distance, int length) {
    COPY258_OVERLAP_HARDSTOP(outptr, distance, length);
}

void ZLIB_INTERNAL inflate_fastish(z_streamp strm, unsigned output_size) {
    struct inflate_state FAR *state = (struct inflate_state FAR *)strm->state;
    inflate_fasterish(state, strm, output_size);
    if (state->mode == LEN) inflate_fastish_tail(state, strm, output_size);
#ifdef ZLIB_DEBUG
    if (state->mode == LEN) {
        if (strm->avail_in > 0 && strm->avail_out > 0) {
            Trace((stderr, "Problem: invalid state for LEN return: avail_in:%u, avail_out:%u, len:%u\n", strm->avail_in, strm->avail_out, state->length));
        }
    }
    else if (state->mode == MATCH) {
        if (strm->avail_out > 0 || state->length <= 0) {
            Trace((stderr, "Problem: invalid state for MATCH return: avail_in:%u, avail_out:%u, len:%u\n", strm->avail_in, strm->avail_out, state->length));
        }
    }
#endif
}


/*
   Decode literal, length, and distance codes and write out the resulting
   literal and match bytes until either not enough input or output is
   available, an end-of-block is encountered, or a data error is encountered.
   When large enough input and output buffers are supplied to inflate(), for
   example, a 16K input buffer and a 64K output buffer, more than 95% of the
   inflate execution time is spent in this routine.

   Entry assumptions:

        state->mode == LEN
        strm->avail_in >= INFLATE_FAST_MIN_INPUT
        strm->avail_out >= INFLATE_FAST_MIN_OUTPUT
        start >= strm->avail_out
        state->bits < 8

   On return, state->mode is one of:

        LEN -- ran out of enough output space or enough available input
        TYPE -- reached end of block code, inflate() to interpret next block
        BAD -- error in block data

   Notes:

    - The maximum input bits used by a length/distance pair is 15 bits for the
      length code, 5 bits for the length extra, 15 bits for the distance code,
      and 13 bits for the distance extra.  This totals 48 bits, or six bytes.
      Therefore if strm->avail_in >= 6, then there is enough input to avoid
      checking for available input while decoding.

    - The maximum bytes that a single length/distance pair can output is 258
      bytes, which is the maximum length that can be coded.  inflate_fast()
      requires strm->avail_out >= 258 for each loop to avoid checking for
      output space.
 */
void ZLIB_INTERNAL inflate_fast(z_streamp strm, unsigned start) {
    struct inflate_state FAR *state;
    z_const unsigned char FAR *in;      /* local strm->next_in */
    z_const unsigned char FAR *last;    /* have enough input while in < last */
    unsigned char FAR *out;     /* local strm->next_out */
    unsigned char FAR *beg;     /* inflate()'s initial strm->next_out */
    unsigned char FAR *end;     /* while out < end, enough space available */
#ifdef INFLATE_STRICT
    unsigned dmax;              /* maximum distance from zlib header */
#endif
    unsigned wsize;             /* window size or zero if not using window */
    unsigned whave;             /* valid bytes in the window */
    unsigned wnext;             /* window write index */
    unsigned char FAR *window;  /* allocated sliding window, if wsize != 0 */
    unsigned long hold;         /* local strm->hold */
    unsigned bits;              /* local strm->bits */
    code const FAR *lcode;      /* local strm->lencode */
    code const FAR *dcode;      /* local strm->distcode */
    unsigned lmask;             /* mask for first level of length codes */
    unsigned dmask;             /* mask for first level of distance codes */
    code const *here;           /* retrieved table entry */
    unsigned op;                /* code bits, operation, extra bits, or */
                                /*  window position, window bytes to copy */
    unsigned len;               /* match length, unused bytes */
    unsigned dist;              /* match distance */
    unsigned char FAR *from;    /* where to copy match from */

    /* copy state to local variables */
    state = (struct inflate_state FAR *)strm->state;
    in = strm->next_in;
    last = in + (strm->avail_in - (INFLATE_FAST_MIN_INPUT - 1));
    out = strm->next_out;
    beg = out - (start - strm->avail_out);
    end = out + (strm->avail_out - (INFLATE_FAST_MIN_OUTPUT - 1));
#ifdef INFLATE_STRICT
    dmax = state->dmax;
#endif
    wsize = state->wsize;
    whave = state->whave;
    wnext = state->wnext;
    window = state->window;
    hold = state->hold;
    bits = state->bits;
    lcode = state->lencode;
    dcode = state->distcode;
    lmask = (1U << state->lenbits) - 1;
    dmask = (1U << state->distbits) - 1;

    /* decode literals and length/distances until end-of-block or not enough
       input data or output space */
    do {
        if (bits < 15) {
            hold += (unsigned long)(*in++) << bits;
            bits += 8;
            hold += (unsigned long)(*in++) << bits;
            bits += 8;
        }
        here = lcode + (hold & lmask);
      dolen:
        op = (unsigned)(here->bits);
        hold >>= op;
        bits -= op;
        op = (unsigned)(here->op);
        if (op == 0) {                          /* literal */
            Tracevv((stderr, here->val >= 0x20 && here->val < 0x7f ?
                    "inflate:         literal '%c'\n" :
                    "inflate:         literal 0x%02x\n", here->val));
            *out++ = (unsigned char)(here->val);
        }
        else if (op & 16) {                     /* length base */
            len = (unsigned)(here->val);
            op &= 15;                           /* number of extra bits */
            if (op) {
                if (bits < op) {
                    hold += (unsigned long)(*in++) << bits;
                    bits += 8;
                }
                len += (unsigned)hold & ((1U << op) - 1);
                hold >>= op;
                bits -= op;
            }
            Tracevv((stderr, "inflate:         length %u\n", len));
            if (bits < 15) {
                hold += (unsigned long)(*in++) << bits;
                bits += 8;
                hold += (unsigned long)(*in++) << bits;
                bits += 8;
            }
            here = dcode + (hold & dmask);
          dodist:
            op = (unsigned)(here->bits);
            hold >>= op;
            bits -= op;
            op = (unsigned)(here->op);
            if (op & 16) {                      /* distance base */
                dist = (unsigned)(here->val);
                op &= 15;                       /* number of extra bits */
                if (bits < op) {
                    hold += (unsigned long)(*in++) << bits;
                    bits += 8;
                    if (bits < op) {
                        hold += (unsigned long)(*in++) << bits;
                        bits += 8;
                    }
                }
                dist += (unsigned)hold & ((1U << op) - 1);
#ifdef INFLATE_STRICT
                if (dist > dmax) {
                    strm->msg = (char *)"invalid distance too far back";
                    state->mode = BAD;
                    break;
                }
#endif
                hold >>= op;
                bits -= op;
                Tracevv((stderr, "inflate:         distance %u\n", dist));
                op = (unsigned)(out - beg);     /* max distance in output */
                if (dist > op) {                /* see if copy from window */
                    op = dist - op;             /* distance back in window */
                    if (op > whave) {
                        if (state->sane) {
                            strm->msg =
                                (char *)"invalid distance too far back";
                            state->mode = BAD;
                            break;
                        }
#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
                        if (len <= op - whave) {
                            do {
                                *out++ = 0;
                            } while (--len);
                            continue;
                        }
                        len -= op - whave;
                        do {
                            *out++ = 0;
                        } while (--op > whave);
                        if (op == 0) {
                            from = out - dist;
                            do {
                                *out++ = *from++;
                            } while (--len);
                            continue;
                        }
#endif
                    }
                    from = window;
                    if (wnext == 0) {           /* very common case */
                        from += wsize - op;
                        if (op < len) {         /* some from window */
                            len -= op;
                            do {
                                *out++ = *from++;
                            } while (--op);
                            from = out - dist;  /* rest from output */
                        }
                    }
                    else if (wnext < op) {      /* wrap around window */
                        from += wsize + wnext - op;
                        op -= wnext;
                        if (op < len) {         /* some from end of window */
                            len -= op;
                            do {
                                *out++ = *from++;
                            } while (--op);
                            from = window;
                            if (wnext < len) {  /* some from start of window */
                                op = wnext;
                                len -= op;
                                do {
                                    *out++ = *from++;
                                } while (--op);
                                from = out - dist;      /* rest from output */
                            }
                        }
                    }
                    else {                      /* contiguous in window */
                        from += wnext - op;
                        if (op < len) {         /* some from window */
                            len -= op;
                            do {
                                *out++ = *from++;
                            } while (--op);
                            from = out - dist;  /* rest from output */
                        }
                    }
                    while (len > 2) {
                        *out++ = *from++;
                        *out++ = *from++;
                        *out++ = *from++;
                        len -= 3;
                    }
                    if (len) {
                        *out++ = *from++;
                        if (len > 1)
                            *out++ = *from++;
                    }
                }
                else {
                    from = out - dist;          /* copy direct from output */
                    do {                        /* minimum length is three */
                        *out++ = *from++;
                        *out++ = *from++;
                        *out++ = *from++;
                        len -= 3;
                    } while (len > 2);
                    if (len) {
                        *out++ = *from++;
                        if (len > 1)
                            *out++ = *from++;
                    }
                }
            }
            else if ((op & 64) == 0) {          /* 2nd level distance code */
                here = dcode + here->val + (hold & ((1U << op) - 1));
                goto dodist;
            }
            else {
                strm->msg = (char *)"invalid distance code";
                state->mode = BAD;
                break;
            }
        }
        else if ((op & 64) == 0) {              /* 2nd level length code */
            here = lcode + here->val + (hold & ((1U << op) - 1));
            goto dolen;
        }
        else if (op & 32) {                     /* end-of-block */
            Tracevv((stderr, "inflate:         end of block\n"));
            state->mode = TYPE;
            break;
        }
        else {
            strm->msg = (char *)"invalid literal/length code";
            state->mode = BAD;
            break;
        }
    } while (in < last && out < end);

    /* return unused bytes (on entry, bits < 8, so in won't go too far back) */
    len = bits >> 3;
    in -= len;
    bits -= len << 3;
    hold &= (1U << bits) - 1;

    /* update state and return */
    strm->next_in = in;
    strm->next_out = out;
    strm->avail_in = (unsigned)(in < last ?
        (INFLATE_FAST_MIN_INPUT - 1) + (last - in) :
        (INFLATE_FAST_MIN_INPUT - 1) - (in - last));
    strm->avail_out = (unsigned)(out < end ?
        (INFLATE_FAST_MIN_OUTPUT - 1) + (end - out) :
        (INFLATE_FAST_MIN_OUTPUT - 1) - (out - end));
    state->hold = hold;
    state->bits = bits;
    return;
}

/*
   inflate_fast() speedups that turned out slower (on a PowerPC G3 750CXe):
   - Using bit fields for code structure
   - Different op definition to avoid & for extra bits (do & for table bits)
   - Three separate decoding do-loops for direct, window, and wnext == 0
   - Special case for distance > 1 copies to do overlapped load and store copy
   - Explicit branch predictions (based on measured branch probabilities)
   - Deferring match copy and interspersed it with decoding subsequent codes
   - Swapping literal/length else
   - Swapping window/direct else
   - Larger unrolled copy loops (three is about right)
   - Moving len -= 3 statement into middle of loop
 */

#endif /* !ASMINF */
