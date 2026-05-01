#include "qmctoolscl.h"

EXPORT void dnb2_gen_gray_float(
    // Gray-order base-2 digital net directly converted to floats
    const unsigned long long r, // replications in x
    const unsigned long long n, // points
    const unsigned long long d, // dimension
    const unsigned long long bs_r, // batch size for replications
    const unsigned long long bs_n, // batch size for points
    const unsigned long long bs_d, // batch size for dimension
    const unsigned long long n_start, // starting index in sequence
    const unsigned long long mmax, // columns in each generating matrix
    const unsigned long long r_x, // replications of generating matrices
    const unsigned char apply_shift, // whether to apply digital shifts
    const unsigned long long *lshifts, // left shift applied before digital shift, size r
    const unsigned long long *shiftsb, // digital shifts of size r*d
    const unsigned long long *tmaxes, // bits in integer representation of size r
    const unsigned long long *C, // generating matrices of size r_x*d*mmax
    double *x // float digital net points of size r*n*d
){
    unsigned long long l0 = 0*bs_r;
    unsigned long long i0 = 0*bs_n;
    unsigned long long j0 = 0*bs_d;
    unsigned long long ii_max = (n-i0)<bs_n ? (n-i0):bs_n;
    unsigned long long jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    unsigned long long ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    unsigned long long b,t,ll,l,l_x,ii,i,jj,j,idx,cidx,shift_idx;
    unsigned long long itrue,xb,val;
    double scale;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        l_x = l%r_x;
        scale = ldexp(1.0,-(int)tmaxes[l]);
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            itrue = n_start+i0;
            t = itrue^(itrue>>1);
            xb = 0;
            b = 0;
            while(t>0){
                if(t&1){
                    cidx = l_x*d*mmax+j*mmax+b;
                    xb ^= C[cidx];
                }
                b += 1;
                t >>= 1;
            }
            shift_idx = l*d+j;
            for(ii=0; ii<ii_max; ii++){
                i = i0+ii;
                if(ii>0){
                    itrue = i+n_start;
                    b = 0;
                    while(!((itrue>>b)&1)){
                        b += 1;
                    }
                    cidx = l_x*d*mmax+j*mmax+b;
                    xb ^= C[cidx];
                }
                idx = l*n*d+i*d+j;
                val = apply_shift ? ((xb<<lshifts[l%r_x])^shiftsb[shift_idx]) : xb;
                x[idx] = ((double)(val))*scale;
            }
        }
    }
}
