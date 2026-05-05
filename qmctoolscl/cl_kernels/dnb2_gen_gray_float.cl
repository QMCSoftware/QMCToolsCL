__kernel void dnb2_gen_gray_float(
    // Gray-order base-2 digital net directly converted to floats
    const ulong r, // replications in x
    const ulong n, // points
    const ulong d, // dimension
    const ulong bs_r, // batch size for replications
    const ulong bs_n, // batch size for points
    const ulong bs_d, // batch size for dimension
    const ulong n_start, // starting index in sequence
    const ulong mmax, // columns in each generating matrix
    const ulong r_x, // replications of generating matrices
    const char apply_shift, // whether to apply digital shifts
    __global const ulong *lshifts, // left shift applied before digital shift, size r
    __global const ulong *shiftsb, // digital shifts of size r*d
    __global const ulong *tmaxes, // bits in integer representation of size r
    __global const ulong *C, // generating matrices of size r_x*d*mmax
    __global double *x // float digital net points of size r*n*d
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong i0 = get_global_id(1)*bs_n;
    ulong j0 = get_global_id(2)*bs_d;
    ulong ii_max = (n-i0)<bs_n ? (n-i0):bs_n;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong b,t,ll,l,l_x,ii,i,jj,j,idx,cidx,shift_idx;
    ulong itrue,xb,val;
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
