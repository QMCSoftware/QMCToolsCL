__kernel void gdn_digital_shift(
    // Digital shift a generalized digital net
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong bs_r, // batch size for replications
    const ulong bs_n, // batch size for points
    const ulong bs_d, // batch size for dimension
    const ulong r_x, // replications of xdig
    const ulong r_b, // replications of bases
    const ulong tmax, // rows of each generating matrix
    const ulong tmax_new, // rows of each new generating matrix
    __global const ulong *bases, // bases for each dimension of size r_b*d
    __global const ulong *shifts, // digital shifts of size r*d*tmax_new
    __global const ulong *xdig, // binary digital net points of size r_x*n*d*tmax
    __global ulong *xdig_new // float digital net points of size r*n*d*tmax_new
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong i0 = get_global_id(1)*bs_n;
    ulong j0 = get_global_id(2)*bs_d;
    ulong ii_max = (n-i0)<bs_n ? (n-i0):bs_n;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong b,ll,l,ii,i,jj,j,t,idx_xdig,idx_xdig_new,idx_shift;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            for(jj=0; jj<jj_max; jj++){
                j = j0+jj;
                idx_xdig = (l%r_x)*n*d*tmax+i*d*tmax+j*tmax;
                idx_xdig_new = l*n*d*tmax_new+i*d*tmax_new+j*tmax_new;
                idx_shift = l*d*tmax_new+j*tmax_new;
                b = bases[(l%r_b)*d+j];
                for(t=0; t<tmax; t++){
                    xdig_new[idx_xdig_new+t] = (xdig[idx_xdig+t]+shifts[idx_shift+t])%b;
                }
                for(t=tmax; t<tmax_new; t++){
                    xdig_new[idx_xdig_new+t] = shifts[idx_shift+t];
                }
            }
        }
    }
}