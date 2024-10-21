__kernel void dnb2_digital_shift(
    // Digital shift base 2 digital net 
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong r_x, // replications of xb
    __global const ulong *lshifts, // left shift applied to each element of xb
    __global const ulong *xb, // binary base 2 digital net points of size r_x*n*d
    __global const ulong *shiftsb, // digital shifts of size r*d
    __global ulong *xrb // digital shifted digital net points of size r*n*d
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ii_max = (n-i0)<batch_size_n ? (n-i0):batch_size_n;
    ulong jj_max = (d-j0)<batch_size_d ? (d-j0):batch_size_d;
    ulong ll_max = (r-l0)<batch_size_r ? (r-l0):batch_size_r;
    ulong ll,l,ii,i,jj,j,idx;
    ulong nelem_x = r_x*n*d;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            for(jj=0; jj<jj_max; jj++){
                j = j0+jj;
                idx = l*n*d+i*d+j;
                xrb[idx] = (xb[(idx)%nelem_x]<<lshifts[l%r_x])^shiftsb[l*d+j];
            }
        }
    }
}
