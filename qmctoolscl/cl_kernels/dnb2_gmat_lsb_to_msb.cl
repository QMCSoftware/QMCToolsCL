__kernel void dnb2_gmat_lsb_to_msb(
    // Convert base 2 generating matrices with integers stored in Least Significant Bit order to Most Significant Bit order
    const ulong r, // replications
    const ulong d, // dimension
    const ulong mmax, // columns in each generating matrix 
    const ulong bs_r, // batch size for replications
    const ulong bs_d, // batch size for dimensions
    const ulong bs_mmax, // batch size for columns
    __global const ulong *tmaxes, // length r vector of bits in each integer of the resulting MSB generating matrices
    __global const ulong *C_lsb, // original generating matrices of size r*d*mmax
    __global ulong *C_msb // new generating matrices of size r*d*mmax
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong j0 = get_global_id(1)*bs_d;
    ulong k0 = get_global_id(2)*bs_mmax;
    ulong kk_max = (mmax-k0)<bs_mmax ? (mmax-k0):bs_mmax;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong tmax,t,ll,l,jj,j,kk,k,v,vnew,idx;
    ulong bigone = 1;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        tmax = tmaxes[l];
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            for(kk=0; kk<kk_max; kk++){
                k = k0+kk;
                idx = l*d*mmax+j*mmax+k;
                v = C_lsb[idx];
                vnew = 0;
                t = 0; 
                while(v!=0){
                    if(v&1){
                        vnew += bigone<<(tmax-t-1);
                    }
                    v >>= 1;
                    t += 1;
                }
                C_msb[idx] = vnew;
            }
        }
    }
}