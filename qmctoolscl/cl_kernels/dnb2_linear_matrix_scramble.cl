__kernel void dnb2_linear_matrix_scramble(
    // Linear matrix scrambling for base 2 generating matrices
    const ulong r, // replications
    const ulong d, // dimension
    const ulong mmax, // columns in each generating matrix 
    const ulong bs_r, // batch size for replications
    const ulong bs_d, // batch size for dimensions
    const ulong bs_mmax, // batch size for columns
    const ulong r_C, // original generating matrices
    const ulong tmax_new, // bits in the integers of the resulting generating matrices
    __global const ulong *S, // scrambling matrices of size r*d*tmax_new
    __global const ulong *C, // original generating matrices of size r_C*d*mmax
    __global ulong *C_lms // resulting generating matrices of size r*d*mmax
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong j0 = get_global_id(1)*bs_d;
    ulong k0 = get_global_id(2)*bs_mmax;
    ulong kk_max = (mmax-k0)<bs_mmax ? (mmax-k0):bs_mmax;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong b,t,ll,l,jj,j,kk,k,u,v,udotv,vnew,idx;
    ulong bigone = 1;
    ulong nelemC = r_C*d*mmax;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            for(kk=0; kk<kk_max; kk++){
                k = k0+kk;
                idx = l*d*mmax+j*mmax+k;
                v = C[idx%nelemC];
                vnew = 0;
                for(t=0; t<tmax_new; t++){
                    u = S[l*d*tmax_new+j*tmax_new+t];
                    udotv = u&v;
                    // Brian Kernighan algorithm: https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
                    b = 0;
                    while(udotv){
                        b += 1;
                        udotv &= (udotv-1);
                    }
                    if((b%2)==1){
                        vnew += bigone<<(tmax_new-t-1);
                    }
                }
                C_lms[idx] = vnew;
            }
        }
    }
}