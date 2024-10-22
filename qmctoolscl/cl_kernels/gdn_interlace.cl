__kernel void gdn_interlace(
    // Interlace generating matrices or transpose of point sets to attain higher order digital nets
    const ulong r, // replications
    const ulong d_alpha, // dimension of resulting generating matrices 
    const ulong mmax, // columns of generating matrices
    const ulong bs_r, // batch size for replications
    const ulong bs_d_alpha, // batch size for dimension of resulting generating matrices
    const ulong bs_mmax, // batch size for replications
    const ulong d, // dimension of original generating matrices
    const ulong tmax, // rows of original generating matrices
    const ulong tmax_alpha, // rows of interlaced generating matrices
    const ulong alpha, // interlacing factor
    __global const ulong *C, // original generating matrices of size r*d*mmax*tmax
    __global ulong *C_alpha // resulting interlaced generating matrices of size r*d_alpha*mmax*tmax_alpha
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong j0_alpha = get_global_id(1)*bs_d_alpha;
    ulong k0 = get_global_id(2)*bs_mmax;
    ulong kk_max = (mmax-k0)<bs_mmax ? (mmax-k0):bs_mmax;
    ulong jj_alpha_max = (d_alpha-j0_alpha)<bs_d_alpha ? (d_alpha-j0_alpha):bs_d_alpha;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong ll,l,jj_alpha,j_alpha,kk,k,t_alpha,t,jj,j;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(jj_alpha=0; jj_alpha<jj_alpha_max; jj_alpha++){
            j_alpha = j0_alpha+jj_alpha;
             for(kk=0; kk<kk_max; kk++){
                k = k0+kk;
                for(t_alpha=0; t_alpha<tmax_alpha; t_alpha++){
                    t = t_alpha / alpha; 
                    jj = t_alpha%alpha; 
                    j = j_alpha*alpha+jj;
                    C_alpha[l*d_alpha*mmax*tmax_alpha+j_alpha*mmax*tmax_alpha+k*tmax_alpha+t_alpha] = C[l*d*mmax*tmax+j*mmax*tmax+k*tmax+t];
                }
            }
        }
    }
}