__kernel void gdn_undo_interlace(
    // Undo interlacing of generating matrices 
    const ulong r, // replications
    const ulong d, // dimension of resulting generating matrices 
    const ulong mmax, // columns in generating matrices
    const ulong bs_r, // batch size of replications
    const ulong bs_d, // batch size of dimension of resulting generating matrices
    const ulong bs_mmax, // batch size of columns in generating matrices
    const ulong d_alpha, // dimension of interlaced generating matrices
    const ulong tmax, // rows of original generating matrices
    const ulong tmax_alpha, // rows of interlaced generating matrices
    const ulong alpha, // interlacing factor
    __global const ulong *C_alpha, // interlaced generating matrices of size r*d_alpha*mmax*tmax_alpha
    __global ulong *C // original generating matrices of size r*d*mmax*tmax
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong j0 = get_global_id(1)*bs_d;
    ulong k0 = get_global_id(2)*bs_mmax;
    ulong kk_max = (mmax-k0)<bs_mmax ? (mmax-k0):bs_mmax;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong ll,l,j_alpha,kk,k,t_alpha,tt_alpha,t,jj,j;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
             for(kk=0; kk<kk_max; kk++){
                k = k0+kk;
                for(t=0; t<tmax; t++){
                    j_alpha = j/alpha;
                    tt_alpha = j%alpha;
                    t_alpha = t*alpha+tt_alpha;
                    C[l*d*mmax*tmax+j*mmax*tmax+k*tmax+t] = C_alpha[l*d_alpha*mmax*tmax_alpha+j_alpha*mmax*tmax_alpha+k*tmax_alpha+t_alpha];
                }
            }
        }
    }
}