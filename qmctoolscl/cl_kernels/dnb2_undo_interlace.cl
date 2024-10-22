__kernel void dnb2_undo_interlace(
    // Undo interlacing of generating matrices in base 2
    const ulong r, // replications
    const ulong d, // dimension of resulting generating matrices 
    const ulong mmax, // columns in generating matrices
    const ulong bs_r, // batch size of replications
    const ulong bs_d, // batch size of dimension of resulting generating matrices
    const ulong bs_mmax, // batch size of columns in generating matrices
    const ulong d_alpha, // dimension of interlaced generating matrices
    const ulong tmax, // bits in integers of original generating matrices 
    const ulong tmax_alpha, // bits in integers of interlaced generating matrices
    const ulong alpha, // interlacing factor
    __global const ulong *C_alpha, // interlaced generating matrices of size r*d_alpha*mmax
    __global ulong *C // original generating matrices of size r*d*mmax
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong j0 = get_global_id(1)*bs_d;
    ulong k0 = get_global_id(2)*bs_mmax;
    ulong kk_max = (mmax-k0)<bs_mmax ? (mmax-k0):bs_mmax;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong ll,l,j_alpha,kk,k,t_alpha,tt_alpha,t,jj,j,v,b;
    ulong bigone = 1;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
             for(kk=0; kk<kk_max; kk++){
                k = k0+kk;
                v = 0;
                for(t=0; t<tmax; t++){
                    j_alpha = j/alpha;
                    tt_alpha = j%alpha;
                    t_alpha = t*alpha+tt_alpha;
                    b = (C_alpha[l*d_alpha*mmax+j_alpha*mmax+k]>>(tmax_alpha-t_alpha-1))&1;
                    if(b){
                        v += (bigone<<(tmax-t-1));
                    }
                }
                C[l*d*mmax+j*mmax+k] = v;
            }
        }
    }
}