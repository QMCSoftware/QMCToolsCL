__kernel void gdn_linear_matrix_scramble(
    // Linear matrix scramble for generalized digital net 
    const ulong r, // replications 
    const ulong d, // dimension 
    const ulong mmax, // columns in each generating matrix
    const ulong bs_r, // batch size for replications
    const ulong bs_d, // batch size for dimension
    const ulong bs_mmax, // batch size columns
    const ulong r_C, // number of replications of C 
    const ulong r_b, // number of replications of bases
    const ulong tmax, // number of rows in each generating matrix 
    const ulong tmax_new, // new number of rows in each generating matrix 
    __global const ulong *bases, // bases for each dimension of size r*d 
    __global const ulong *S, // scramble matrices of size r*d*tmax_new*tmax
    __global const ulong *C, // generating matrices of size r_C*d*mmax*tmax 
    __global ulong *C_lms // new generating matrices of size r*d*mmax*tmax_new
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong j0 = get_global_id(1)*bs_d;
    ulong k0 = get_global_id(2)*bs_mmax;
    ulong kk_max = (mmax-k0)<bs_mmax ? (mmax-k0):bs_mmax;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong ll,l,jj,j,kk,k,t,c,b,v,idx_C,idx_C_lms,idx_S; 
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            b = bases[(l%r_b)*d+j];
            for(kk=0; kk<kk_max; kk++){
                k = k0+kk;
                idx_C = (l%r_C)*d*mmax*tmax+j*mmax*tmax+k*tmax;
                idx_C_lms = l*d*mmax*tmax_new+j*mmax*tmax_new+k*tmax_new;
                for(t=0; t<tmax_new; t++){
                    v = 0;
                    idx_S = l*d*tmax_new*tmax+j*tmax_new*tmax+t*tmax;
                    for(c=0; c<tmax; c++){
                        v += (S[idx_S+c]*C[idx_C+c])%b;
                    }
                    C_lms[idx_C_lms+t] = v;
                }
            }
        }
    }
}