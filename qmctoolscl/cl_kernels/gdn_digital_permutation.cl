__kernel void gdn_digital_permutation(
    // Permutation of digits for a generalized digital net
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong r_x, // replications of xdig
    const ulong r_b, // replications of bases
    const ulong tmax, // rows of each generating matrix
    const ulong tmax_new, // rows of each new generating matrix
    const ulong bmax, // common permutation size, typically the maximum basis
    __global const ulong *perms, // permutations of size r*d*tmax_new*bmax
    __global const ulong *xdig, // binary digital net points of size r_x*n*d*tmax
    __global ulong *xdig_new // float digital net points of size r*n*d*tmax_new
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ii_max = (n-i0)<batch_size_n ? (n-i0):batch_size_n;
    ulong jj_max = (d-j0)<batch_size_d ? (d-j0):batch_size_d;
    ulong ll_max = (r-l0)<batch_size_r ? (r-l0):batch_size_r;
    ulong ll,l,ii,i,jj,j,t,idx_xdig,idx_xdig_new,idx_perm,p;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            for(jj=0; jj<jj_max; jj++){
                j = j0+jj;
                idx_xdig = (l%r_x)*n*d*tmax+i*d*tmax+j*tmax;
                idx_xdig_new = l*n*d*tmax_new+i*d*tmax_new+j*tmax_new;
                idx_perm = l*d*tmax_new*bmax+j*tmax_new*bmax;
                for(t=0; t<tmax; t++){
                    p = xdig[idx_xdig+t];
                    xdig_new[idx_xdig_new+t] = perms[idx_perm+t*bmax+p];
                }
                for(t=tmax; t<tmax_new; t++){
                    xdig_new[idx_xdig_new+t] = perms[idx_perm+t*bmax]; // index 0 of the permutation 
                }
            }
        }
    }
}
