__kernel void gdn_integer_to_float(
    // Convert digits of generalized digital net to floats
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong bs_r, // batch size for replications
    const ulong bs_n, // batch size for points
    const ulong bs_d, // batch size for dimension
    const ulong r_b, // replications of bases 
    const ulong tmax, // rows of each generating matrix
    __global const ulong *bases, // bases for each dimension of size r_b*d
    __global const ulong *xdig, // binary digital net points of size r*n*d*tmax
    __global double *x // float digital net points of size r*n*d
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong i0 = get_global_id(1)*bs_n;
    ulong j0 = get_global_id(2)*bs_d;
    ulong ii_max = (n-i0)<bs_n ? (n-i0):bs_n;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong ll,l,ii,i,jj,j,t,idx_xdig;
    double recip,v,xdig_double,b;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            for(jj=0; jj<jj_max; jj++){
                j = j0+jj;
                idx_xdig = l*n*d*tmax+i*d*tmax+j*tmax;
                v = 0.;
                b = (double) bases[(l%r_b)*d+j];
                recip = 1/b;
                for(t=0; t<tmax; t++){
                    xdig_double = (double) (xdig[idx_xdig+t]);
                    v += recip*xdig_double;
                    recip /= b;
                }
                x[l*n*d+i*d+j] = v;
            }
        }
    }
}