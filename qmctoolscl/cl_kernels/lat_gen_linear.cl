__kernel void lat_gen_linear(
    // Lattice points in linear order
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong bs_r, // batch size for replications
    const ulong bs_n, // batch size for points
    const ulong bs_d, // batch size for dimension
    __global const ulong *g, // pointer to generating vector of size r*d
    __global double *x // pointer to point storage of size r*n*d
){
    ulong l0 = get_global_id(0)*bs_r;
    ulong i0 = get_global_id(1)*bs_n;
    ulong j0 = get_global_id(2)*bs_d;
    ulong ii_max = (n-i0)<bs_n ? (n-i0):bs_n;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    double n_double = n;
    double ifrac;
    ulong ll,l,ii,i,jj,j;
    for(ii=0; ii<ii_max; ii++){
        i = i0+ii;
        ifrac = i/n_double;
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            for(ll=0; ll<ll_max; ll++){
                l = l0+ll;
                x[l*n*d+i*d+j] = (double)(fmod((double)(g[l*d+j]*ifrac),(double)(1.)));
            }
        }
    }
}