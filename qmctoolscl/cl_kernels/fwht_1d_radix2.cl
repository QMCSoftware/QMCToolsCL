__kernel void fwht_1d_radix2(
    // Fast Walsh-Hadamard Transform for real valued inputs.
    // FWHT is done in place along the last dimension where the size is required to be a power of 2. 
    // Follows the divide-and-conquer algorithm described in https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
    const ulong d1, // first dimension
    const ulong d2, // second dimension
    const ulong n_half, // half of the last dimension along which FWHT is performed
    const ulong bs_d1, // batch size first dimension 
    const ulong bs_d2, // batch size second dimension
    const ulong bs_n_half, // batch size for half of the last dimension
    __global double *x // array of size d1*d2*2n_half on which to perform FWHT in place
){
    ulong j10 = get_global_id(0)*bs_d1;
    ulong j20 = get_global_id(1)*bs_d2;
    ulong i0 = get_global_id(2)*bs_n_half;
    ulong ii_max = (n_half-i0)<bs_n_half ? (n_half-i0):bs_n_half;
    ulong jj1_max = (d1-j10)<bs_d1 ? (d1-j10):bs_d1;
    ulong jj2_max = (d2-j20)<bs_d2 ? (d2-j20):bs_d2;
    ulong ii,i,i1,i2,jj1,jj2,j1,j2,k,s,f,idx;
    double x1,x2;
    ulong n = 2*n_half;
    ulong m = (ulong)(log2((double)n));
    double sqrt2 = sqrt((double)2);
    for(k=0; k<m; k++){
        s = m-k-1;
        f = 1<<s; 
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            if((i>>s)&1){
                i2 = i+n_half;
                i1 = i2^f;
            }
            else{
                i1 = i;
                i2 = i1^f;
            }
            for(jj1=0; jj1<jj1_max; jj1++){
                j1 = j10+jj1;
                for(jj2=0; jj2<jj2_max; jj2++){
                    j2 = j20+jj2;
                    idx = j1*d2*n+j2*n;
                    x1 = x[idx+i1];
                    x2 = x[idx+i2];
                    x[idx+i1] = (x1+x2)/sqrt2;
                    x[idx+i2] = (x1-x2)/sqrt2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}