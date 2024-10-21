__kernel void lat_gen_gray(
    // Lattice points in Gray code order
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong n_start, // starting index in sequence
    __global const ulong *g, // pointer to generating vector of size r*d 
    __global double *x // pointer to point storage of size r*n*d
){   
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ii_max = (n-i0)<batch_size_n ? (n-i0):batch_size_n;
    ulong jj_max = (d-j0)<batch_size_d ? (d-j0):batch_size_d;
    ulong ll_max = (r-l0)<batch_size_r ? (r-l0):batch_size_r;
    double ifrac;
    ulong p,v,itrue,b,ll,l,ii,i,jj,j,idx;
    ulong n0 = n_start+i0;
    p = ceil(log2((double)n0+1));
    v = 0; 
    b = 0;
    ulong t = n0^(n0>>1);
    while(t>0){
        if(t&1){
            v+= 1<<(p-b-1);
        }
        b += 1;
        t >>= 1;
    }
    for(ii=0; ii<ii_max; ii++){
        i = i0+ii;
        ifrac = ldexp((double)v,-p);
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            idx = i*d+j;
            for(ll=0; ll<ll_max; ll++){
                l = l0+ll;
                x[l*n*d+idx] = (double)(fmod((double)(g[l*d+j]*ifrac),(double)(1.)));
            }
        }
        itrue = i+n_start+1;
        if((itrue&(itrue-1))==0){ // if itrue>0 is a power of 2
            p += 1;
            v <<= 1;
        }
        b = 0;
        while(!((itrue>>b)&1)){
            b += 1;
        }
        v ^= 1<<(p-b-1);
    }
}