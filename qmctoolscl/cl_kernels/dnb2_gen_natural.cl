__kernel void dnb2_gen_natural(
    // Binary representation of digital net in base 2 in natural order
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong n_start, // starting index in sequence
    const ulong mmax, // columns in each generating matrix
    __global const ulong *C, // generating matrices of size r*d*mmax
    __global ulong *xb // binary digital net points of size r*n*d
){   
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ii_max = (n-i0)<batch_size_n ? (n-i0):batch_size_n;
    ulong jj_max = (d-j0)<batch_size_d ? (d-j0):batch_size_d;
    ulong ll_max = (r-l0)<batch_size_r ? (r-l0):batch_size_r;
    ulong b,t,ll,l,ii,i,jj,j,prev_i,new_i;
    ulong itrue = n_start+i0;
    // initial index 
    t = itrue^(itrue>>1);
    prev_i = (t-n_start)*d;
    if(n>0){
        // initialize first values 0 
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            for(ll=0; ll<ll_max; ll++){
                l = l0+ll;
                xb[l*n*d+prev_i+j] = 0;
            }
        }
        // set first values
        b = 0;
        while(t>0){
            if(t&1){
                for(jj=0; jj<jj_max; jj++){
                    j = j0+jj;
                    for(ll=0; ll<ll_max; ll++){
                        l = l0+ll;
                        xb[l*n*d+prev_i+j] ^= C[l*d*mmax+j*mmax+b];
                    }
                }
            }
            b += 1;
            t >>= 1;
        }
    }
    // set remaining values
    for(ii=1; ii<ii_max; ii++){
        i = i0+ii;
        itrue = i+n_start;
        t = itrue^(itrue>>1);
        new_i = (t-n_start)*d;
        b = 0;
        while(!((itrue>>b)&1)){
            b += 1;
        }
        for(jj=0; jj<jj_max; jj++){
            j = j0+jj;
            for(ll=0; ll<ll_max; ll++){
                l = l0+ll;
                xb[l*n*d+new_i+j] = xb[l*n*d+prev_i+j]^C[l*d*mmax+j*mmax+b];
            }
        }
        prev_i = new_i;
    }
}