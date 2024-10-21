__kernel void gdn_gen_natural(
    // Generalized digital net where the base can be different for each dimension e.g. for the Halton sequence
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong bs_r, // batch size for replications
    const ulong bs_n, // batch size for points
    const ulong bs_d, // batch size for dimension
    const ulong r_b, // number of replications of bases
    const ulong mmax, // columns in each generating matrix
    const ulong tmax, // rows of each generating matrix
    const ulong n_start, // starting index in sequence
    __global const ulong *bases, // bases for each dimension of size r_b*d
    __global const ulong *C, // generating matrices of size r*d*mmax*tmax
    __global ulong *xdig // generalized digital net sequence of digits of size r*n*d*tmax
){   
    ulong l0 = get_global_id(0)*bs_r;
    ulong i0 = get_global_id(1)*bs_n;
    ulong j0 = get_global_id(2)*bs_d;
    ulong ii_max = (n-i0)<bs_n ? (n-i0):bs_n;
    ulong jj_max = (d-j0)<bs_d ? (d-j0):bs_d;
    ulong ll_max = (r-l0)<bs_r ? (r-l0):bs_r;
    ulong idx_xdig,idx_C,b,dig,itrue,icp,ii,i,jj,j,ll,l,t,k;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            itrue = i+n_start;
            for(jj=0; jj<jj_max; jj++){
                j = j0+jj;
                idx_xdig = l*n*d*tmax+i*d*tmax+j*tmax;
                for(t=0; t<tmax; t++){
                    xdig[idx_xdig+t] = 0;
                }
                b = bases[(l%r_b)*d+j];
                k = 0;
                icp = itrue; 
                while(icp>0){
                    dig = icp%b;
                    icp = (icp-dig)/b;
                    if(dig>0){
                        idx_C = l*d*mmax*tmax+j*mmax*tmax+k*tmax;
                        for(t=0; t<tmax; t++){
                            xdig[idx_xdig+t] = (xdig[idx_xdig+t]+dig*C[idx_C+t])%b;
                        }
                    }
                    k += 1;
                }
            }
        }
    }
}