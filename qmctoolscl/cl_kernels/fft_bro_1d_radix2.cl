__kernel void fft_bro_1d_radix2(
    // Fast Fourier Transform for inputs in bit reversed order.
    // FFT is done in place along the last dimension where the size is required to be a power of 2. 
    // Follows a decimation-in-time procedure described in https://www.cmlab.csie.ntu.edu.tw/cml/dsp/training/coding/transform/fft.html. 
    const ulong d1, // first dimension
    const ulong d2, // second dimension
    const ulong n_half, // half of the last dimension of size n = 2n_half along which FFT is performed
    const ulong bs_d1, // batch size first dimension 
    const ulong bs_d2, // batch size second dimension
    const ulong bs_n_half, // batch size for half of the last dimension
    __global double *twiddler, // size n vector used to store real twiddle factors
    __global double *twiddlei, // size n vector used to store imaginary twiddle factors 
    __global double *xr, // real array of size d1*d2*n on which to perform FFT in place
    __global double *xi // imaginary array of size d1*d2*n on which to perform FFT in place
){
    ulong j10 = get_global_id(0)*bs_d1;
    ulong j20 = get_global_id(1)*bs_d2;
    ulong i0 = get_global_id(2)*bs_n_half;
    ulong ii_max = (n_half-i0)<bs_n_half ? (n_half-i0):bs_n_half;
    ulong jj1_max = (d1-j10)<bs_d1 ? (d1-j10):bs_d1;
    ulong jj2_max = (d2-j20)<bs_d2 ? (d2-j20):bs_d2;
    ulong ii,i,i1,i2,t,jj1,jj2,j1,j2,k,s,f,idx;
    double xr1,xr2,xi1,xi2,yr,yi,v1,v2,cosv,sinv;
    double PI = acos(-1.);
    ulong n = 2*n_half;
    ulong m = (ulong)(log2((double)n));
    ulong bigone = 1;
    double sqrt2 = sqrt((double)2);
    // initialize twiddle factors
    for(ii=0; ii<ii_max; ii++){
        i = i0+ii;
        i1 = 2*i;
        i2 = i1+1;
        v1 = -2*PI*i1/n;
        twiddler[i1] = cos(v1);
        twiddlei[i1] = sin(v1);
        v2 = -2*PI*i2/n;
        twiddler[i2] = cos(v2);
        twiddlei[i2] = sin(v2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // remaining butterflies
    for(k=0; k<m; k++){
        s = m-k-1;
        f = 1<<k; 
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            if((i>>k)&1){
                i2 = i+n_half;
                i1 = i2^f;
            }
            else{
                i1 = i;
                i2 = i1^f;
            }
            t = (i1%f)*(bigone<<s);
            cosv = twiddler[t];
            sinv = twiddlei[t];
            for(jj1=0; jj1<jj1_max; jj1++){
                j1 = j10+jj1;
                for(jj2=0; jj2<jj2_max; jj2++){
                    j2 = j20+jj2;
                    idx = j1*d2*n+j2*n;
                    xr1 = xr[idx+i1];
                    xr2 = xr[idx+i2];
                    xi1 = xi[idx+i1];
                    xi2 = xi[idx+i2];
                    yr = xr2*cosv-xi2*sinv;
                    yi = xr2*sinv+xi2*cosv;
                    xr[idx+i1] = (xr1+yr)/sqrt2;
                    xi[idx+i1] = (xi1+yi)/sqrt2;
                    xr[idx+i2] = (xr1-yr)/sqrt2;
                    xi[idx+i2] = (xi1-yi)/sqrt2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}