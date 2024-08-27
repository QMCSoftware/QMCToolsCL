__kernel void lattice_linear(
    const ulong r,
    const ulong n, 
    const ulong d, 
    const ulong batch_size_r_x,
    const ulong batch_size_n, 
    const ulong batch_size_d, 
    __global const ulong *g,
    __global double *x)
{
    ulong l0 = get_global_id(0)*batch_size_r_x;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    double n_double = n;
    double ifrac;
    ulong ll,l,ii,i,jj,j;
    for(ii=0; ii<batch_size_n; ii++){
        i = i0+ii;
        ifrac = i/n_double;
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
            for(ll=0; ll<batch_size_r_x; ll++){
                l = l0+ll;
                x[l*n*d+i*d+j] = fmod(g[l*d+j]*ifrac,1);
                if(l==(r-1)){
                    break;
                }
            }
            if(j==(d-1)){
                break;
            }
        }
        if(i==(n-1)){
            break;
        }
    }
}

__kernel void lattice_b2(
    const ulong r,
    const ulong n, 
    const ulong d,
    const ulong batch_size_r_x,
    const ulong batch_size_n, 
    const ulong batch_size_d, 
    const ulong nstart,
    const char gc,
    __global const ulong *g,
    __global double *x)
{   
    ulong l0 = get_global_id(0)*batch_size_r_x;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    double ifrac;
    ulong p,v,itrue,igc,b,ll,l,ii,i,jj,j,idx;
    ulong n0 = nstart+i0;
    if(n0==0){
        p = 0;
        v = 0;
    }
    else{
        p = ceil(log2((double)n0+1));
        v = 0; 
        b = 0;
        ulong t = gc ? n0 : n0^(n0>>1); 
        while(t>0){
            if(t&1){
                v+= 1<<(p-b-1);
            }
            b += 1;
            t >>= 1;
        }
    }
    for(ii=0; ii<batch_size_n; ii++){
        i = i0+ii;
        ifrac = ldexp((double)v,-p);
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
            if(gc){
                idx = i*d+j;
            }
            else{
                itrue = i+nstart;
                igc = itrue^(itrue>>1);
                idx = (igc-nstart)*d+j;
            }
            for(ll=0; ll<batch_size_r_x; ll++){
                l = l0+ll;
                x[l*n*d+idx] = fmod(g[l*d+j]*ifrac,1);
                if(l==(r-1)){
                    break;
                }
            }
            if(j==(d-1)){
                break;
            }
        }
        if((i==(n-1))||(ii==(batch_size_n-1))){
            break;
        }
        itrue = i+nstart+1;
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

__kernel void lattice_rshift(
    const ulong r,
    const ulong n, 
    const ulong d, 
    const ulong batch_size_r,
    const ulong batch_size_n, 
    const ulong batch_size_d,
    const ulong r_x,
    __global const double *x,
    __global const double *shifts,
    __global double *xr)
{
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ll,l,ii,i,jj,j,idx;
    ulong nelem_x = r_x*n*d;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(ii=0; ii<batch_size_n; ii++){
            i = i0+ii;
            for(jj=0; jj<batch_size_d; jj++){
                j = j0+jj;
                idx = l*n*d+i*d+j;
                xr[idx] = fmod(x[(idx)%nelem_x]+shifts[l*d+j],1);
                if(j==(d-1)){
                    break;
                }
            }
            if(i==(n-1)){
                break;
            }
        }
        if(l==(r-1)){
            break;
        }
    }
}
