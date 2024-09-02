__kernel void lattice_linear(
    // Lattice points in linear ordering so $x_{lij} = g_{lj} i/n mod 1$ for $0 \leq l < r$ and $0 \leq i < n$ and $0 \leq j < d$
    const ulong r, // number of replications
    const ulong n, // number of points
    const ulong d, // dimension of points
    const ulong batch_size_r_x, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    __global const ulong *g, // pointer to generating vector of size r*d
    __global double *x // pointer to point storage of size r*n*d
){
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

__kernel void gen_mats_lsb_to_msb_b2(
    const ulong r,
    const ulong d,
    const ulong mmax, 
    const ulong batch_size_r, 
    const ulong batch_size_d, 
    const ulong batch_size_mmax,
    __global const ulong *tmaxes,
    __global const ulong *C,
    __global ulong *Cnew)
{
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0 = get_global_id(1)*batch_size_d;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong tmax,t,ll,l,jj,j,kk,k,v,vnew,idx;
    ulong bigone = 1;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        tmax = tmaxes[l];
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
            for(kk=0; kk<batch_size_mmax; kk++){
                k = k0+kk;
                idx = l*d*mmax+j*mmax+k;
                v = C[idx];
                vnew = 0;
                t = 0; 
                while(v!=0){
                    if(v&1){
                        vnew += bigone<<(tmax-t-1);
                    }
                    v >>= 1;
                    t += 1;
                }
                Cnew[idx] = vnew;
                if(k==(mmax-1)){
                    break;
                }
            }
            if(j==(d-1)){
                break;
            }
        }
        if(l==(r-1)){
            break;
        }
    }
}

__kernel void gen_mats_linear_matrix_scramble_b2(
    const ulong r,
    const ulong d,
    const ulong mmax, 
    const ulong batch_size_r,
    const ulong batch_size_d, 
    const ulong batch_size_mmax,
    const ulong r_C,
    const ulong tmax_new,
    __global const ulong *tmaxes,
    __global const ulong *S,
    __global const ulong *C,
    __global ulong *Cnew)
{
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0 = get_global_id(1)*batch_size_d;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong tmax,b,t,ll,l,jj,j,kk,k,u,v,udotv,vnew,idx;
    ulong bigone = 1;
    ulong nelemC = r_C*d*mmax;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        tmax = tmaxes[l%r_C];
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
            for(kk=0; kk<batch_size_mmax; kk++){
                k = k0+kk;
                idx = l*d*mmax+j*mmax+k;
                v = C[idx%nelemC];
                vnew = 0;
                for(t=0; t<tmax_new; t++){
                    u = S[l*d*tmax_new+j*tmax_new+t];
                    udotv = u&v;
                    // Brian Kernighan algorithm: https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
                    b = 0;
                    while(udotv){
                        b += 1;
                        udotv &= (udotv-1);
                    }
                    if((b%2)==1){
                        vnew += bigone<<(tmax_new-t-1);
                    }
                }
                Cnew[idx] = vnew;
                if(k==(mmax-1)){
                    break;
                }
            }
            if(j==(d-1)){
                break;
            }
        }
        if(l==(r-1)){
            break;
        }
    }
}

__kernel void digital_net_b2_binary(
    const ulong r,
    const ulong n, 
    const ulong d, 
    const ulong batch_size_r_x,
    const ulong batch_size_n, 
    const ulong batch_size_d, 
    const ulong nstart,
    const char gc,
    const ulong mmax,
    __global const ulong *C,
    __global ulong *xb)
{   
    ulong l0 = get_global_id(0)*batch_size_r_x;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong b,t,ll,l,ii,i,jj,j,prev_i,new_i;
    ulong itrue = nstart+i0;
    // initial index 
    t = itrue^(itrue>>1);
    prev_i = gc ? i0*d : (t-nstart)*d;
    // initialize first values 0 
    for(jj=0; jj<batch_size_d; jj++){
        j = j0+jj;
        for(ll=0; ll<batch_size_r_x; ll++){
            l = l0+ll;
            xb[l*n*d+prev_i+j] = 0;
            if(l==(r-1)){
                break;
            }
        }
        if(j==(d-1)){
            break;
        }
    }
    // set first values
    b = 0;
    while(t>0){
        if(t&1){
            for(jj=0; jj<batch_size_d; jj++){
                j = j0+jj;
                for(ll=0; ll<batch_size_r_x; ll++){
                    l = l0+ll;
                    xb[l*n*d+prev_i+j] ^= C[l*d*mmax+j*mmax+b];
                }
            }
        }
        b += 1;
        t >>= 1;
    }
    // set remaining values
    for(ii=1; ii<batch_size_n; ii++){
        i = i0+ii;
        itrue = i+nstart;
        if(gc){
            new_i = i*d;
        }
        else{
            t = itrue^(itrue>>1);
            new_i = (t-nstart)*d;
        }
        b = 0;
        while(!((itrue>>b)&1)){
            b += 1;
        }
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
            for(ll=0; ll<batch_size_r_x; ll++){
                l = l0+ll;
                xb[l*n*d+new_i+j] = xb[l*n*d+prev_i+j]^C[l*d*mmax+j*mmax+b];
                if(l==(r-1)){
                    break;
                }
            }
            if(j==(d-1)){
                break;
            }
        }
        prev_i = new_i;
        if(i==(n-1)){
            break;
        }
    }
}

__kernel void digital_net_b2_binary_rdshift(
    const ulong r,
    const ulong n, 
    const ulong d, 
    const ulong batch_size_r,
    const ulong batch_size_n, 
    const ulong batch_size_d,
    const ulong r_x,
    __global const ulong *lshifts,
    __global const ulong *xb,
    __global const ulong *shiftsb,
    __global ulong *xrb)
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
                xrb[idx] = (xb[(idx)%nelem_x]<<lshifts[l%r_x])^shiftsb[l*d+j];
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

__kernel void digital_net_b2_from_binary(
    const ulong r_or_r_x,
    const ulong n, 
    const ulong d, 
    const ulong batch_size_r_or_r_x,
    const ulong batch_size_n, 
    const ulong batch_size_d,
    __global const ulong *tmaxes,
    __global const ulong *xb,
    __global double *x)
{
    ulong l0 = get_global_id(0)*batch_size_r_or_r_x;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ll,l,ii,i,jj,j,idx;
    for(ll=0; ll<batch_size_r_or_r_x; ll++){
        l = l0+ll;
        for(ii=0; ii<batch_size_n; ii++){
            i = i0+ii;
            for(jj=0; jj<batch_size_d; jj++){
                j = j0+jj;
                idx = l*n*d+i*d+j;
                x[idx] = ldexp((double)(xb[idx]),-tmaxes[l]);
                if(j==(d-1)){
                    break;
                }
            }
            if(i==(n-1)){
                break;
            }
        }
        if(l==(r_or_r_x-1)){
            break;
        }
    }
}

__kernel void interlace_b2(
    const ulong r,
    const ulong d_alpha, 
    const ulong mmax, 
    const ulong batch_size_r,
    const ulong batch_size_d_alpha, 
    const ulong batch_size_mmax,
    const ulong d,
    const ulong tmax,
    const ulong tmax_alpha,
    const ulong alpha,
    __global const ulong *C,
    __global ulong *C_alpha)
{
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0_alpha = get_global_id(1)*batch_size_d_alpha;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong ll,l,jj_alpha,j_alpha,kk,k,t_alpha,t,jj,j,v,b;
    ulong bigone = 1;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(jj_alpha=0; jj_alpha<batch_size_d_alpha; jj_alpha++){
            j_alpha = j0_alpha+jj_alpha;
             for(kk=0; kk<batch_size_mmax; kk++){
                k = k0+kk;
                v = 0;
                for(t_alpha=0; t_alpha<tmax_alpha; t_alpha++){
                    t = t_alpha / alpha; 
                    jj = t_alpha%alpha; 
                    j = j_alpha*alpha+jj;
                    b = (C[l*d*mmax+j*mmax+k]>>(tmax-t-1))&1;
                    if(b){
                        v += (bigone<<(tmax_alpha-t_alpha-1));
                    }
                }
                C_alpha[l*d_alpha*mmax+j_alpha*mmax+k] = v;
                if(k==(mmax-1)){
                    break;
                }
            }
            if(j_alpha==(d_alpha-1)){
                break;
            }
        }
        if(l==(r-1)){
            break;
        }
    }
}

__kernel void undo_interlace_b2(
    const ulong r,
    const ulong d, 
    const ulong mmax, 
    const ulong batch_size_r,
    const ulong batch_size_d, 
    const ulong batch_size_mmax,
    const ulong d_alpha,
    const ulong tmax,
    const ulong tmax_alpha,
    const ulong alpha,
    __global const ulong *C_alpha,
    __global ulong *C)
{
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0 = get_global_id(1)*batch_size_d;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong ll,l,j_alpha,kk,k,t_alpha,tt_alpha,t,jj,j,v,b;
    ulong bigone = 1;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
             for(kk=0; kk<batch_size_mmax; kk++){
                k = k0+kk;
                v = 0;
                for(t=0; t<tmax; t++){
                    j_alpha = j/alpha;
                    tt_alpha = j%alpha;
                    t_alpha = t*alpha+tt_alpha;
                    b = (C_alpha[l*d_alpha*mmax+j_alpha*mmax+k]>>(tmax_alpha-t_alpha-1))&1;
                    if(b){
                        v += (bigone<<(tmax-t-1));
                    }
                }
                C[l*d*mmax+j*mmax+k] = v;
                if(k==(mmax-1)){
                    break;
                }
            }
            if(j==(d-1)){
                break;
            }
        }
        if(l==(r-1)){
            break;
        }
    }
}
