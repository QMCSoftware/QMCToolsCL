__kernel void lat_gen_linear(
    // Lattice points in linear order
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    __global const ulong *g, // pointer to generating vector of size r*d
    __global double *x // pointer to point storage of size r*n*d
){
    ulong l0 = get_global_id(0)*batch_size_r;
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
            for(ll=0; ll<batch_size_r; ll++){
                l = l0+ll;
                x[l*n*d+i*d+j] = (double)(fmod((double)(g[l*d+j]*ifrac),(double)(1.)));
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

__kernel void lat_gen_natural_gray(
    // Lattice points in Gray code or natural order
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong n_start, // starting index in sequence
    const char gc, // flag to use Gray code or natural order
    __global const ulong *g, // pointer to generating vector of size r*d 
    __global double *x // pointer to point storage of size r*n*d
){   
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    double ifrac;
    ulong p,v,itrue,igc,b,ll,l,ii,i,jj,j,idx;
    ulong n0 = n_start+i0;
    if(n0==0){
        p = 0;
        v = 0;
    }
    else{
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
                itrue = i+n_start;
                igc = itrue^(itrue>>1);
                idx = (igc-n_start)*d+j;
            }
            for(ll=0; ll<batch_size_r; ll++){
                l = l0+ll;
                x[l*n*d+idx] = (double)(fmod((double)(g[l*d+j]*ifrac),(double)(1.)));
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

__kernel void lat_shift_mod_1(
    // Shift mod 1 for lattice points
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong r_x, // replications in x
    __global const double *x, // lattice points of size r_x*n*d
    __global const double *shifts, // shifts of size r*d
    __global double *xr // pointer to point storage of size r*n*d
){
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
                xr[idx] = (double)(fmod((double)(x[(idx)%nelem_x]+shifts[l*d+j]),(double)(1.)));
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

__kernel void dnb2_gmat_lsb_to_msb(
    // Convert base 2 generating matrices with integers stored in Least Significant Bit order to Most Significant Bit order
    const ulong r, // replications
    const ulong d, // dimension
    const ulong mmax, // columns in each generating matrix 
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_d, // batch size for dimensions
    const ulong batch_size_mmax, // batch size for columns
    __global const ulong *tmaxes, // length r vector of bits in each integer of the resulting MSB generating matrices
    __global const ulong *C_lsb, // original generating matrices of size r*d*mmax
    __global ulong *C_msb // new generating matrices of size r*d*mmax
){
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
                v = C_lsb[idx];
                vnew = 0;
                t = 0; 
                while(v!=0){
                    if(v&1){
                        vnew += bigone<<(tmax-t-1);
                    }
                    v >>= 1;
                    t += 1;
                }
                C_msb[idx] = vnew;
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

__kernel void dnb2_linear_matrix_scramble(
    // Linear matrix scrambling for base 2 generating matrices
    const ulong r, // replications
    const ulong d, // dimension
    const ulong mmax, // columns in each generating matrix 
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_d, // batch size for dimensions
    const ulong batch_size_mmax, // batch size for columns
    const ulong r_C, // original generating matrices
    const ulong tmax_new, // bits in the integers of the resulting generating matrices
    __global const ulong *S, // scrambling matrices of size r*d*tmax_new
    __global const ulong *C, // original generating matrices of size r_C*d*mmax
    __global ulong *C_lms // resulting generating matrices of size r*d*mmax
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0 = get_global_id(1)*batch_size_d;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong b,t,ll,l,jj,j,kk,k,u,v,udotv,vnew,idx;
    ulong bigone = 1;
    ulong nelemC = r_C*d*mmax;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
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
                C_lms[idx] = vnew;
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

__kernel void dnb2_gen_natural_gray(
    // Binary representation of digital net in base 2 in either Gray code or natural order
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong n_start, // starting index in sequence
    const char gc, // flag to use Gray code or natural order
    const ulong mmax, // columns in each generating matrix
    __global const ulong *C, // generating matrices of size r*d*mmax
    __global ulong *xb // binary digital net points of size r*n*d
){   
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong b,t,ll,l,ii,i,jj,j,prev_i,new_i;
    ulong itrue = n_start+i0;
    // initial index 
    t = itrue^(itrue>>1);
    prev_i = gc ? i0*d : (t-n_start)*d;
    // initialize first values 0 
    for(jj=0; jj<batch_size_d; jj++){
        j = j0+jj;
        for(ll=0; ll<batch_size_r; ll++){
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
                for(ll=0; ll<batch_size_r; ll++){
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
        itrue = i+n_start;
        if(gc){
            new_i = i*d;
        }
        else{
            t = itrue^(itrue>>1);
            new_i = (t-n_start)*d;
        }
        b = 0;
        while(!((itrue>>b)&1)){
            b += 1;
        }
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
            for(ll=0; ll<batch_size_r; ll++){
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

__kernel void dnb2_digital_shift(
    // Digital shift base 2 digital net 
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong r_x, // replications of xb
    __global const ulong *lshifts, // left shift applied to each element of xb
    __global const ulong *xb, // binary base 2 digital net points of size r_x*n*d
    __global const ulong *shiftsb, // digital shifts of size r*d
    __global ulong *xrb // digital shifted digital net points of size r*n*d
){
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

__kernel void dnb2_integer_to_float(
    // Convert base 2 binary digital net points to floats
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    __global const ulong *tmaxes, // bits in integers of each generating matrix of size r
    __global const ulong *xb, // binary digital net points of size r*n*d
    __global double *x // float digital net points of size r*n*d
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ll,l,ii,i,jj,j,idx;
    for(ll=0; ll<batch_size_r; ll++){
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
        if(l==(r-1)){
            break;
        }
    }
}

__kernel void dnb2_interlace(
    // Interlace generating matrices or transpose of point sets to attain higher order digital nets in base 2
    const ulong r, // replications
    const ulong d_alpha, // dimension of resulting generating matrices 
    const ulong mmax, // columns of generating matrices
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_d_alpha, // batch size for dimension of resulting generating matrices
    const ulong batch_size_mmax, // batch size for replications
    const ulong d, // dimension of original generating matrices
    const ulong tmax, // bits in integers of original generating matrices
    const ulong tmax_alpha, // bits in integers of resulting generating matrices
    const ulong alpha, // interlacing factor
    __global const ulong *C, // original generating matrices of size r*d*mmax
    __global ulong *C_alpha // resulting interlaced generating matrices of size r*d_alpha*mmax
){
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

__kernel void dnb2_undo_interlace(
    // Undo interlacing of generating matrices in base 2
    const ulong r, // replications
    const ulong d, // dimension of resulting generating matrices 
    const ulong mmax, // columns in generating matrices
    const ulong batch_size_r, // batch size of replications
    const ulong batch_size_d, // batch size of dimension of resulting generating matrices
    const ulong batch_size_mmax, // batch size of columns in generating matrices
    const ulong d_alpha, // dimension of interlaced generating matrices
    const ulong tmax, // bits in integers of original generating matrices 
    const ulong tmax_alpha, // bits in integers of interlaced generating matrices
    const ulong alpha, // interlacing factor
    __global const ulong *C_alpha, // interlaced generating matrices of size r*d_alpha*mmax
    __global ulong *C // original generating matrices of size r*d*mmax
){
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

__kernel void gdn_linear_matrix_scramble(
    // Linear matrix scramble for generalized digital net 
    const ulong r, // replications 
    const ulong d, // dimension 
    const ulong mmax, // columns in each generating matrix
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_d, // batch size for dimension
    const ulong batch_size_mmax, // batch size columns
    const ulong r_C, // number of replications of C 
    const ulong r_b, // number of replications of bases
    const ulong tmax, // number of rows in each generating matrix 
    const ulong tmax_new, // new number of rows in each generating matrix 
    __global const ulong *bases, // bases for each dimension of size r*d 
    __global const ulong *S, // scramble matrices of size r*d*tmax_new*tmax
    __global const ulong *C, // generating matrices of size r_C*d*mmax*tmax 
    __global ulong *C_lms // new generating matrices of size r*d*mmax*tmax_new
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0 = get_global_id(1)*batch_size_d;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong ll,l,jj,j,kk,k,t,c,b,v,idx_C,idx_C_lms,idx_S; 
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
            b = bases[(l%r_b)*d+j];
            for(kk=0; kk<batch_size_mmax; kk++){
                k = k0+kk;
                idx_C = (l%r_C)*d*mmax*tmax+j*mmax*tmax+k*tmax;
                idx_C_lms = l*d*mmax*tmax_new+j*mmax*tmax_new+k*tmax_new;
                for(t=0; t<tmax_new; t++){
                    v = 0;
                    idx_S = l*d*tmax_new*tmax+j*tmax_new*tmax+t*tmax;
                    for(c=0; c<tmax; c++){
                        v += (S[idx_S+c]*C[idx_C+c])%b;
                    }
                    C_lms[idx_C_lms+t] = v;
                }
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

__kernel void gdn_gen_natural(
    // Generalized digital net where the base can be different for each dimension e.g. for the Halton sequence
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong r_b, // number of replications of bases
    const ulong mmax, // columns in each generating matrix
    const ulong tmax, // rows of each generating matrix
    const ulong n_start, // starting index in sequence
    __global const ulong *bases, // bases for each dimension of size r_b*d
    __global const ulong *C, // generating matrices of size r*d*mmax*tmax
    __global ulong *xdig // generalized digital net sequence of digits of size r*n*d*tmax
){   
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong idx_xdig,idx_C,b,dig,itrue,icp,ii,i,jj,j,ll,l,t,k;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(ii=0; ii<batch_size_n; ii++){
            i = i0+ii;
            itrue = i+n_start;
            for(jj=0; jj<batch_size_d; jj++){
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

__kernel void gdn_gen_natural_same_base(
    // Generalized digital net with the same base for each dimension e.g. a digital net in base greater than 2
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong mmax, // columns in each generating matrix
    const ulong tmax, // rows of each generating matrix
    const ulong n_start, // starting index in sequence
    const ulong b, // common base
    __global const ulong *C, // generating matrices of size r*d*mmax*tmax
    __global ulong *xdig // generalized digital net sequence of digits of size r*n*d*tmax
){   
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong idx_xdig,idx_C,dig,itrue,icp,ii,i,jj,j,ll,l,t,k;
    // initialize xdig everything to 0
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(ii=0; ii<batch_size_n; ii++){
            i = i0+ii;
            for(jj=0; jj<batch_size_d; jj++){
                j = j0+jj;
                idx_xdig = l*n*d*tmax+i*d*tmax+j*tmax;
                for(t=0; t<tmax; t++){
                    xdig[idx_xdig+t] = 0;
                }
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
    // now set the points
    for(ii=0; ii<batch_size_n; ii++){
        i = i0+ii;
        itrue = i+n_start;
        k = 0;
        icp = itrue; 
        while(icp>0){
            dig = icp%b;
            icp = (icp-dig)/b;
            if(dig>0){
                for(ll=0; ll<batch_size_r; ll++){
                    l = l0+ll;
                    for(jj=0; jj<batch_size_d; jj++){
                        j = j0+jj;
                        idx_xdig = l*n*d*tmax+i*d*tmax+j*tmax;
                        idx_C = l*d*mmax*tmax+j*mmax*tmax+k*tmax;
                        for(t=0; t<tmax; t++){
                            xdig[idx_xdig+t] = (xdig[idx_xdig+t]+dig*C[idx_C+t])%b;
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
            k += 1;
        }
        if(i==(n-1)){
            break;
        }
    }
}
                
__kernel void gdn_digital_shift(
    // Digital shift a generalized digital net
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
    __global const ulong *bases, // bases for each dimension of size r_b*d
    __global const ulong *shifts, // digital shifts of size r*d*tmax_new
    __global const ulong *xdig, // binary digital net points of size r_x*n*d*tmax
    __global ulong *xdig_new // float digital net points of size r*n*d*tmax_new
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong b,ll,l,ii,i,jj,j,t,idx_xdig,idx_xdig_new,idx_shift;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(ii=0; ii<batch_size_n; ii++){
            i = i0+ii;
            for(jj=0; jj<batch_size_d; jj++){
                j = j0+jj;
                idx_xdig = (l%r_x)*n*d*tmax+i*d*tmax+j*tmax;
                idx_xdig_new = l*n*d*tmax_new+i*d*tmax_new+j*tmax_new;
                idx_shift = l*d*tmax_new+j*tmax_new;
                b = bases[(l%r_b)*d+j];
                for(t=0; t<tmax; t++){
                    xdig_new[idx_xdig_new+t] = (xdig[idx_xdig+t]+shifts[idx_shift+t])%b;
                }
                for(t=tmax; t<tmax_new; t++){
                    xdig_new[idx_xdig_new+t] = shifts[idx_shift+t];
                }
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
    ulong ll,l,ii,i,jj,j,t,idx_xdig,idx_xdig_new,idx_perm,p;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(ii=0; ii<batch_size_n; ii++){
            i = i0+ii;
            for(jj=0; jj<batch_size_d; jj++){
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

__kernel void gdn_integer_to_float(
    // Convert digits of generalized digital net to floats
    const ulong r, // replications
    const ulong n, // points
    const ulong d, // dimension
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_n, // batch size for points
    const ulong batch_size_d, // batch size for dimension
    const ulong r_b, // replications of bases 
    const ulong tmax, // rows of each generating matrix
    __global const ulong *bases, // bases for each dimension of size r_b*d
    __global const ulong *xdig, // binary digital net points of size r*n*d*tmax
    __global double *x // float digital net points of size r*n*d
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong i0 = get_global_id(1)*batch_size_n;
    ulong j0 = get_global_id(2)*batch_size_d;
    ulong ll,l,ii,i,jj,j,t,idx_xdig;
    double recip,v,xdig_double,b;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(ii=0; ii<batch_size_n; ii++){
            i = i0+ii;
            for(jj=0; jj<batch_size_d; jj++){
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

__kernel void gdn_interlace(
    // Interlace generating matrices or transpose of point sets to attain higher order digital nets
    const ulong r, // replications
    const ulong d_alpha, // dimension of resulting generating matrices 
    const ulong mmax, // columns of generating matrices
    const ulong batch_size_r, // batch size for replications
    const ulong batch_size_d_alpha, // batch size for dimension of resulting generating matrices
    const ulong batch_size_mmax, // batch size for replications
    const ulong d, // dimension of original generating matrices
    const ulong tmax, // rows of original generating matrices
    const ulong tmax_alpha, // rows of interlaced generating matrices
    const ulong alpha, // interlacing factor
    __global const ulong *C, // original generating matrices of size r*d*mmax*tmax
    __global ulong *C_alpha // resulting interlaced generating matrices of size r*d_alpha*mmax*tmax_alpha
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0_alpha = get_global_id(1)*batch_size_d_alpha;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong ll,l,jj_alpha,j_alpha,kk,k,t_alpha,t,jj,j;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(jj_alpha=0; jj_alpha<batch_size_d_alpha; jj_alpha++){
            j_alpha = j0_alpha+jj_alpha;
             for(kk=0; kk<batch_size_mmax; kk++){
                k = k0+kk;
                for(t_alpha=0; t_alpha<tmax_alpha; t_alpha++){
                    t = t_alpha / alpha; 
                    jj = t_alpha%alpha; 
                    j = j_alpha*alpha+jj;
                    C_alpha[l*d_alpha*mmax*tmax_alpha+j_alpha*mmax*tmax_alpha+k*tmax_alpha+t_alpha] = C[l*d*mmax*tmax+j*mmax*tmax+k*tmax+t];
                }
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

__kernel void gdn_undo_interlace(
    // Undo interlacing of generating matrices 
    const ulong r, // replications
    const ulong d, // dimension of resulting generating matrices 
    const ulong mmax, // columns in generating matrices
    const ulong batch_size_r, // batch size of replications
    const ulong batch_size_d, // batch size of dimension of resulting generating matrices
    const ulong batch_size_mmax, // batch size of columns in generating matrices
    const ulong d_alpha, // dimension of interlaced generating matrices
    const ulong tmax, // rows of original generating matrices
    const ulong tmax_alpha, // rows of interlaced generating matrices
    const ulong alpha, // interlacing factor
    __global const ulong *C_alpha, // interlaced generating matrices of size r*d_alpha*mmax*tmax_alpha
    __global ulong *C // original generating matrices of size r*d*mmax*tmax
){
    ulong l0 = get_global_id(0)*batch_size_r;
    ulong j0 = get_global_id(1)*batch_size_d;
    ulong k0 = get_global_id(2)*batch_size_mmax;
    ulong ll,l,j_alpha,kk,k,t_alpha,tt_alpha,t,jj,j;
    for(ll=0; ll<batch_size_r; ll++){
        l = l0+ll;
        for(jj=0; jj<batch_size_d; jj++){
            j = j0+jj;
             for(kk=0; kk<batch_size_mmax; kk++){
                k = k0+kk;
                for(t=0; t<tmax; t++){
                    j_alpha = j/alpha;
                    tt_alpha = j%alpha;
                    t_alpha = t*alpha+tt_alpha;
                    C[l*d*mmax*tmax+j*mmax*tmax+k*tmax+t] = C_alpha[l*d_alpha*mmax*tmax_alpha+j_alpha*mmax*tmax_alpha+k*tmax_alpha+t_alpha];
                }
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

__kernel void fwht_1d_b2(
    // In place, vectorized 1 dimensional Fast Walsh-Hadamard Transform where the size of the last dimension is a power of 2
    const ulong d1, // first dimenion
    const ulong d2, // second dimension
    const ulong n_half, // half of the last dimenion along which FWHT is performed
    const ulong batch_size_d1, // batch size first dimension 
    const ulong batch_size_d2, // batch size second dimension
    const ulong batch_size_n_half, // batch size for half of the last dimension
    __global double *x // array of size d1*d2*2n_half on which to perform FWHT in place
){
    ulong j10 = get_global_id(0)*batch_size_d1;
    ulong j20 = get_global_id(1)*batch_size_d2;
    ulong i0 = get_global_id(2)*batch_size_n_half;
    ulong ii,i,i1,i2,jj1,jj2,j1,j2,k,s,f,idx;
    double x1,x2;
    ulong n = 2*n_half;
    ulong m = (ulong)(log2((double)n));
    for(k=0; k<m; k++){
        s = m-k-1;
        f = 1<<s; 
        for(ii=0; ii<batch_size_n_half; ii++){
            i = i0+ii;
            if((i>>s)&1){
                i2 = i+n_half;
                i1 = i2^f;
            }
            else{
                i1 = i;
                i2 = i1^f;
            }
            for(jj1=0; jj1<batch_size_d1; jj1++){
                j1 = j10+jj1;
                for(jj2=0; jj2<batch_size_d2; jj2++){
                    j2 = j20+jj2;
                    idx = j1*d2*n+j2*n;
                    x1 = x[idx+i1];
                    x2 = x[idx+i2];
                    x[idx+i1] = x1+x2;
                    x[idx+i2] = x1-x2;
                    if(j2==(d2-1)){
                        break;
                    }
                }
                if(j1==(d1-1)){
                    break;
                }
            }
            if(i==(n_half-1)){
                break;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void fft_1d_b2(
    // In place, vectorized 1 dimensional Fast Fourier Transform where the size of the last dimension is a power of 2
    const ulong d1, // first dimenion
    const ulong d2, // second dimension
    const ulong n_half, // half of the last dimenion along which FFT is performed
    const ulong batch_size_d1, // batch size first dimension 
    const ulong batch_size_d2, // batch size second dimension
    const ulong batch_size_n_half, // batch size for half of the last dimension
    __global double *xr, // real array of size d1*d2*2n_half on which to perform FFT in place
    __global double *xi // imaginary array of size d1*d2*2n_half on which to perform FFT in place
){
    ulong j10 = get_global_id(0)*batch_size_d1;
    ulong j20 = get_global_id(1)*batch_size_d2;
    ulong i0 = get_global_id(2)*batch_size_n_half;
    ulong ii,i,i1,i2,t,jj1,jj2,j1,j2,k,s,f,idx;
    double xr1,xr2,xi1,xi2,yr,yi,v,cosv,sinv;
    ulong n = 2*n_half;
    ulong m = (ulong)(log2((double)n)); 
    for(k=0; k<m; k++){
        s = m-k-1;
        f = 1<<s; 
        printf("k = %lu\n",k);
        for(ii=0; ii<batch_size_n_half; ii++){
            i = i0+ii;
            if((i>>s)&1){
                i2 = i+n_half;
                i1 = i2^f;
            }
            else{
                i1 = i;
                i2 = i1^f;
            }
            t = i1%f;
            for(jj1=0; jj1<batch_size_d1; jj1++){
                j1 = j10+jj1;
                for(jj2=0; jj2<batch_size_d2; jj2++){
                    j2 = j20+jj2;
                    idx = j1*d2*n+j2*n;
                    xr1 = xr[idx+i1];
                    xr2 = xr[idx+i2];
                    xi1 = xi[idx+i1];
                    xi2 = xi[idx+i2];
                    xr[idx+i1] = xr1+xr2;
                    xi[idx+i1] = xi1+xi2;
                    yr = xr1-xr2;
                    yi = xi1-xi2;
                    v = 2*M_PI*t/(2*f);
                    cosv = cos(v);
                    sinv = sin(v);
                    printf("\ti1 = %lu\ti2 = %lu\tW_%lu^%lu\tcosv = %lf\tsinv = %lf\n",i1,i2,2*f,t,cosv,sinv);
                    xr[idx+i2] = yr*cosv-yi*sinv;
                    xi[idx+i2] = yr*sinv+yi*cosv;
                    // can disregard most of the bottom of the graph for real valued inputs 
                    if(j2==(d2-1)){
                        break;
                    }
                }
                if(j1==(d1-1)){
                    break;
                }
            }
            if(i==(n_half-1)){
                break;
            }
        }
        printf("\n");
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
