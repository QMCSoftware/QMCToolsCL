#include "qmctoolscl.h"

EXPORT void dnb2_integer_to_float(
    // Convert base 2 binary digital net points to floats
    const unsigned long long r, // replications
    const unsigned long long n, // points
    const unsigned long long d, // dimension
    const unsigned long long batch_size_r, // batch size for replications
    const unsigned long long batch_size_n, // batch size for points
    const unsigned long long batch_size_d, // batch size for dimension
    const unsigned long long *tmaxes, // bits in integers of each generating matrix of size r
    const unsigned long long *xb, // binary digital net points of size r*n*d
    double *x // float digital net points of size r*n*d
){
    unsigned long long l0 = 0*batch_size_r;
    unsigned long long i0 = 0*batch_size_n;
    unsigned long long j0 = 0*batch_size_d;
    unsigned long long ii_max = (n-i0)<batch_size_n ? (n-i0):batch_size_n;
    unsigned long long jj_max = (d-j0)<batch_size_d ? (d-j0):batch_size_d;
    unsigned long long ll_max = (r-l0)<batch_size_r ? (r-l0):batch_size_r;
    unsigned long long ll,l,ii,i,jj,j,idx;
    for(ll=0; ll<ll_max; ll++){
        l = l0+ll;
        for(ii=0; ii<ii_max; ii++){
            i = i0+ii;
            for(jj=0; jj<jj_max; jj++){
                j = j0+jj;
                idx = l*n*d+i*d+j;
                x[idx] = ldexp((double)(xb[idx]),-tmaxes[l]);
            }
        }
    }
}