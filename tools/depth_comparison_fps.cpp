// Exact scalar double-precision version of anchor_fps, for isolated CPU analysis.
// No fast-math or fused multiply-add: preserve deterministic nearest/tie rules.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
extern "C" void depth_fps(const double* p, int64_t n, int64_t count,
                          const int64_t* initial, int64_t ninitial, int64_t* out) {
    if (!count || !n) return;
    std::vector<double> nearest(n, std::numeric_limits<double>::infinity());
    std::vector<bool> used(n, false);
    double mean[3] = {0, 0, 0};
    for (int64_t i=0;i<n;++i) for(int k=0;k<3;++k) mean[k]+=p[3*i+k];
    for(int k=0;k<3;++k) mean[k]/=n;
    auto squared=[&](int64_t i, const double* q) {
        double x=p[3*i]-q[0], y=p[3*i+1]-q[1], z=p[3*i+2]-q[2];
        return (x*x+y*y)+z*z;
    };
    int64_t index=0; double best=std::numeric_limits<double>::infinity();
    for(int64_t i=0;i<n;++i) {double d=squared(i,mean); if(d<best){best=d;index=i;}}
    for(int64_t k=0;k<ninitial;++k){
        for(int64_t i=0;i<n;++i) nearest[i]=std::min(nearest[i],squared(i,p+3*initial[k]));
        used[initial[k]]=true;
    }
    auto next=[&]() {
        int64_t j=0; double d=-std::numeric_limits<double>::infinity();
        for(int64_t i=0;i<n;++i){if(used[i]) nearest[i]=-std::numeric_limits<double>::infinity();
            if(nearest[i]>d){d=nearest[i];j=i;}}
        return j;
    };
    if(ninitial) index=next();
    for(int64_t k=0;k<count;++k){
        out[k]=index;used[index]=true;
        for(int64_t i=0;i<n;++i) nearest[i]=std::min(nearest[i],squared(i,p+3*index));
        index=next();
    }
}
