#include <tl_templates/musa/reduce.h>
#include <cstdio>
using tl::scalar_max_nan;
using tl::scalar_min_nan;
__global__ void edges(unsigned short *out){
 const unsigned short a[]={0x7e12,0x3c00,0x8000,0,0xfc00,0x7c00,0x4000};
 const unsigned short b[]={0x3c00,0x7e12,0,0x8000,0x7c00,0xfc00,0x3c00};
 int i=threadIdx.x;
 auto x=half_t(__ushort_as_half(a[i])),y=half_t(__ushort_as_half(b[i]));
 out[i*4]=__half_as_ushort(scalar_max_nan(x,y).to_half());
 out[i*4+1]=__half_as_ushort(scalar_min_nan(x,y).to_half());
 out[i*4+2]=__half_as_ushort(tl::MaxOpNan{}(x,y).to_half());
 out[i*4+3]=__half_as_ushort(tl::MinOpNan{}(x,y).to_half());
}
int main(){
 unsigned short expected[][2]={{0x7e00,0x7e00},{0x7e00,0x7e00},{0,0x8000},{0,0x8000},{0x7c00,0xfc00},{0x7c00,0xfc00},{0x4000,0x3c00}};
 unsigned short h[28],*d;
 if(musaMalloc(&d,sizeof(h))!=musaSuccess)return 2;
 edges<<<1,7>>>(d);
 if(musaMemcpy(h,d,sizeof(h),musaMemcpyDeviceToHost)!=musaSuccess)return 3;
 int errors=0;for(int i=0;i<7;++i)for(int j=0;j<4;++j)if(h[4*i+j]!=expected[i][j%2]){printf("%d %d %04x vs %04x\n",i,j,h[4*i+j],expected[i][j%2]);++errors;}
 printf("half edge checks 28: %d errors\n",errors);return errors?1:0;
}
