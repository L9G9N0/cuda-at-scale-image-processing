#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
struct Image { int width=0,height=0; std::vector<uint8_t> pixels; };
__global__ void BoxBlurRgb(const uint8_t* in,uint8_t* out,int w,int h) {
 int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=w||y>=h) return;
 for(int c=0;c<3;++c) { int sum=0,n=0;
  for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
   int xx=max(0,min(w-1,x+dx)), yy=max(0,min(h-1,y+dy));
   sum+=in[(yy*w+xx)*3+c]; ++n;
  }
  out[(y*w+x)*3+c]=static_cast<uint8_t>(sum/n);
 }
}
Image ReadPpm(const fs::path& p) {
 std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("Cannot open "+p.string());
 std::string magic; f>>magic; if(magic!="P6") throw std::runtime_error("Expected P6 PPM");
 Image im; int maxv=0; f>>im.width>>im.height>>maxv; f.get();
 if(im.width<1||im.height<1||maxv!=255) throw std::runtime_error("Invalid PPM header");
 size_t bytes=static_cast<size_t>(im.width)*im.height*3; im.pixels.resize(bytes);
 f.read(reinterpret_cast<char*>(im.pixels.data()),bytes); if(!f) throw std::runtime_error("Incomplete PPM");
 return im;
}
void WritePpm(const fs::path& p,const Image& im) {
 std::ofstream f(p,std::ios::binary); if(!f) throw std::runtime_error("Cannot write "+p.string());
 f<<"P6\n"<<im.width<<" "<<im.height<<"\n255\n";
 f.write(reinterpret_cast<const char*>(im.pixels.data()),im.pixels.size());
}
int main(int argc,char** argv) {
 try {
  fs::path input="data/input",output="data/output"; int iterations=3;
  for(int i=1;i<argc;++i) { std::string a=argv[i];
   if(a=="--input-dir"&&i+1<argc) input=argv[++i];
   else if(a=="--output-dir"&&i+1<argc) output=argv[++i];
   else if(a=="--iterations"&&i+1<argc) iterations=std::stoi(argv[++i]);
   else if(a=="--help") { std::cout<<"--input-dir DIR --output-dir DIR --iterations N\n"; return 0; }
   else throw std::runtime_error("Unknown/incomplete argument: "+a);
  }
  if(iterations<1) throw std::runtime_error("iterations must be positive");
  int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices)); if(!devices) throw std::runtime_error("No CUDA GPU");
  fs::create_directories(output); std::vector<fs::path> paths;
  for(const auto& e:fs::directory_iterator(input)) if(e.is_regular_file()&&e.path().extension()==".ppm") paths.push_back(e.path());
  std::sort(paths.begin(),paths.end()); if(paths.empty()) throw std::runtime_error("No PPM files in "+input.string());
  std::ofstream csv(output/"results.csv"); csv<<"image,width,height,iterations,avg_kernel_ms\n";
  double total=0; size_t pixels=0;
  std::cout<<"CUDA devices: "<<devices<<"\nProcessing "<<paths.size()<<" images.\n";
  for(const auto& p:paths) {
   Image im=ReadPpm(p); size_t bytes=im.pixels.size(); uint8_t *di=nullptr,*doo=nullptr;
   CUDA_CHECK(cudaMalloc(&di,bytes)); CUDA_CHECK(cudaMalloc(&doo,bytes));
   CUDA_CHECK(cudaMemcpy(di,im.pixels.data(),bytes,cudaMemcpyHostToDevice));
   dim3 block(16,16),grid((im.width+15)/16,(im.height+15)/16);
   cudaEvent_t start,stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
   float sum=0;
   for(int k=0;k<iterations;++k) {
    CUDA_CHECK(cudaEventRecord(start)); BoxBlurRgb<<<grid,block>>>(di,doo,im.width,im.height);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop)); sum+=ms;
   }
   Image result{im.width,im.height,std::vector<uint8_t>(bytes)};
   CUDA_CHECK(cudaMemcpy(result.pixels.data(),doo,bytes,cudaMemcpyDeviceToHost)); WritePpm(output/p.filename(),result);
   csv<<p.filename().string()<<","<<im.width<<","<<im.height<<","<<iterations<<","<<std::fixed<<std::setprecision(6)<<sum/iterations<<"\n";
   total+=sum; pixels+=static_cast<size_t>(im.width)*im.height;
   std::cout<<"Processed "<<p.filename()<<" "<<im.width<<"x"<<im.height<<" avg_kernel_ms="<<sum/iterations<<"\n";
   CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop)); CUDA_CHECK(cudaFree(di)); CUDA_CHECK(cudaFree(doo));
  }
  std::cout<<"Completed images: "<<paths.size()<<"\nTotal pixels processed: "<<pixels<<"\nTotal kernel time (ms): "<<total<<"\n";
  return 0;
 } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }
}
