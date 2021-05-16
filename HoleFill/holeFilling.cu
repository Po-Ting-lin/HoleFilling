#include "holeFilling.cuh"
#include <math.h>

__global__ void NotKernel(bool* src, bool* dst, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    dst[i] = !src[i];
}

__global__ void NaiveDilationAndKernel(bool* src, bool* dst, bool* ref, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height || x >= width) return;
    bool value = 0;
    unsigned int up_y = max(y - 1, 0);
    unsigned int down_y = min(height - 1, y + 1);
    unsigned int left_x = max(x - 1, 0);
    unsigned int right_x = min(width - 1, x + 1);
    value = src[up_y * width + x] | value;
    value = src[down_y * width + x] | value;
    value = src[y * width + left_x] | value;
    value = src[y * width + right_x] | value;
    value = src[y * width + x] | value;
    dst[y * width + x] = value & ref[y * width + x];
}

__global__ void NaiveDilationAndSmemKernel(bool* src, bool* dst, bool* ref, int width, int height, int tileSize) {
    extern __shared__ bool smem[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    smem[ty * tileSize + tx] = false;
    __syncthreads();
    if (y >= height || x >= width) return;
    smem[ty * tileSize + tx] = src[y * width + x];
    __syncthreads();

    unsigned int up_y = max(ty - 1, 0);
    unsigned int down_y = min(tileSize - 1, ty + 1);
    unsigned int left_x = max(tx - 1, 0);
    unsigned int right_x = min(tileSize - 1, tx + 1);
    bool value = false;
    value = smem[up_y * tileSize + tx] | value;
    value = smem[down_y * tileSize + tx] | value;
    value = smem[ty * tileSize + left_x] | value;
    value = smem[ty * tileSize + right_x] | value;
    dst[y * width + x] = value & ref[y * width + x];
}

__global__ void NotAndOrKernel(bool* src, bool* andRef, bool* orRef, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    src[i] = !src[i] && andRef[i] || orRef[i];
}

__global__ void IsEqualReductionKernel(bool* src, bool* ref, bool* isEqualReduced, int size) {
    __shared__ bool smem[BLOCK_DIM * BLOCK_DIM];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smem[threadIdx.x] = true;
    __syncthreads();

    if (i >= size) return;
    smem[threadIdx.x] = src[i] == ref[i];
    __syncthreads();

    reduction(smem, isEqualReduced);
}

__global__ void IsEqualReductionKernel(bool* isEqual, bool* isEqualReduced, int size){
    __shared__ bool smem[BLOCK_DIM * BLOCK_DIM];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    smem[threadIdx.x] = true;
    __syncthreads();

    if (i >= size) return;
    smem[threadIdx.x] = isEqual[i];
    __syncthreads();

    reduction(smem, isEqualReduced);
}

__device__ void reduction(bool* smem, bool* dst) {
    for (int offset = blockDim.x / 2; offset >= 64; offset /= 2) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] = smem[threadIdx.x] && smem[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        volatile bool* vsmem = smem;
        vsmem[threadIdx.x] = vsmem[threadIdx.x] && vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] = vsmem[threadIdx.x] && vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] = vsmem[threadIdx.x] && vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] = vsmem[threadIdx.x] && vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] = vsmem[threadIdx.x] && vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] = vsmem[threadIdx.x] && vsmem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0) {
        dst[blockIdx.x] = smem[0];
    }
}

void HoleFilling::borderImageInit(cv::Mat& src, cv::Mat& dst) {
    int imageSize = src.rows * src.cols;
    uchar* dstPtr = dst.data;
    uchar* srcPtr = src.data;
    int y, x;

    for (int p = 0; p < imageSize; p++, dstPtr++, srcPtr++) {
        // is Border?
        y = p / src.cols;
        x = p % src.cols;
        if (y == 0 || y == src.rows - 1 || x == 0 || x == src.cols - 1) {
            *dstPtr = CV_8U_DYNAMICRANGE - 1 - *srcPtr;
        }
        else {
            *dstPtr = 0;
        }
    }
}

void HoleFilling::speedBorderImageInit(cv::Mat& src, cv::Mat& dst) {
    src.copyTo(dst);
    int imageSize = src.rows * src.cols;
    uchar* preMarkerPtr = dst.data;
    bool* visitedArr = new bool[imageSize];

#pragma omp parallel for
    for (int i = 0; i < imageSize; i++) {
        visitedArr[i] = false;
    }

    // left to right
#pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int i = y * src.cols + x;
            // black and not visited
            if (!preMarkerPtr[i] && !visitedArr[i]) {
                preMarkerPtr[i] = 255;
                visitedArr[i] = true;
            }
            else break;
        }
    }

    // right to left
#pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = src.cols - 1; x >= 0; x--) {
            int i = y * src.cols + x;
            // black and not visited
            if (!preMarkerPtr[i] && !visitedArr[i]) {
                preMarkerPtr[i] = 255;
                visitedArr[i] = true;
            }
            else break;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < imageSize; i++) {
        if (!visitedArr[i]) preMarkerPtr[i] = 0;
    }
    delete[] visitedArr;
}

void HoleFilling::Process(cv::Mat& src, cv::Mat& dst) {
    cv::Mat preMarker(src.rows, src.cols, CV_8U, cv::Scalar(0));
    speedBorderImageInit(src, preMarker);
    fillprocess(src, preMarker, dst);
}

void HoleFilling::fillprocess(cv::Mat& src, cv::Mat& premarker, cv::Mat& dst) {
    int width = src.cols;
    int height = src.rows;
    int imageSize = width * height;
    bool* h_src_image = (bool*)malloc(imageSize * sizeof(bool));
    bool* h_premarker = (bool*)malloc(imageSize * sizeof(bool));
    bool* h_dst_image = (bool*)malloc(imageSize * sizeof(bool));

#pragma omp parallel for
    for (int i = 0; i < imageSize; i++) {
        h_src_image[i] = src.data[i] == 0 ? false : true;
        h_premarker[i] = premarker.data[i] == 0 ? false : true;
    }

    bool* d_src_image;
    bool* d_c_image;
    bool* d_image;
    bool* d_temp_image;
    bool* d_reduced_is_equal;
    int tileSize1 = BLOCK_DIM;
    int tileSize2 = 31;
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM));
    dim3 block2(tileSize2, tileSize2);
    dim3 grid2(iDivUp(width, tileSize2), iDivUp(height, tileSize2));
    dim3 block1D(BLOCK_DIM * BLOCK_DIM);
    dim3 grid1D(iDivUp(imageSize, BLOCK_DIM * BLOCK_DIM));

    gpuErrorCheck(cudaMalloc((void**)&d_src_image, imageSize * sizeof(bool)));
    gpuErrorCheck(cudaMalloc((void**)&d_c_image, imageSize * sizeof(bool)));
    gpuErrorCheck(cudaMalloc((void**)&d_image, imageSize * sizeof(bool)));
    gpuErrorCheck(cudaMalloc((void**)&d_temp_image, imageSize * sizeof(bool)));
    gpuErrorCheck(cudaMalloc((void**)&d_reduced_is_equal, grid1D.x * sizeof(bool)));

    gpuErrorCheck(cudaMemset(d_reduced_is_equal, 0, grid1D.x * sizeof(bool)));
    gpuErrorCheck(cudaMemcpy(d_src_image, h_src_image, imageSize * sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_image, h_premarker, imageSize * sizeof(bool), cudaMemcpyHostToDevice));

    NotKernel << <grid1D, block1D >> > (d_src_image, d_c_image, imageSize);
    gpuErrorCheck(cudaDeviceSynchronize());
    bool is_equal = false;

    for (int i = 0; i < maxIters; i++) {
        //printf("iter: %d\n", i);
        NaiveDilationAndKernel << <grid, block>> > (d_image, d_temp_image, d_c_image, width, height);
        //if (i % 2 == 0) NaiveDilationAndSmemKernel << <grid, block, tileSize1 * tileSize1 * sizeof(bool) >> > (d_image, d_temp_image, d_c_image, width, height, tileSize1);
        //else NaiveDilationAndSmemKernel << <grid2, block2, tileSize2* tileSize2 * sizeof(bool) >> > (d_image, d_temp_image, d_c_image, width, height, tileSize2);
        IsEqualReductionKernel << <grid1D, block1D >> > (d_temp_image, d_image, d_reduced_is_equal, imageSize);
        IsEqualReductionKernel << <1, block1D >> > (d_reduced_is_equal, d_reduced_is_equal, grid1D.x);
        gpuErrorCheck(cudaMemcpy(&is_equal, d_reduced_is_equal, sizeof(bool), cudaMemcpyDeviceToHost));
        if (is_equal) {/*printf("count: %d\n", i)*/; break;}
        else {gpuErrorCheck(cudaMemcpy(d_image, d_temp_image, imageSize * sizeof(bool), cudaMemcpyDeviceToDevice));}
    }
    NotAndOrKernel << <grid1D, block1D >> > (d_temp_image, d_c_image, d_src_image, imageSize);
    gpuErrorCheck(cudaMemcpy(h_dst_image, d_temp_image, imageSize * sizeof(bool), cudaMemcpyDeviceToHost));

#pragma omp parallel for
    for (int i = 0; i < imageSize; i++) {
        dst.data[i] = h_dst_image[i] == 0 ? 0 : 255;
    }

    cudaFree(d_src_image);
    cudaFree(d_c_image);
    cudaFree(d_image);
    cudaFree(d_temp_image);
    cudaFree(d_reduced_is_equal);
    free(h_src_image);
    free(h_premarker);
}

void warmupGPU() {
    cudaFree(0);
}