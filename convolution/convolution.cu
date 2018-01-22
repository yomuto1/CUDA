#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cuda.h"
#include "cublas_v2.h"

#define DEBUG_WRITING (0)

#define WID_SRC (768)
#define HEI_SRC (576)
#define CHN_SRC (3)
#define WID_L00 (608)
#define HEI_L00 (608)
#define CHN_L00 (32)
#define K_W_L00 (3)
#define K_H_L00 (3)
#define PAD_L00 (1)

#define BLOCK (512)

#if (1 == DEBUG_WRITING)
FILE *fp_fprintf_debug;
#endif

static float *cuda_make_array(float *x, size_t n);
static void check_error(cudaError_t status);
static void fill_gpu(int N, float ALPHA, float * X, int INCX);
static dim3 cuda_gridsize(size_t n);
static void im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);
static void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc);
static cublasHandle_t blas_handle();
static int cuda_get_device();

int main(void)
{
    FILE *fp;
    static float sa_image_sized_f32[WID_L00 * HEI_L00 * CHN_SRC];
    static float sa_out_l00_f32[WID_L00 * HEI_L00 * CHN_L00];
    float *p_out_l00_f32;
    static float sa_weights_l00_f32[K_W_L00 * K_H_L00 * CHN_SRC * CHN_L00];
    float *p_weights_gpu_l00_f32;
    static float sa_mean_l00_f32[CHN_L00];
    float *p_mean_gpu_l00_f32;
    static float sa_variance_l00_f32[CHN_L00];
    float *p_variance_gpu_l00_f32;
    static float sa_scale_l00_f32[CHN_L00];
    float *p_scale_gpu_l00_f32;
    static float sa_bias_l00_f32[CHN_L00];
    float *p_bias_gpu_l00_f32;
    float *p_input_gpu_f32;
    float *p_workspace_f32;
    static float sa_ref_l00_f32[WID_L00 * HEI_L00 * CHN_L00];
    int i, j, k;
    size_t fread_return;
    clock_t clk_srt, clk_end;

    printf("yolo reference CUDA code by Hyuk Lee\n");

    memset(sa_out_l00_f32, 0, WID_L00 * HEI_L00 * CHN_L00 * sizeof(float));

#if (1 == DEBUG_WRITING)
    fp_fprintf_debug = fopen("ref_c_debug.txt", "w");
#endif

    /* read input data (letterbox_image currently) */
    fp = fopen("yolo_image_sized.bin", "rb");
    if(NULL == fp)
    {
        printf("read input data fopen error\n");
        return -1;
    }
    fread_return = fread(sa_image_sized_f32, WID_L00 * HEI_L00 * CHN_SRC, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load weights */
    fp = fopen("yolo_cpu_weights_b_0_g_0_3x3x3x32.bin", "rb");
    if(NULL == fp)
    {
        printf("read weights fopen error\n");
        return -1;
    }
    fread_return = fread(sa_weights_l00_f32, K_W_L00 * K_H_L00 * CHN_SRC * CHN_L00, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load mean */
    fp = fopen("yolo_cpu_rolling_mean_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read mean fopen error\n");
        return -1;
    }
    fread_return = fread(sa_mean_l00_f32, CHN_L00, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load variance */
    fp = fopen("yolo_cpu_rolling_variance_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read variance fopen error\n");
        return -1;
    }
    fread_return = fread(sa_variance_l00_f32, CHN_L00, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load scale */
    fp = fopen("yolo_cpu_scales_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read scale fopen error\n");
        return -1;
    }
    fread_return = fread(sa_scale_l00_f32, CHN_L00, sizeof(float), fp);
    fclose(fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    /* load bias */
    fp = fopen("yolo_cpu_biases_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read bias fopen error\n");
        return -1;
    }
    fread_return = fread(sa_bias_l00_f32, CHN_L00, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);

    p_input_gpu_f32 = cuda_make_array(sa_image_sized_f32, WID_L00 * HEI_L00 * CHN_SRC);
    p_out_l00_f32 = cuda_make_array(sa_out_l00_f32, WID_L00 * HEI_L00 * CHN_L00);
    p_weights_gpu_l00_f32 = cuda_make_array(sa_weights_l00_f32, K_W_L00 * K_H_L00 * CHN_SRC * CHN_L00);
    p_mean_gpu_l00_f32 = cuda_make_array(sa_mean_l00_f32, CHN_L00);
    p_variance_gpu_l00_f32 = cuda_make_array(sa_variance_l00_f32, CHN_L00);
    p_scale_gpu_l00_f32 = cuda_make_array(sa_scale_l00_f32, CHN_L00);
    p_bias_gpu_l00_f32 = cuda_make_array(sa_bias_l00_f32, CHN_L00);
    p_workspace_f32 = cuda_make_array(sa_out_l00_f32, WID_L00 * HEI_L00 * CHN_L00);

    clk_srt = clock();
    fill_gpu(WID_L00 * HEI_L00 * CHN_L00, 0, p_out_l00_f32, 1);
#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else
    int m = CHN_L00;
    k = K_W_L00 * K_H_L00 * CHN_SRC;
    int n = WID_L00 * HEI_L00;
    for(i = 0; i < 1; ++i){
        for(j = 0; j < 1; ++j){
            float *a = p_weights_gpu_l00_f32 + j * K_W_L00 * K_H_L00 * CHN_SRC * CHN_L00;
            float *b = p_workspace_f32;
            float *c = p_out_l00_f32 + (i + j)*n*m;

            im2col_gpu(p_input_gpu_f32 + (i + j)*CHN_SRC/HEI_L00*WID_L00,
                CHN_SRC, HEI_L00, WID_L00, K_W_L00, 1, PAD_L00, b);
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
#endif
#if 0
    forward_batchnorm_layer_gpu(l, net);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
#endif
    clk_end = clock();
    printf("l00 convolution: %f secs\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);

#if (1 == DEBUG_WRITING)
    fclose(fp_fprintf_debug);
#endif

    cudaMemcpy(sa_out_l00_f32, p_out_l00_f32, WID_L00 * HEI_L00 * CHN_L00 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(p_input_gpu_f32);
    cudaFree(p_out_l00_f32);
    cudaFree(p_weights_gpu_l00_f32);
    cudaFree(p_mean_gpu_l00_f32);
    cudaFree(p_variance_gpu_l00_f32);
    cudaFree(p_scale_gpu_l00_f32);
    cudaFree(p_bias_gpu_l00_f32);
    cudaFree(p_workspace_f32);

    /* read ref data layer 0 */
    fp = fopen("yolo_convolution_out_ref_c_608x608x32.bin", "rb");
    if(NULL == fp)
    {
        printf("read ref data l00 fopen error\n");
        return -1;
    }
    fread_return = fread(sa_ref_l00_f32, WID_L00 * HEI_L00 * CHN_L00, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);

    for(k = 0; k < CHN_L00; k++)
    {
        for(j = 0 + PAD_L00; j < HEI_L00 - PAD_L00; j++)
        {
            for(i = 0 + PAD_L00; i < WID_L00 - PAD_L00; i++)
            {
                if(fabsf(sa_out_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00] - sa_ref_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00]) > 0.000001f)
                {
                    printf("layer_0_f32 mismatch: w %d, h %d, c %d, out %f, GT %f\n", i, j, k, sa_out_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00], sa_ref_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00]);
                }
            }
        }
    }

    return 0;
}

static float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) printf("Cuda malloc failed\n");
    return x_gpu;
}

static void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        printf("CUDA Error: %s\n", s);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        printf("CUDA Error Prev: %s\n", s);
    } 
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

static void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

static dim3 cuda_gridsize(size_t n)
{
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //*data_col_ptr = data_im_ptr[ii * width + jj];

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

static void im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}

static void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
}

static cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

static int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

