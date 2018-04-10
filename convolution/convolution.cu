#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "cuda.h"
#include "cublas_v2.h"

#define DEBUG_WRITING (0)

#define WID_SRC (768)
#define HEI_SRC (576)
#define CHN_SRC (3)
#define WID_SIZED (608)
#define HEI_SIZED (608)
#define WID_DST (19)
#define HEI_DST (19)
#define CHN_DST (425)

#define SIZE_MAX_WORKSPACE (30000000)
#define NUM_LAYER (32)

#define BLOCK (512)

#define CHK_INTER_LAYER (0)
#define ACCEPTABLE_DIFF (0.005f)

#if (1 == DEBUG_WRITING)
FILE *fp_fprintf_debug;
#endif

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;

static int sa_typ_s32[NUM_LAYER];
static int sa_wid_s32[NUM_LAYER];
static int sa_hei_s32[NUM_LAYER];
static int sa_chn_s32[NUM_LAYER];
static int sa_ker_s32[NUM_LAYER];
static int sa_pad_s32[NUM_LAYER];
static int sa_ibn_s32[NUM_LAYER];
static int sa_nwe_s32[NUM_LAYER];
static unsigned char sa_image_in_u08[WID_SRC * HEI_SRC * CHN_SRC];
#if (1 == CHK_INTER_LAYER)
static float sa_image_sized_f32[WID_SIZED * HEI_SIZED * CHN_SRC];
#endif
static float sa_tmp_buf_f32[SIZE_MAX_WORKSPACE];
static float sa_out_f32[WID_DST * HEI_DST * CHN_DST];
static float *spa_out_f32[NUM_LAYER];
static float *spa_weights_f32[NUM_LAYER];
static float *spa_mean_f32[NUM_LAYER];
static float *spa_variance_f32[NUM_LAYER];
static float *spa_scales_f32[NUM_LAYER];
static float *spa_biases_f32[NUM_LAYER];
static float *sp_gpu_out_f32[NUM_LAYER];
static float *sp_gpu_weights_f32[NUM_LAYER];
static float *sp_gpu_mean_f32[NUM_LAYER];
static float *sp_gpu_variance_f32[NUM_LAYER];
static float *sp_gpu_scales_f32[NUM_LAYER];
static float *sp_gpu_biases_f32[NUM_LAYER];
static float *sp_gpu_input_f32;
static float *sp_gpu_workspace_f32;
static float sa_ref_sized_f32[WID_SIZED * HEI_SIZED * CHN_SRC];
static float *spa_ref_f32[NUM_LAYER];

static void yolo_main(float *p_out_f32, unsigned char *p_image_in_u08);
static float *cuda_make_array(float *x, size_t n);
static void check_error(cudaError_t status);
static void fill_gpu(int N, float ALPHA, float * X, int INCX);
static dim3 cuda_gridsize(size_t n);
static void im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);
static void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc);
static void forward_convolutional_layer_gpu(float *l_output_gpu, float *input_gpu, float *l_weights_gpu, float *workspace_gpu, float *mean_gpu, float *variance_gpu, float *scales_gpu, float *biases_gpu, int l_outputs, int l_n, int l_size, int l_c, int l_out_w, int l_out_h, int l_w, int l_h, int l_stride, int l_pad, int l_batch_normalize, ACTIVATION l_activation);
static cublasHandle_t blas_handle();
static int cuda_get_device();
static void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
static void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
static void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
static void activate_array_gpu(float *x, int n, ACTIVATION a);
static void forward_maxpool_layer_gpu(float *l_output_gpu, float *input_gpu, int layer_out_w, int layer_out_h, int layer_batch, int layer_w, int layer_h, int layer_c, int layer_stride, int layer_size, int layer_pad);
static void forward_route_layer_25_gpu(float *l_output_gpu, float *input_l16);
static void forward_route_layer_28_gpu(float *l_output_gpu, float *input_l27, float *input_l24);
static void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
static void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
static void forward_reorg_layer_gpu(float *l_output_gpu, float *input_gpu, int l_w, int l_h, int l_c, int l_batch, int l_stride);
static void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);
static void forward_region_layer_gpu(float *l_output_gpu, float *input_gpu, float *l_output, int l_batch, int l_inputs, int l_n, int l_w, int l_h, int l_coords, int l_background, int l_classes, int l_outputs);
static int entry_index(int l_w, int l_h, int l_outputs, int l_coords, int l_classes, int batch, int location, int entry);
static void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
static void cuda_pull_array(float *x_gpu, float *x, size_t n);
#if (1 == CHK_INTER_LAYER)
static void check_intermediate_layer_results(int l);
#endif

static unsigned char *sp_gpu_image_in_u08;
static float *sp_gpu_image_in_f32;
static float *sp_gpu_resized_f32;
static float *sp_gpu_part_f32;

int main(void)
{
    FILE *fp;
    FILE *fp_results;
    FILE *fp_weights;
    FILE *fp_mean;
    FILE *fp_variance;
    FILE *fp_scales;
    FILE *fp_biases;
    FILE *fp_netinfo;
    int i, j, k;
    size_t fread_return;
    clock_t clk_srt, clk_end;

    printf("yolo reference CUDA code by Hyuk Lee\n");

#if (1 == DEBUG_WRITING)
    fp_fprintf_debug = fopen("ref_c_debug.txt", "w");
#endif

    /* read net info */
    fp_netinfo = fopen("yolo_gpu_netinfo.bin", "rb");
    if(NULL == fp_netinfo)
    {
        printf("yolo_gpu_netinfo fopen error\n");
        return -1;
    }
    /* load weights */
    fp_weights = fopen("yolo_gpu_weights.bin", "rb");
    if(NULL == fp_weights)
    {
        printf("yolo_gpu_weights fopen error\n");
        return -1;
    }
    fp_mean = fopen("yolo_gpu_mean.bin", "rb");
    if(NULL == fp_mean)
    {
        printf("yolo_gpu_mean fopen error\n");
        return -1;
    }
    fp_variance = fopen("yolo_gpu_variance.bin", "rb");
    if(NULL == fp_variance)
    {
        printf("yolo_gpu_variance fopen error\n");
        return -1;
    }
    fp_scales = fopen("yolo_gpu_scales.bin", "rb");
    if(NULL == fp_scales)
    {
        printf("yolo_gpu_scales fopen error\n");
        return -1;
    }
    fp_biases = fopen("yolo_gpu_biases.bin", "rb");
    if(NULL == fp_biases)
    {
        printf("yolo_gpu_biases fopen error\n");
        return -1;
    }
    /* load ref data */
    fp = fopen("yolo_image_sized.bin", "rb");
    if(NULL == fp)
    {
        printf("yolo_image_sized fopen error\n");
        return -1;
    }
    fread_return = fread(sa_ref_sized_f32, WID_SIZED * HEI_SIZED * CHN_SRC, sizeof(float), fp);
    fclose(fp);
    fp_results = fopen("yolo_gpu_intermediate_results.bin", "rb");
    if(NULL == fp_results)
    {
        printf("yolo_gpu_results fopen error\n");
        return -1;
    }
    for(i = 0; i < NUM_LAYER; i++)
    {
        fread_return = fread(&sa_typ_s32[i], 1, sizeof(int), fp_netinfo);        
        fread_return = fread(&sa_wid_s32[i], 1, sizeof(int), fp_netinfo);        
        fread_return = fread(&sa_hei_s32[i], 1, sizeof(int), fp_netinfo);        
        fread_return = fread(&sa_chn_s32[i], 1, sizeof(int), fp_netinfo);        
        fread_return = fread(&sa_ker_s32[i], 1, sizeof(int), fp_netinfo);        
        fread_return = fread(&sa_pad_s32[i], 1, sizeof(int), fp_netinfo);        
        fread_return = fread(&sa_ibn_s32[i], 1, sizeof(int), fp_netinfo);        
        fread_return = fread(&sa_nwe_s32[i], 1, sizeof(int), fp_netinfo);        
        spa_out_f32[i] = (float *)malloc(sa_wid_s32[i] * sa_hei_s32[i] * sa_chn_s32[i] * sizeof(float));
        sp_gpu_out_f32[i] = cuda_make_array(spa_out_f32[i], sa_wid_s32[i] * sa_hei_s32[i] * sa_chn_s32[i]);
        if(sa_typ_s32[i] == 0) /* convolutional */
        {
            spa_weights_f32[i] = (float *)malloc(sa_nwe_s32[i] * sizeof(float));
            fread_return = fread(spa_weights_f32[i], sa_nwe_s32[i], sizeof(float), fp_weights);
            sp_gpu_weights_f32[i] = cuda_make_array(spa_weights_f32[i], sa_nwe_s32[i]);
            if(sa_ibn_s32[i] == 1)
            {
                spa_mean_f32[i] = (float *)malloc(sa_chn_s32[i] * sizeof(float));
                spa_variance_f32[i] = (float *)malloc(sa_chn_s32[i] * sizeof(float));
                spa_scales_f32[i] = (float *)malloc(sa_chn_s32[i] * sizeof(float));
                fread_return = fread(spa_mean_f32[i], sa_chn_s32[i], sizeof(float), fp_mean);
                fread_return = fread(spa_variance_f32[i], sa_chn_s32[i], sizeof(float), fp_variance);
                fread_return = fread(spa_scales_f32[i], sa_chn_s32[i], sizeof(float), fp_scales);
                sp_gpu_mean_f32[i] = cuda_make_array(spa_mean_f32[i], sa_chn_s32[i]);
                sp_gpu_variance_f32[i] = cuda_make_array(spa_variance_f32[i], sa_chn_s32[i]);
                sp_gpu_scales_f32[i] = cuda_make_array(spa_scales_f32[i], sa_chn_s32[i]);
            }
            spa_biases_f32[i] = (float *)malloc(sa_chn_s32[i] * sizeof(float));
            fread_return = fread(spa_biases_f32[i], sa_chn_s32[i], sizeof(float), fp_biases);
            sp_gpu_biases_f32[i] = cuda_make_array(spa_biases_f32[i], sa_chn_s32[i]);
        }
        spa_ref_f32[i] = (float *)malloc(sa_wid_s32[i] * sa_hei_s32[i] * sa_chn_s32[i] * sizeof(float));
        fread_return = fread(spa_ref_f32[i], sa_wid_s32[i] * sa_hei_s32[i] * sa_chn_s32[i], sizeof(float), fp_results);
    }
    fclose(fp_results);
    fclose(fp_weights);
    fclose(fp_mean);
    fclose(fp_variance);
    fclose(fp_scales);
    fclose(fp_biases);
    fclose(fp_netinfo);

    /* read input data */
    fp = fopen("yolo_image_in.bin", "rb");
    if(NULL == fp)
    {
        printf("yolo_image_in fopen error\n");
        return -1;
    }
    fread_return = fread(sa_image_in_u08, WID_SRC * HEI_SRC * CHN_SRC, sizeof(unsigned char), fp);
    fclose(fp);

    sp_gpu_workspace_f32 = cuda_make_array(sa_tmp_buf_f32, SIZE_MAX_WORKSPACE);
    sp_gpu_input_f32 = cuda_make_array(sa_tmp_buf_f32, WID_SIZED * HEI_SIZED * CHN_SRC);
    cudaMalloc((void **)&sp_gpu_image_in_u08, WID_SRC * HEI_SRC * CHN_SRC * sizeof(unsigned char));
    cudaMalloc((void **)&sp_gpu_image_in_f32, WID_SRC * HEI_SRC * CHN_SRC * sizeof(float));
    cudaMalloc((void **)&sp_gpu_resized_f32, 608 * 456 * 3 * sizeof(float));
    cudaMalloc((void **)&sp_gpu_part_f32, 608 * 576 * 3 * sizeof(float));

    clk_srt = clock();
    yolo_main(sa_out_f32, sa_image_in_u08);
    clk_end = clock();
    printf("yolo 1: %f s\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);

    for(k = 0; k < CHN_DST; k++)
    {
        for(j = 0; j < HEI_DST; j++)
        {
            for(i = 0; i < WID_DST; i++)
            {
                if(fabsf(sa_out_f32[i + j * WID_DST + k * WID_DST * HEI_DST] - spa_ref_f32[NUM_LAYER - 1][i + j * WID_DST + k * WID_DST * HEI_DST]) > ACCEPTABLE_DIFF)
                {
                    printf("final results mismatch: w %d, h %d, c %d, out %f, GT %f\n", i, j, k, sa_out_f32[i + j * WID_DST + k * WID_DST * HEI_DST], spa_ref_f32[NUM_LAYER - 1][i + j * WID_DST + k * WID_DST * HEI_DST]);
                }
            }
        }
    }

#if (1 == CHK_INTER_LAYER)
    for(i = 0; i < NUM_LAYER - 1; i++)
    {
        check_intermediate_layer_results(i);
    }
#endif

    clk_srt = clock();
    yolo_main(sa_out_f32, sa_image_in_u08);
    clk_end = clock();
    printf("yolo 2: %f s\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);

    for(k = 0; k < CHN_DST; k++)
    {
        for(j = 0; j < HEI_DST; j++)
        {
            for(i = 0; i < WID_DST; i++)
            {
                if(fabsf(sa_out_f32[i + j * WID_DST + k * WID_DST * HEI_DST] - spa_ref_f32[NUM_LAYER - 1][i + j * WID_DST + k * WID_DST * HEI_DST]) > ACCEPTABLE_DIFF)
                {
                    printf("final results mismatch: w %d, h %d, c %d, out %f, GT %f\n", i, j, k, sa_out_f32[i + j * WID_DST + k * WID_DST * HEI_DST], spa_ref_f32[NUM_LAYER - 1][i + j * WID_DST + k * WID_DST * HEI_DST]);
                }
            }
        }
    }

#if (1 == DEBUG_WRITING)
    fclose(fp_fprintf_debug);
#endif

    if(0 == fread_return)
    {
        printf("problem on fread\n");
    }

    for(i = 0; i < NUM_LAYER; i++)
    {
        free(spa_out_f32[i]);
        cudaFree(sp_gpu_out_f32[i]);
        if(sa_typ_s32[i] == 0) /* convolutional */
        {
            free(spa_weights_f32[i]);
            cudaFree(sp_gpu_weights_f32[i]);
            if(sa_ibn_s32[i] == 1)
            {
                free(spa_mean_f32[i]);
                free(spa_variance_f32[i]);
                free(spa_scales_f32[i]);
                cudaFree(sp_gpu_mean_f32[i]);
                cudaFree(sp_gpu_variance_f32[i]);
                cudaFree(sp_gpu_scales_f32[i]);
            }
            free(spa_biases_f32[i]);
            cudaFree(sp_gpu_biases_f32[i]);
        }
        free(spa_ref_f32[i]);
    }

    cudaFree(sp_gpu_input_f32);
    cudaFree(sp_gpu_workspace_f32);
    cudaFree(sp_gpu_image_in_u08);
    cudaFree(sp_gpu_image_in_f32);
    cudaFree(sp_gpu_resized_f32);
    cudaFree(sp_gpu_part_f32);

    return 0;
}

__global__ void u08_to_f32_3ch_kernel(float *p_out_f32, unsigned char *p_src_u08, int wid_s32, int hei_s32)
{
    int threadIdx_x_s32 = threadIdx.x;
    int threadIdx_y_s32 = threadIdx.y;
    int iCount_s32 = blockIdx.x * blockDim.x + threadIdx_x_s32;
    int jCount_s32 = blockIdx.y * blockDim.y + threadIdx_y_s32;

    if( (iCount_s32 < wid_s32) && (jCount_s32 < hei_s32) )
    {
        p_out_f32[iCount_s32 + jCount_s32 * wid_s32] = p_src_u08[iCount_s32 + jCount_s32 * wid_s32] / 255.f;
        p_out_f32[iCount_s32 + jCount_s32 * wid_s32 + wid_s32 * hei_s32] = p_src_u08[iCount_s32 + jCount_s32 * wid_s32 + wid_s32 * hei_s32] / 255.f;
        p_out_f32[iCount_s32 + jCount_s32 * wid_s32 + wid_s32 * hei_s32 * 2] = p_src_u08[iCount_s32 + jCount_s32 * wid_s32 + wid_s32 * hei_s32 * 2] / 255.f;
    }
}

#define W_SCALE (1.263591f)
#define H_SCALE (1.263736f)

__global__ void resize_image_1_kernel(float *part, float *im)
{
    int threadIdx_x_s32 = threadIdx.x;
    int threadIdx_y_s32 = threadIdx.y;
    int iCount_s32 = blockIdx.x * blockDim.x + threadIdx_x_s32;
    int jCount_s32 = blockIdx.y * blockDim.y + threadIdx_y_s32;

    if( (iCount_s32 < WID_SIZED) && (jCount_s32 < 576) )
    {
        float sx = iCount_s32 * W_SCALE;
        int ix = (int) sx;
        float dx = sx - ix;
        float val;
        val = (1 - dx) * im[0 * HEI_SRC * WID_SRC + jCount_s32 * WID_SRC + ix] + dx * im[0 * HEI_SRC * WID_SRC + jCount_s32 * WID_SRC + ix + 1];
        part[0 * WID_SIZED * HEI_SRC + jCount_s32 * WID_SIZED + iCount_s32] = val;
        val = (1 - dx) * im[1 * HEI_SRC * WID_SRC + jCount_s32 * WID_SRC + ix] + dx * im[1 * HEI_SRC * WID_SRC + jCount_s32 * WID_SRC + ix + 1];
        part[1 * WID_SIZED * HEI_SRC + jCount_s32 * WID_SIZED + iCount_s32] = val;
        val = (1 - dx) * im[2 * HEI_SRC * WID_SRC + jCount_s32 * WID_SRC + ix] + dx * im[2 * HEI_SRC * WID_SRC + jCount_s32 * WID_SRC + ix + 1];
        part[2 * WID_SIZED * HEI_SRC + jCount_s32 * WID_SIZED + iCount_s32] = val;
    }
}

__global__ void resize_image_2_kernel(float *resized, float *part)
{
    int threadIdx_x_s32 = threadIdx.x;
    int threadIdx_y_s32 = threadIdx.y;
    int iCount_s32 = blockIdx.x * blockDim.x + threadIdx_x_s32;
    int jCount_s32 = blockIdx.y * blockDim.y + threadIdx_y_s32;

    if( (iCount_s32 < WID_SIZED) && (jCount_s32 < 456) )
    {
        float sy = jCount_s32 * H_SCALE;
        int iy = (int) sy;
        float dy = sy - iy;
        float val;
        val = (1-dy) * part[0 * 608 * 576 + iy * 608 + iCount_s32];
        resized[0 * 608 * 456 + jCount_s32 * 608 + iCount_s32] = val;
        val = dy * part[0 * 608 * 576 + (iy + 1) * 608 + iCount_s32];
        resized[0 * 608 * 456 + jCount_s32 * 608 + iCount_s32] += val;
        val = (1-dy) * part[1 * 608 * 576 + iy * 608 + iCount_s32];
        resized[1 * 608 * 456 + jCount_s32 * 608 + iCount_s32] = val;
        val = dy * part[1 * 608 * 576 + (iy + 1) * 608 + iCount_s32];
        resized[1 * 608 * 456 + jCount_s32 * 608 + iCount_s32] += val;
        val = (1-dy) * part[2 * 608 * 576 + iy * 608 + iCount_s32];
        resized[2 * 608 * 456 + jCount_s32 * 608 + iCount_s32] = val;
        val = dy * part[2 * 608 * 576 + (iy + 1) * 608 + iCount_s32];
        resized[2 * 608 * 456 + jCount_s32 * 608 + iCount_s32] += val;
    }
}

__global__ void fill_image_kernel(float *boxed)
{
    int threadIdx_x_s32 = threadIdx.x;
    int threadIdx_y_s32 = threadIdx.y;
    int iCount_s32 = blockIdx.x * blockDim.x + threadIdx_x_s32;
    int jCount_s32 = blockIdx.y * blockDim.y + threadIdx_y_s32;

    if( (iCount_s32 < WID_SIZED) && (jCount_s32 < HEI_SIZED) )
    {
        boxed[iCount_s32 + jCount_s32 * WID_SIZED] = 0.5f;
        boxed[iCount_s32 + jCount_s32 * WID_SIZED + WID_SIZED * HEI_SIZED] = 0.5f;
        boxed[iCount_s32 + jCount_s32 * WID_SIZED + WID_SIZED * HEI_SIZED * 2] = 0.5f;
    }
}

__global__ void embed_image_kernel(float *resized, float *boxed, int dx, int dy)
{
    int threadIdx_x_s32 = threadIdx.x;
    int threadIdx_y_s32 = threadIdx.y;
    int iCount_s32 = blockIdx.x * blockDim.x + threadIdx_x_s32;
    int jCount_s32 = blockIdx.y * blockDim.y + threadIdx_y_s32;

    if( (iCount_s32 < WID_SIZED) && (jCount_s32 < 456) )
    {
        float val;
        val = resized[0 * 608 * 456 + jCount_s32 * 608 + iCount_s32];
        boxed[0 * 608 * 608 + (dy + jCount_s32) * 608 + (dx + iCount_s32)] = val;
        val = resized[1 * 608 * 456 + jCount_s32 * 608 + iCount_s32];
        boxed[1 * 608 * 608 + (dy + jCount_s32) * 608 + (dx + iCount_s32)] = val;
        val = resized[2 * 608 * 456 + jCount_s32 * 608 + iCount_s32];
        boxed[2 * 608 * 608 + (dy + jCount_s32) * 608 + (dx + iCount_s32)] = val;
    }
}

static void yolo_main(float *p_out_f32, unsigned char *p_image_in_u08)
{
#if (1 == CHK_INTER_LAYER)
    int i, j, k;
#endif
    int l;
    dim3 grid_img_resize_0( 16, 16 );
    dim3 grid_numblocks_resize_0( WID_SRC / grid_img_resize_0.x, HEI_SRC / grid_img_resize_0.y );
    dim3 grid_img_resize_1( 16, 16 );
    dim3 grid_numblocks_resize_1( WID_SIZED / grid_img_resize_1.x, HEI_SRC / grid_img_resize_1.y );
    dim3 grid_img_resize_2( 16, 8 );
    dim3 grid_numblocks_resize_2( WID_SIZED / grid_img_resize_2.x, 456 / grid_img_resize_2.y );
    dim3 grid_img_resize_3( 16, 16 );
    dim3 grid_numblocks_resize_3( WID_SIZED / grid_img_resize_3.x, HEI_SIZED / grid_img_resize_3.y );
    dim3 grid_img_resize_4( 16, 8 );
    dim3 grid_numblocks_resize_4( WID_SIZED / grid_img_resize_4.x, 456 / grid_img_resize_4.y );

    cudaMemcpy(sp_gpu_image_in_u08, p_image_in_u08, WID_SRC * HEI_SRC * CHN_SRC * sizeof(unsigned char), cudaMemcpyHostToDevice);

    u08_to_f32_3ch_kernel<<<grid_numblocks_resize_0, grid_img_resize_0>>>(sp_gpu_image_in_f32, sp_gpu_image_in_u08, WID_SRC, HEI_SRC);
    resize_image_1_kernel<<<grid_numblocks_resize_1, grid_img_resize_1>>>(sp_gpu_part_f32, sp_gpu_image_in_f32);
    resize_image_2_kernel<<<grid_numblocks_resize_2, grid_img_resize_2>>>(sp_gpu_resized_f32, sp_gpu_part_f32);
    fill_image_kernel<<<grid_numblocks_resize_3, grid_img_resize_3>>>(sp_gpu_input_f32);
    embed_image_kernel<<<grid_numblocks_resize_4, grid_img_resize_4>>>(sp_gpu_resized_f32, sp_gpu_input_f32, 0, 76);
    check_error(cudaPeekAtLastError());

#if (1 == CHK_INTER_LAYER)
    cudaMemcpy(sa_image_sized_f32, sp_gpu_input_f32, WID_SIZED * HEI_SIZED * CHN_SRC * sizeof(float), cudaMemcpyDeviceToHost);

    for(k = 0; k < CHN_SRC; k++)
    {
        for(j = 0; j < HEI_SIZED; j++)
        {
            for(i = 0; i < WID_SIZED; i++)
            {
                if(fabsf(sa_image_sized_f32[i + j * WID_SIZED + k * WID_SIZED * HEI_SIZED] - sa_ref_sized_f32[i + j * WID_SIZED + k * WID_SIZED * HEI_SIZED]) > ACCEPTABLE_DIFF)
                {
                    printf("resize mismatch: w %d, h %d, c %d, out %f, GT %f\n", i, j, k, sa_image_sized_f32[i + j * WID_SIZED + k * WID_SIZED * HEI_SIZED], sa_ref_sized_f32[i + j * WID_SIZED + k * WID_SIZED * HEI_SIZED]);
                }
            }
        }
    }
#endif

    l = 0;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_input_f32, sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], CHN_SRC, sa_wid_s32[l], sa_hei_s32[l], WID_SIZED, HEI_SIZED, 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 1;
    forward_maxpool_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sa_wid_s32[l], sa_hei_s32[l], 1, sa_wid_s32[l - 1], sa_hei_s32[l - 1], sa_chn_s32[l], 2, 2, 0);

    l = 2;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 3;
    forward_maxpool_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sa_wid_s32[l], sa_hei_s32[l], 1, sa_wid_s32[l - 1], sa_hei_s32[l - 1], sa_chn_s32[l], 2, 2, 0);


    l = 4;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 5;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 6;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 7;
    forward_maxpool_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sa_wid_s32[l], sa_hei_s32[l], 1, sa_wid_s32[l - 1], sa_hei_s32[l - 1], sa_chn_s32[l], 2, 2, 0);

    l = 8;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 9;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 10;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 11;
    forward_maxpool_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sa_wid_s32[l], sa_hei_s32[l], 1, sa_wid_s32[l - 1], sa_hei_s32[l - 1], sa_chn_s32[l], 2, 2, 0);

    l = 12;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 13;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 14;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 15;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 16;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 17;
    forward_maxpool_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sa_wid_s32[l], sa_hei_s32[l], 1, sa_wid_s32[l - 1], sa_hei_s32[l - 1], sa_chn_s32[l], 2, 2, 0);

    l = 18;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 19;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 20;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 21;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 22;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 23;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 24;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 25;
    forward_route_layer_25_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[16]);

    l = 26;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 27;
    forward_reorg_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sa_wid_s32[l - 1], sa_hei_s32[l - 1], sa_chn_s32[l - 1], 1, 2);

    l = 28;
    forward_route_layer_28_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[27], sp_gpu_out_f32[24]);

    l = 29;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LEAKY);

    l = 30;
    forward_convolutional_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], sp_gpu_weights_f32[l], sp_gpu_workspace_f32, sp_gpu_mean_f32[l], sp_gpu_variance_f32[l], sp_gpu_scales_f32[l], sp_gpu_biases_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l], sa_chn_s32[l], sa_ker_s32[l], sa_chn_s32[l - 1], sa_wid_s32[l], sa_hei_s32[l], sa_wid_s32[l - 1], sa_hei_s32[l - 1], 1, sa_pad_s32[l], sa_ibn_s32[l], LINEAR);

    l = 31;
    forward_region_layer_gpu(sp_gpu_out_f32[l], sp_gpu_out_f32[l - 1], spa_out_f32[l], 1, sa_wid_s32[l - 1] * sa_hei_s32[l - 1] * sa_chn_s32[l - 1], 5, sa_wid_s32[l - 1], sa_hei_s32[l - 1], 4, 0, 80, sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l]);

    cudaMemcpy(p_out_f32, sp_gpu_out_f32[l], WID_DST * HEI_DST * CHN_DST * sizeof(float), cudaMemcpyDeviceToHost);
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

static void forward_convolutional_layer_gpu(float *l_output_gpu, float *input_gpu, float *l_weights_gpu, float *workspace_gpu, float *mean_gpu, float *variance_gpu, float *scales_gpu, float *biases_gpu, int l_outputs, int l_n, int l_size, int l_c, int l_out_w, int l_out_h, int l_w, int l_h, int l_stride, int l_pad, int l_batch_normalize, ACTIVATION l_activation)
{
    fill_gpu(l_outputs, 0, l_output_gpu, 1);

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l_srcTensorDesc,
                net.input_gpu,
                l_weightDesc,
                l_weights_gpu,
                l_convDesc,
                l_fw_algo,
                net.workspace,
                l_workspace_size,
                &one,
                l_dstTensorDesc,
                l_output_gpu);

#else
    int m = l_n;
    int k = l_size*l_size*l_c;
    int n = l_out_w*l_out_h;
    float *a = l_weights_gpu;
    float *b = workspace_gpu;
    float *c = l_output_gpu;

    im2col_gpu(input_gpu,
        l_c, l_h, l_w, l_size, l_stride, l_pad, b);
    gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
#endif

    if (l_batch_normalize) {
        normalize_gpu(l_output_gpu, mean_gpu, variance_gpu, 1, l_n, l_out_w * l_out_h);
        scale_bias_gpu(l_output_gpu, scales_gpu, 1, l_n, l_out_w * l_out_h);
        add_bias_gpu(l_output_gpu, biases_gpu, 1, l_n, l_out_w * l_out_h);
    } else {
        add_bias_gpu(l_output_gpu, biases_gpu, 1, l_n, l_out_w * l_out_h);
    }

    activate_array_gpu(l_output_gpu, l_outputs, l_activation);
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

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    x[index] = (x[index] - mean[f])/(sqrtf(variance[f] + .00001f));
}

static void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

static void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

static void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}

__device__ float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}
__device__ float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

__device__ float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
__device__ float linear_activate_kernel(float x){return x;}
__device__ float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
__device__ float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}
__device__ float relu_activate_kernel(float x){return x*(x>0);}
__device__ float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
__device__ float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
__device__ float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
__device__ float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}
__device__ float tanh_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}
__device__ float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
__device__ float stair_activate_kernel(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floorf(x/2);
    else return (x - n) + floorf(x/2);
}

__device__ float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

__global__ void activate_array_kernel(float *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_kernel(x[i], a);
}

static void activate_array_gpu(float *x, int n, ACTIVATION a) 
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
    check_error(cudaPeekAtLastError());
}

__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output)
{
    int h = (in_h + 2*pad)/stride;
    int w = (in_w + 2*pad)/stride;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad;
    int h_offset = -pad;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
}

static void forward_maxpool_layer_gpu(float *l_output_gpu, float *input_gpu, int layer_out_w, int layer_out_h, int layer_batch, int layer_w, int layer_h, int layer_c, int layer_stride, int layer_size, int layer_pad)
{
    int h = layer_out_h;
    int w = layer_out_w;
    int c = layer_c;

    size_t n = h*w*c*layer_batch;

    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer_h, layer_w, layer_c, layer_stride, layer_size, layer_pad, input_gpu, l_output_gpu);
    check_error(cudaPeekAtLastError());
}

static void forward_route_layer_25_gpu(float *l_output_gpu, float *input_l16)
{
    int offset = 0;
    float *input = input_l16;
    int input_size = 739328;
    copy_gpu(input_size, input, 1, l_output_gpu + offset, 1);
}

static void forward_route_layer_28_gpu(float *l_output_gpu, float *input_l27, float *input_l24)
{
    int offset = 0;
    float *input = input_l27;
    int input_size = 92416;
    copy_gpu(input_size, input, 1, l_output_gpu + offset, 1);
    offset += input_size;
    input = input_l24;
    input_size = 369664;
    copy_gpu(input_size, input, 1, l_output_gpu + offset, 1);
}

static void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

static void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

static void forward_reorg_layer_gpu(float *l_output_gpu, float *input_gpu, int l_w, int l_h, int l_c, int l_batch, int l_stride)
{
    reorg_gpu(input_gpu, l_w, l_h, l_c, l_batch, l_stride, 0, l_output_gpu);
}

__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
}

static void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int size = w*h*c*batch;
    reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, x, w, h, c, batch, stride, forward, out);
    check_error(cudaPeekAtLastError());
}

static void forward_region_layer_gpu(float *l_output_gpu, float *input_gpu, float *l_output, int l_batch, int l_inputs, int l_n, int l_w, int l_h, int l_coords, int l_background, int l_classes, int l_outputs)
{
    copy_gpu(l_batch*l_inputs, input_gpu, 1, l_output_gpu, 1);
    int b, n;
    for (b = 0; b < l_batch; ++b){
        for(n = 0; n < l_n; ++n){
            int index = entry_index(l_w, l_h, l_outputs, l_coords, l_classes, b, n*l_w*l_h, 0);
            activate_array_gpu(l_output_gpu + index, 2*l_w*l_h, LOGISTIC);
            index = entry_index(l_w, l_h, l_outputs, l_coords, l_classes, b, n*l_w*l_h, l_coords);
            activate_array_gpu(l_output_gpu + index,   l_w*l_h, LOGISTIC);
            index = entry_index(l_w, l_h, l_outputs, l_coords, l_classes, b, n*l_w*l_h, l_coords + 1);
        }
    }
    {
        int index = entry_index(l_w, l_h, l_outputs, l_coords, l_classes, 0, 0, l_coords + !l_background);
        softmax_gpu(input_gpu + index, l_classes + l_background, l_batch*l_n, l_inputs/l_n, l_w*l_h, 1, l_w*l_h, 1, l_output_gpu + index);
    }
    {
        cuda_pull_array(l_output_gpu, l_output, l_batch*l_outputs);
    }
}

static int entry_index(int l_w, int l_h, int l_outputs, int l_coords, int l_classes, int batch, int location, int entry)
{
    int n =   location / (l_w*l_h);
    int loc = location % (l_w*l_h);
    return batch*l_outputs + n*l_w*l_h*(l_coords+l_classes+1) + entry*l_w*l_h + loc;
}

__device__ void softmax_device(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = expf(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

static void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
    check_error(cudaPeekAtLastError());
}

static void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

#if (1 == CHK_INTER_LAYER)
static void check_intermediate_layer_results(int l)
{
    int i, j, k;

    cudaMemcpy(spa_out_f32[l], sp_gpu_out_f32[l], sa_wid_s32[l] * sa_hei_s32[l] * sa_chn_s32[l] * sizeof(float), cudaMemcpyDeviceToHost);
    for(k = 0; k < sa_chn_s32[l]; k++)
    {
        for(j = 0; j < sa_hei_s32[l]; j++)
        {
            for(i = 0; i < sa_wid_s32[l]; i++)
            {
                if(fabsf(spa_out_f32[l][i + j * sa_wid_s32[l] + k * sa_wid_s32[l] * sa_hei_s32[l]] - spa_ref_f32[l][i + j * sa_wid_s32[l] + k * sa_wid_s32[l] * sa_hei_s32[l]]) > ACCEPTABLE_DIFF)
                {
                    printf("layer %d mismatch: w %d, h %d, c %d, out %f, GT %f\n", l, i, j, k, spa_out_f32[l][i + j * sa_wid_s32[l] + k * sa_wid_s32[l] * sa_hei_s32[l]], spa_ref_f32[l][i + j * sa_wid_s32[l] + k * sa_wid_s32[l] * sa_hei_s32[l]]);
                }
            }
        }
    }
}
#endif

