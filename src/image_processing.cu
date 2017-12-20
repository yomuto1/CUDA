#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <time.h>
#include <math.h>
#include <stdio.h>

#define ITER_COUNT (5)

using namespace cv;
using namespace std;

static void img_resize_natC( unsigned char *p_out_u08, unsigned char *p_in_u08 );
static void gradient_magnitude_natC( unsigned char *p_out_u08, unsigned char *p_in_u08 );
static void img_resize_omp( unsigned char *p_out_u08, unsigned char *p_in_u08 );
static void gradient_magnitude_omp( unsigned char *p_out_u08, unsigned char *p_in_u08 );

__global__ void img_resize_cuda( unsigned char *p_out_u08, unsigned char *p_in_u08 )
{
    int threadIdx_x_s32 = threadIdx.x;
    int threadIdx_y_s32 = threadIdx.y;
    int iCount_s32 = blockIdx.x * blockDim.x + threadIdx_x_s32;
    int jCount_s32 = blockIdx.y * blockDim.y + threadIdx_y_s32;

    if( (iCount_s32 < 800) && (jCount_s32 < 600) )
    {
        __shared__ unsigned char a_in_u08[32][8];

        a_in_u08[threadIdx_x_s32][threadIdx_y_s32] = p_in_u08[iCount_s32 + jCount_s32 * 800];

        __syncthreads();

        p_out_u08[iCount_s32 * 4 + jCount_s32 * 4 * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 1 + jCount_s32 * 4 * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 2 + jCount_s32 * 4 * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 3 + jCount_s32 * 4 * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + (jCount_s32 * 4 + 1) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 1 + (jCount_s32 * 4 + 1) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 2 + (jCount_s32 * 4 + 1) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 3 + (jCount_s32 * 4 + 1) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + (jCount_s32 * 4  + 2) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 1 + (jCount_s32 * 4 + 2) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 2 + (jCount_s32 * 4 + 2) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 3 + (jCount_s32 * 4 + 2) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + (jCount_s32 * 4 + 3) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 1 + (jCount_s32 * 4 + 3) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 2 + (jCount_s32 * 4 + 3) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
        p_out_u08[iCount_s32 * 4 + 3 + (jCount_s32 * 4 + 3) * 800 * 4] = a_in_u08[threadIdx_x_s32][threadIdx_y_s32];
    }
}

__global__ void gradient_magnitude_cuda( unsigned char *p_out_u08, unsigned char *p_in_u08 )
{
    int threadIdx_x_s32 = threadIdx.x;
    int threadIdx_y_s32 = threadIdx.y;
    int iCount_s32 = blockIdx.x * blockDim.x + threadIdx_x_s32;
    int jCount_s32 = blockIdx.y * blockDim.y + threadIdx_y_s32;

    unsigned int gx_u32, gy_u32;    

    if( (iCount_s32 < 3200) && (jCount_s32 < 2400) )
    {
        __shared__ unsigned char a_in_u08[64 + 2][16 + 2];

        a_in_u08[threadIdx_x_s32 + 1][threadIdx_y_s32 + 1] = p_in_u08[iCount_s32 + jCount_s32 * 3200];
        if(0 == threadIdx_x_s32)
        {
            a_in_u08[threadIdx_x_s32][threadIdx_y_s32 + 1] = p_in_u08[iCount_s32 - 1 + jCount_s32 * 3200];
            if(0 == threadIdx_y_s32)
            {
                a_in_u08[threadIdx_x_s32][threadIdx_y_s32] = p_in_u08[iCount_s32 - 1 + (jCount_s32 - 1) * 3200];
            }
        }
        if(0 == threadIdx_y_s32)
        {
            a_in_u08[threadIdx_x_s32 + 1][threadIdx_y_s32] = p_in_u08[iCount_s32 + (jCount_s32 - 1) * 3200];
        }
        if(63 == threadIdx_x_s32)
        {
            a_in_u08[threadIdx_x_s32 + 2][threadIdx_y_s32 + 1] = p_in_u08[iCount_s32 + 1 + jCount_s32 * 3200];
            if(15 == threadIdx_y_s32)
            {
                a_in_u08[threadIdx_x_s32 + 2][threadIdx_y_s32 + 2] = p_in_u08[iCount_s32 + 1 + (jCount_s32 + 1) * 3200];
            }
        }
        if(15 == threadIdx_y_s32)
        {
            a_in_u08[threadIdx_x_s32 + 1][threadIdx_y_s32 + 2] = p_in_u08[iCount_s32 + (jCount_s32 + 1) * 3200];
        }

        __syncthreads();

        const unsigned char x_p1_u08 = a_in_u08[threadIdx_x_s32 + 2][threadIdx_y_s32 + 1];
        const unsigned char x_m1_u08 = a_in_u08[threadIdx_x_s32][threadIdx_y_s32 + 1];
        const unsigned char y_p1_u08 = a_in_u08[threadIdx_x_s32 + 1][threadIdx_y_s32 + 2];
        const unsigned char y_m1_u08 = a_in_u08[threadIdx_x_s32 + 1][threadIdx_y_s32];

        gx_u32 = abs(x_p1_u08 - x_m1_u08);
        gy_u32 = abs(y_p1_u08 - y_m1_u08);

        p_out_u08[iCount_s32 + jCount_s32 * 3200] = (unsigned char)(gx_u32 + gy_u32);
    }
}


int main(void)
{
    Mat image_in;
    static unsigned char sa_image_inter_u08[3200 * 2400];
    Mat image_out;
    static unsigned char sa_image_out_natC_u08[3200 * 2400];
    static unsigned char sa_image_out_CUDA_u08[3200 * 2400];

    clock_t time_init;
    clock_t time_sum = 0;

    unsigned int iCount_u32, jCount_u32;

    time_init = clock();
    time_init = clock() - time_init;

    cout <<  "Hello CUDA" << std::endl;

    image_in = imread( "../../../Pictures/PNG_transparency_demonstration_1.png", CV_LOAD_IMAGE_GRAYSCALE);

    if( image_in.data == NULL )
    {
        cout <<  "Could not open or find the image" << std::endl;
        
        return -1;
    }

    imwrite( "../../../Pictures/out_img_1.png", image_in );

    /* 4 times resize with CPU OpenCV */
    time_sum = clock();
    for( iCount_u32 = 0; iCount_u32 < ITER_COUNT; iCount_u32++ )
    {
        resize( image_in, image_out, Size(), 4.0, 4.0, CV_INTER_NN );
    }
    time_sum = clock() - time_init;

    cout <<  "LINEAR upscaling to 4x by CPU OpenCV " << (float)time_sum / CLOCKS_PER_SEC << std::endl;

    imwrite( "../../../Pictures/out_img_1_opencv_cpu.png", image_out );

    memset(sa_image_inter_u08, 0, 800 * 600 * 4 * 4 * sizeof(unsigned char));
    memset(image_out.data, 0, 800 * 600 * 4 * 4 * sizeof(unsigned char));

    /* 4 times resize with CPU natC */
    time_sum = clock();
    for( iCount_u32 = 0; iCount_u32 < ITER_COUNT; iCount_u32++ )
    {
        img_resize_natC( sa_image_inter_u08, image_in.data );
        gradient_magnitude_natC( image_out.data, sa_image_inter_u08 );
    }
    time_sum = clock() - time_init;

    cout <<  "LINEAR upscaling to 4x by CPU natC " << (float)time_sum / CLOCKS_PER_SEC << std::endl;

    memcpy(sa_image_out_natC_u08, image_out.data, 3200 * 2400 * sizeof(unsigned char));

    imwrite( "../../../Pictures/out_img_1_opencv_natC.png", image_out );

    memset(sa_image_inter_u08, 0, 800 * 600 * 4 * 4 * sizeof(unsigned char));
    memset(image_out.data, 0, 800 * 600 * 4 * 4 * sizeof(unsigned char));

    /* 4 times resize with CPU OMP */
    time_sum = clock();
    for( iCount_u32 = 0; iCount_u32 < ITER_COUNT; iCount_u32++ )
    {
        img_resize_omp( sa_image_inter_u08, image_in.data );
        gradient_magnitude_omp( image_out.data, sa_image_inter_u08 );
    }
    time_sum = clock() - time_init;

    cout <<  "LINEAR upscaling to 4x by GPU OMP " << (float)time_sum / CLOCKS_PER_SEC << std::endl;

    imwrite( "../../../Pictures/out_img_1_GPU_OMP.png", image_out );

    memset(sa_image_inter_u08, 0, 800 * 600 * 4 * 4 * sizeof(unsigned char));
    memset(image_out.data, 0, 800 * 600 * 4 * 4 * sizeof(unsigned char));

    /* 4 times resize with GPU CUDA */
    {
        unsigned char *p_image_in_u08;
        unsigned char *p_image_inter_u08;
        unsigned char *p_image_out_u08;

        dim3 grid_img_resize( 32, 8 );
        dim3 grid_numblocks_resize( 800 / grid_img_resize.x, 600 / grid_img_resize.y );
        dim3 grid_gradient_magnitude( 64, 16 );
        dim3 grid_numblocks_gradient( 3200 / grid_gradient_magnitude.x, 2400 / grid_gradient_magnitude.y);

        cudaMalloc( (void**)&p_image_in_u08, 800 * 600 * sizeof(unsigned char) );
        cudaMalloc( (void**)&p_image_inter_u08, 800 * 600 * 4 * 4 * sizeof(unsigned char) );
        cudaMalloc( (void**)&p_image_out_u08, 800 * 600 * 4 * 4 * sizeof(unsigned char) );

        cudaMemcpy( p_image_in_u08, image_in.data, 800 * 600 * sizeof(unsigned char), cudaMemcpyHostToDevice );

        time_sum = clock();
        for( iCount_u32 = 0; iCount_u32 < ITER_COUNT; iCount_u32++ )
        {
            img_resize_cuda<<<grid_numblocks_resize, grid_img_resize>>>( p_image_inter_u08, p_image_in_u08 );
            gradient_magnitude_cuda<<<grid_numblocks_gradient, grid_gradient_magnitude>>>( p_image_out_u08, p_image_inter_u08 );
            cudaMemcpy( image_out.data, p_image_out_u08, 800 * 600 * 4 * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost );
        }
        time_sum = clock() - time_init;

        cudaMemcpy( image_out.data, p_image_out_u08, 800 * 600 * 4 * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost );

        cout <<  "LINEAR upscaling to 4x by GPU CUDA " << (float)time_sum / CLOCKS_PER_SEC << std::endl;

        memcpy(sa_image_out_CUDA_u08, image_out.data, 3200 * 2400 * sizeof(unsigned char));

        imwrite( "../../../Pictures/out_img_1_GPU_CUDA.png", image_out );

        for( jCount_u32 = 1; jCount_u32 < 2400 - 1; jCount_u32++ )
        {
            for( iCount_u32 = 1; iCount_u32 < 3200 - 1; iCount_u32++ )
            {
                if( sa_image_out_natC_u08[iCount_u32 + jCount_u32 * 3200] != sa_image_out_CUDA_u08[iCount_u32 + jCount_u32 * 3200] )
                {
                    printf("not matched: (%d, %d), %d, %d\n", iCount_u32, jCount_u32, sa_image_out_natC_u08[iCount_u32 + jCount_u32 * 3200], sa_image_out_CUDA_u08[iCount_u32 + jCount_u32 * 3200]);
                }
            }
        }

        cudaFree( p_image_in_u08 );
        cudaFree( p_image_inter_u08 );
        cudaFree( p_image_out_u08 );
    }

    return 0;
}

static void img_resize_natC( unsigned char *p_out_u08, unsigned char *p_in_u08 )
{
    unsigned int iCount_u32, jCount_u32;

    for( jCount_u32 = 0; jCount_u32 < 600; jCount_u32++ )
    {
        for( iCount_u32 = 0; iCount_u32 < 800; iCount_u32++ )
        {
            p_out_u08[iCount_u32 * 4 + 0 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 0 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 0 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 0 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
        }
    }
}

static void gradient_magnitude_natC( unsigned char *p_out_u08, unsigned char *p_in_u08 )
{
    unsigned int iCount_u32, jCount_u32;
    unsigned char gx_u08, gy_u08;

    for( jCount_u32 = 1; jCount_u32 < 2400 - 1; jCount_u32++ )
    {
        for( iCount_u32 = 1; iCount_u32 < 3200 - 1; iCount_u32++ )
        {
            gx_u08 = abs(p_in_u08[iCount_u32 + 1 + jCount_u32 * 3200] - p_in_u08[iCount_u32 - 1 + jCount_u32 * 3200]);
            gy_u08 = abs(p_in_u08[iCount_u32 + (jCount_u32 + 1) * 3200] - p_in_u08[iCount_u32 + (jCount_u32 - 1) * 3200]);

            p_out_u08[iCount_u32 + jCount_u32 * 3200] = gx_u08 + gy_u08;
        }
    }
}

static void img_resize_omp( unsigned char *p_out_u08, unsigned char *p_in_u08 )
{
    unsigned int iCount_u32, jCount_u32;

#pragma omp parallel for
    for( jCount_u32 = 0; jCount_u32 < 600; jCount_u32++ )
    {
#pragma omp parallel for
        for( iCount_u32 = 0; iCount_u32 < 800; iCount_u32++ )
        {
            p_out_u08[iCount_u32 * 4 + 0 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + jCount_u32 * 4 * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 0 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + (jCount_u32 * 4 + 1) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 0 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + (jCount_u32 * 4 + 2) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 0 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 1 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 2 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
            p_out_u08[iCount_u32 * 4 + 3 + (jCount_u32 * 4 + 3) * 800 * 4] = p_in_u08[iCount_u32 + jCount_u32 * 800];
        }
    }
}

static void gradient_magnitude_omp( unsigned char *p_out_u08, unsigned char *p_in_u08 )
{
    unsigned int iCount_u32, jCount_u32;
    unsigned char gx_u08, gy_u08;

#pragma omp parallel for
    for( jCount_u32 = 1; jCount_u32 < 2400 - 1; jCount_u32++ )
    {
#pragma omp parallel for
        for( iCount_u32 = 1; iCount_u32 < 3200 - 1; iCount_u32++ )
        {
            gx_u08 = abs(p_in_u08[iCount_u32 + 1 + jCount_u32 * 3200] - p_in_u08[iCount_u32 - 1 + jCount_u32 * 3200]);
            gy_u08 = abs(p_in_u08[iCount_u32 + (jCount_u32 + 1) * 3200] - p_in_u08[iCount_u32 + (jCount_u32 - 1) * 3200]);

            p_out_u08[iCount_u32 + jCount_u32 * 3200] = gx_u08 + gy_u08;
        }
    }
}

