#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#define PI 3.141592653589793
#define running_block_size 32
#define running_grid_size 131072




// __global__ defines the funciton that can be called from the host (CPU) and executed in the device (GPU)
__global__
void check_int(int n, int *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%d\n", i, x[i]);
                printf("%.5e\n", x[i]);
            }
        }
    }
}

__global__
void check_long_int(int n, long long int *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\n", i);
                printf("index: %d\t%lld\n", i, x[i]);
                // printf("%lld\n", x[i]);
            }
        }
    }
}



__global__
void check_double(int n, double *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\n", x[i]);
            }
        }
    }
}


__global__
void check_double_k(double *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;

    if (index<5)
    {
        printf("%d %lf\n", index, x[438048+index]);
    }
}


__global__
void check_double_sci(int n, double *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%.5e\n", x[i]);
            }
        }
    }
}



__global__
void check_int_custom2(int n, int *x1, int *x2, int *x3, int *x4, int *x5, int *x6, 
    int *x7, int *x8, int *x9, int *x10, int *x11, int *x12)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%d %d %d %d %d %d %d %d %d %d %d %d\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], 
                    x7[i], x8[i], x9[i], x10[i], x11[i], x12[i]);
            }
        }
    }
}








//convert fractional coordinate to cartesian coordinate
void frac2car(double frac_a, double frac_b, double frac_c, double *frac2car_a, double *frac2car_b, double *frac2car_c,
                double *cart_x, double *cart_y, double *cart_z)
{
    cart_x[0] = frac_a*frac2car_a[0] + frac_b*frac2car_a[1] + frac_c*frac2car_a[2];
    cart_y[0] = frac_a*frac2car_b[0] + frac_b*frac2car_b[1] + frac_c*frac2car_b[2];
    cart_z[0] = frac_a*frac2car_c[0] + frac_b*frac2car_c[1] + frac_c*frac2car_c[2];
}



//expand the lattice to a larger size
void pbc_expand(int *N_atom_frame, int *times_x, int *times_y, int *times_z, double *frac_a_frame, double *frac_b_frame, double *frac_c_frame,
                double *epsilon_frame, double *sigma_frame, double *mass_frame)
{
    int i, ii, iii, iiii;
    int j;
    iiii = 0;
    for (j=0; j<N_atom_frame[0]; j++)
    {
        for (i=0; i<times_x[0]; i++)
        {
            for (ii=0; ii<times_y[0]; ii++)
            {
                for (iii=0; iii<times_z[0]; iii++)
                {
                    if ((i!=0)||(ii!=0)||(iii!=0))
                    {
                        frac_a_frame[N_atom_frame[0]+iiii] = frac_a_frame[j] + i;
                        frac_b_frame[N_atom_frame[0]+iiii] = frac_b_frame[j] + ii;
                        frac_c_frame[N_atom_frame[0]+iiii] = frac_c_frame[j] + iii;
                        epsilon_frame[N_atom_frame[0]+iiii] = epsilon_frame[j];
                        sigma_frame[N_atom_frame[0]+iiii] = sigma_frame[j];
                        mass_frame[N_atom_frame[0]+iiii] = mass_frame[j];
                        iiii++;
                    }
                }
            }
        }
    }
}









__device__
double frac2car_x_device(double frac_a, double frac_b, double frac_c, double *frac2car_a_device)
{
    return (frac_a*frac2car_a_device[0] + frac_b*frac2car_a_device[1] + frac_c*frac2car_a_device[2]);
}

__device__
double frac2car_y_device(double frac_a, double frac_b, double frac_c, double *frac2car_b_device)
{
    return (frac_a*frac2car_b_device[0] + frac_b*frac2car_b_device[1] + frac_c*frac2car_b_device[2]);
}

__device__
double frac2car_z_device(double frac_a, double frac_b, double frac_c, double *frac2car_c_device)
{
    return (frac_a*frac2car_c_device[0] + frac_b*frac2car_c_device[1] + frac_c*frac2car_c_device[2]);
}




__device__
double rotate_moleucle_x_device(double rot_alpha_rad, double rot_beta_rad, double rot_gamma_rad,
                                double vector_adsorbate_x_device, double vector_adsorbate_y_device, double vector_adsorbate_z_device)
{
    return ( vector_adsorbate_x_device*cos(rot_gamma_rad)*cos(rot_beta_rad) 
            - vector_adsorbate_y_device*sin(rot_gamma_rad)*cos(rot_alpha_rad) 
            + vector_adsorbate_y_device*sin(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad) 
            + vector_adsorbate_z_device*sin(rot_gamma_rad)*sin(rot_alpha_rad) 
            + vector_adsorbate_z_device*cos(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad) );
}

__device__
double rotate_moleucle_y_device(double rot_alpha_rad, double rot_beta_rad, double rot_gamma_rad,
                                double vector_adsorbate_x_device, double vector_adsorbate_y_device, double vector_adsorbate_z_device)
{
    return ( vector_adsorbate_x_device*sin(rot_gamma_rad)*cos(rot_beta_rad) 
            + vector_adsorbate_y_device*cos(rot_gamma_rad)*cos(rot_alpha_rad) 
            + vector_adsorbate_y_device*sin(rot_gamma_rad)*sin(rot_beta_rad)*sin(rot_alpha_rad)
            - vector_adsorbate_z_device*cos(rot_gamma_rad)*sin(rot_alpha_rad) 
            + vector_adsorbate_z_device*sin(rot_gamma_rad)*sin(rot_beta_rad)*cos(rot_alpha_rad) );
}

__device__
double rotate_moleucle_z_device(double rot_alpha_rad, double rot_beta_rad, double rot_gamma_rad,
                                double vector_adsorbate_x_device, double vector_adsorbate_y_device, double vector_adsorbate_z_device)
{
    return ( -vector_adsorbate_x_device*sin(rot_beta_rad) 
            + vector_adsorbate_y_device*cos(rot_beta_rad)*sin(rot_alpha_rad) 
            + vector_adsorbate_z_device*cos(rot_alpha_rad)*cos(rot_beta_rad) );
}


__device__
double cal_dis_device(double loc_x_device, double loc_y_device, double loc_z_device, 
    double frame_x_device, double frame_y_device, double frame_z_device)
{

    return ( sqrt( pow((loc_x_device-frame_x_device),2)+pow((loc_y_device-frame_y_device),2)+pow((loc_z_device-frame_z_device),2) ) );
}



__device__
double cal_pure_lj_device(double epsilon_cal_device, double sigma_cal_device, double distance)
{
    return ( 4*epsilon_cal_device*(pow((sigma_cal_device/distance),12) - pow((sigma_cal_device/distance),6)) );
}







// solution 1 for GPU Vext on the initial plane
__global__
void V_batch(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
        double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
        int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
        double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
        int *times_x_device, int *times_y_device, int *times_z_device,
        int *N_atom_frame_extend_device,
        double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
        double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
        double *cutoff_device,

        int *index_a_start_device, int *index_b_start_device, int *index_c_start_device,
        int *index_a_end_device, int *index_b_end_device, int *index_c_end_device,
        int *index_alpha_start_device, int *index_beta_start_device, int *index_gamma_start_device,
        int *index_alpha_end_device, int *index_beta_end_device, int *index_gamma_end_device, 

        int *N_a_device, int *N_b_device, int *N_c_device,
        int *N_alpha_device, int *N_beta_device, int *N_gamma_device, 
        
        int *index_a_device, int *index_b_device, int *index_c_device,
        int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
        int *index_adsorbate_device, int *index_frame_device,

        double *a_Vext_cal_device, double *b_Vext_cal_device, double *c_Vext_cal_device,
        double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device,
        double *loc_x_device, double *loc_y_device, double *loc_z_device,
        double *vector_adsorbate_x_rot_device, double *vector_adsorbate_y_rot_device, double *vector_adsorbate_z_rot_device,
        double *adsorbate_cart_x_rot_device, double *adsorbate_cart_y_rot_device, double *adsorbate_cart_z_rot_device, 
        double *modify_frame_a_device, double *modify_frame_b_device, double *modify_frame_c_device,
        double *minimum_distance_device,
        double *V_result_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j, jj;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
       

    for (i=index; 
        i<( ( (index_a_end_device[0]-index_a_start_device[0]+1)*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
    *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)*(index_gamma_end_device[0]-index_gamma_start_device[0]+1)
    *N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0] )  ); 
        i+=stride)
    {
        index_a_device[i] = (int) ( (i) / ((index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) );
        index_b_device[i] = (int) ( (i - index_a_device[i]*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) 
            / ((index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) );
        index_c_device[i] = (int) ( (i - index_a_device[i]*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_b_device[i]*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) 
            / ((index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) );
        index_alpha_device[i] = (int) ( (i - index_a_device[i]*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_b_device[i]*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_c_device[i]*(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) 
            / ((index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) );
        index_beta_device[i] = (int) ( (i - index_a_device[i]*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_b_device[i]*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_c_device[i]*(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_alpha_device[i]*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) 
            / ((index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) );
        index_gamma_device[i] = (int) ( (i - index_a_device[i]*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_b_device[i]*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_c_device[i]*(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_alpha_device[i]*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_beta_device[i]*(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) 
            / (N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) );
        index_adsorbate_device[i] = (int) ( (i - index_a_device[i]*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_b_device[i]*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_c_device[i]*(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_alpha_device[i]*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_beta_device[i]*(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]) 
            / (N_atom_frame_extend_device[0]) );
        index_frame_device[i] = (int) ( (i - index_a_device[i]*(index_b_end_device[0]-index_b_start_device[0]+1)*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_b_device[i]*(index_c_end_device[0]-index_c_start_device[0]+1)
            *(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_c_device[i]*(index_alpha_end_device[0]-index_alpha_start_device[0]+1)*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_alpha_device[i]*(index_beta_end_device[0]-index_beta_start_device[0]+1)
            *(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_beta_device[i]*(index_gamma_end_device[0]-index_gamma_start_device[0]+1)*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*N_atom_frame_extend_device[0]
            - index_adsorbate_device[i]*N_atom_frame_extend_device[0]) );



        a_Vext_cal_device[i] = index_a_start_device[0]*1.0/N_a_device[0] + index_a_device[i]*1.0/N_a_device[0];
        b_Vext_cal_device[i] = index_b_start_device[0]*1.0/N_b_device[0] + index_b_device[i]*1.0/N_b_device[0];
        c_Vext_cal_device[i] = index_c_start_device[0]*1.0/N_c_device[0] + index_c_device[i]*1.0/N_c_device[0];

        rot_alpha_rad_device[i] = index_alpha_start_device[0]*2.0*PI/N_alpha_device[0] + index_alpha_device[i]*2.0*PI/N_alpha_device[0];
        rot_beta_rad_device[i] = index_beta_start_device[0]*1.0*PI/N_beta_device[0] + index_beta_device[i]*1.0*PI/N_beta_device[0];
        rot_gamma_rad_device[i] = index_gamma_start_device[0]*2.0*PI/N_gamma_device[0] + index_gamma_device[i]*2.0*PI/N_gamma_device[0];

        // if ((i==0)||(i==1)||(i==424)||(i==425)||(i==848)||(i==849)||(i==5088)||(i==30528)||(i==4240)||(i==30955392)||(i==30955393)||(i==56078288))
        // if ((i==0)||(i==59)||(i==60)||(i==3599)||(i==3600))
        // {
        //     printf("index: %d gpu check: %d %d %d %d %d %d %d %d numerical: %lf %lf %lf %lf %lf %lf\n", i, index_a_device[i], index_b_device[i], index_c_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i], index_adsorbate_device[i], index_frame_device[i],
        //         a_Vext_cal_device[i], b_Vext_cal_device[i], c_Vext_cal_device[i], rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }

        // if ((i==0)||(i==15477696))
        // {
        //     printf("index: %d gpu check: %d %d %d %d %d %d %d %d numerical: %lf %lf %lf %lf %lf %lf\n", i, index_a_device[i], index_b_device[i], index_c_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i], index_adsorbate_device[i], index_frame_device[i],
        //         a_Vext_cal_device[i], b_Vext_cal_device[i], c_Vext_cal_device[i], rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }

        loc_x_device[i] = frac2car_x_device(a_Vext_cal_device[i], b_Vext_cal_device[i], c_Vext_cal_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(a_Vext_cal_device[i], b_Vext_cal_device[i], c_Vext_cal_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(a_Vext_cal_device[i], b_Vext_cal_device[i], c_Vext_cal_device[i], frac2car_c_device);

        // if ()


        vector_adsorbate_x_rot_device[i] = rotate_moleucle_x_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);
        vector_adsorbate_y_rot_device[i] = rotate_moleucle_y_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);
        vector_adsorbate_z_rot_device[i] = rotate_moleucle_z_device(rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i],
            vector_adsorbate_x_device[index_adsorbate_device[i]], vector_adsorbate_y_device[index_adsorbate_device[i]], vector_adsorbate_z_device[index_adsorbate_device[i]]);

        adsorbate_cart_x_rot_device[i] = loc_x_device[i]+vector_adsorbate_x_rot_device[i];
        adsorbate_cart_y_rot_device[i] = loc_y_device[i]+vector_adsorbate_y_rot_device[i];
        adsorbate_cart_z_rot_device[i] = loc_z_device[i]+vector_adsorbate_z_rot_device[i];

        V_result_device[i] = 0;






        // z-direction
        if ( (adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_c_device)) 
            > 0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]] + times_z_device[0];
        }
        else if ( (adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_c_device)) 
            < -0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]] - times_z_device[0];
        }
        else
        {
            modify_frame_c_device[i] = frac_c_frame_device[index_frame_device[i]];
        }



        // y-direction
        if ( (adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_b_device)) 
            > 0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]] + times_y_device[0];
        }
        else if ( (adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_b_device)) 
            < -0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]] - times_y_device[0];
        }
        else
        {
            modify_frame_b_device[i] = frac_b_frame_device[index_frame_device[i]];
        }

        // x-direction
        if ( (adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_a_device)) 
            > 0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]] + times_x_device[0];
        }
        else if ( (adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_device[i]], frac_b_frame_device[index_frame_device[i]], 
                frac_c_frame_device[index_frame_device[i]], frac2car_a_device)) 
            < -0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]] - times_x_device[0];
        }
        else
        {
            modify_frame_a_device[i] = frac_a_frame_device[index_frame_device[i]];
        }

        minimum_distance_device[i] = cal_dis_device(adsorbate_cart_x_rot_device[i], adsorbate_cart_y_rot_device[i], adsorbate_cart_z_rot_device[i], 
                    frac2car_x_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_a_device), 
                    frac2car_y_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_b_device), 
                    frac2car_z_device(modify_frame_a_device[i], modify_frame_b_device[i], modify_frame_c_device[i], frac2car_c_device));

        if (minimum_distance_device[i] < 
            (((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5)*0.1))
        {
            minimum_distance_device[i] = (((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5)*0.1);
        }

        if (minimum_distance_device[i] < cutoff_device[0])
        {
            V_result_device[i] = V_result_device[i] + 
            ( cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0]) );
        }
        
    }

}






// solution 1 for GPU Vext on the initial plane
__global__
void cal_H_seg_from_Vext(double *V_result_device, 
        int *N_a_device, int *N_b_device, int *N_c_device,
        int *N_alpha_device, int *N_beta_device, int *N_gamma_device, 
        
        int *index_a_device, int *index_b_device, int *index_c_device,
        int *index_alpha_device, int *index_beta_device, int *index_gamma_device,

        double *rot_beta_rad_device, double *temperature_device,

        double *H_segment_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
       

    for (i=index; 
        i<( ( N_a_device[0]*N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] )  ); 
        i+=stride)
    {
        index_a_device[i] = (int) ( (i) / (N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0]) );
        index_b_device[i] = (int) ( (i - index_a_device[i]*N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0]) 
            / (N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0]) );
        index_c_device[i] = (int) ( (i - index_a_device[i]*N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_b_device[i]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0])
            / (N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0]) );
        index_alpha_device[i] = (int) ( (i - index_a_device[i]*N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_b_device[i]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_c_device[i]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0])
            / (N_beta_device[0]*N_gamma_device[0]) );
        index_beta_device[i] = (int) ( (i - index_a_device[i]*N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_b_device[i]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_c_device[i]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_alpha_device[i]*N_beta_device[0]*N_gamma_device[0] )
            / (N_gamma_device[0]) );
        index_gamma_device[i] = (int) ( (i - index_a_device[i]*N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_b_device[i]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_c_device[i]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0] 
            - index_alpha_device[i]*N_beta_device[0]*N_gamma_device[0] 
            - index_beta_device[i]*N_gamma_device[0]) );

        rot_beta_rad_device[i] = index_beta_device[i]*1.0*PI/N_beta_device[0];



        if (V_result_device[i]>1e5)
        {
            H_segment_device[i] = 0;
        }
        else
        {
            H_segment_device[i] = exp(-V_result_device[i]/temperature_device[0])*sin(rot_beta_rad_device[i])
            /(N_a_device[0]*N_b_device[0]*N_c_device[0]*N_alpha_device[0]*N_beta_device[0]*N_gamma_device[0])
            *PI*0.5;
        }

        // if (i==438052)
        // {
        //     printf("%d %lf %.5e\n", i, V_result_device[i], H_segment_device[i]);
        // }
        

        // if ((i==0)||(i==6)||(i==12)||(i==24)||(i==30)||(i==36)||(i==429845)||(i==60261)||(i==474551)||(i==474552))
        // {
        //     printf("index: %d gpu check: %d %d %d %d %d %d numerical: %lf %lf %lf\n", i, index_a_device[i], index_b_device[i], index_c_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i],
        //         rot_beta_rad_device[i], V_result_device[i], H_segment_device[i]);
        // }


        
    }

}











int main(int argc, char *argv[])
{
    clock_t t;
    t = clock();
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //------------------------------------------------------File I/O--------------------------------------------
	//---------define host variable to store input parameters
	FILE *fp1;
	int buffersize = 512;
	char str[buffersize];
	//define read-in parameters
	int Nmax_a, Nmax_b, Nmax_c;
	double La, Lb, Lc, dL;
    int a_N, b_N, c_N;
	double alpha, beta, gamma;
    double alpha_rad, beta_rad, gamma_rad;
    int alpha_N, beta_N, gamma_N;
    int FH_signal;
    double mass, temperature;
	double cutoff[1];
	int N_atom_frame[1], N_atom_adsorbate[1];
    //define ancillary parameters
    double center_of_mass_x[1], center_of_mass_y[1], center_of_mass_z[1], total_mass_adsorbate;
    double temp_x[1], temp_y[1], temp_z[1];
    double cart_x, cart_y, cart_z;
    double cart_x_extended[1], cart_y_extended[1], cart_z_extended[1];
    int times_x[1], times_y[1], times_z[1], times;
    double a;
    double shift;
    // double loc_a, loc_b, loc_c, loc_x, loc_y, loc_z, loc_u;
    double temp_frame_a, temp_frame_b, temp_frame_c;
    double temp_u;
    int i, ii, iii, iiii, j, jj, jjj, k, kk;
    double dis;
    //---------define host variable to store input parameters------done------
    //---------read input file parameters
	fp1 = fopen(argv[1], "r");
	// fp1 = fopen("AMUWIP_charged.input", "r");
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%d %d %d\n", &Nmax_a, &Nmax_b, &Nmax_c);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf %lf\n", &La, &Lb, &Lc, &dL);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf\n", &alpha, &beta, &gamma);
    alpha_rad = alpha*PI/180;
    beta_rad = beta*PI/180;
    gamma_rad = gamma*PI/180;
    fgets(str, buffersize, fp1);
    fscanf(fp1, "%d %d %d\n", &alpha_N, &beta_N, &gamma_N);
	fgets(str, buffersize, fp1);
    fscanf(fp1,"%lf %d %lf %lf\n", &cutoff[0], &FH_signal, &total_mass_adsorbate, &temperature);
    //read adsorbate information
    fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
    fscanf(fp1,"%d\n", &N_atom_adsorbate[0]);
    double x_adsorbate[N_atom_adsorbate[0]], y_adsorbate[N_atom_adsorbate[0]], z_adsorbate[N_atom_adsorbate[0]];
    double epsilon_adsorbate[N_atom_adsorbate[0]], sigma_adsorbate[N_atom_adsorbate[0]], mass_adsorbate[N_atom_adsorbate[0]];
    double vector_adsorbate_x[N_atom_adsorbate[0]], vector_adsorbate_y[N_atom_adsorbate[0]], vector_adsorbate_z[N_atom_adsorbate[0]];
    fgets(str, buffersize, fp1);
    center_of_mass_x[0] = 0;
    center_of_mass_y[0] = 0;
    center_of_mass_z[0] = 0;
    for (i=0; i<N_atom_adsorbate[0]; i++)
    {
        fscanf(fp1,"%lf %lf %lf %lf %lf %lf\n", &x_adsorbate[i], &y_adsorbate[i], &z_adsorbate[i], &epsilon_adsorbate[i], 
            &sigma_adsorbate[i], &mass_adsorbate[i]);
        // printf("%lf %lf %lf %lf %lf %lf\n", x_adsorbate[i], y_adsorbate[i], z_adsorbate[i], epsilon_adsorbate[i], 
        //     sigma_adsorbate[i], mass_adsorbate[i]);
        center_of_mass_x[0] += 1.0*x_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
        center_of_mass_y[0] += 1.0*y_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
        center_of_mass_z[0] += 1.0*z_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
    }
    //determin the vector of each atom with respect to the center of mass
    for (i=0; i<N_atom_adsorbate[0]; i++)
    {
        vector_adsorbate_x[i] = x_adsorbate[i] - center_of_mass_x[0];
        vector_adsorbate_y[i] = y_adsorbate[i] - center_of_mass_y[0];
        vector_adsorbate_z[i] = z_adsorbate[i] - center_of_mass_z[0];
    }
    //read framework information
	fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
	fscanf(fp1,"%d\n", &N_atom_frame[0]);
	fgets(str, buffersize, fp1);
    //frac2car parameter calculation
	double frac2car_a[3];
	double frac2car_b[3];
	double frac2car_c[3];
    frac2car_a[0] = La;
    frac2car_a[1] = Lb*cos(gamma_rad);
    frac2car_a[2] = Lc*cos(beta_rad);
    frac2car_b[0] = 0;
    frac2car_b[1] = Lb*sin(gamma_rad);
    frac2car_b[2] = Lc*( (cos(alpha_rad)-cos(beta_rad)*cos(gamma_rad)) / sin(gamma_rad) );
    frac2car_c[2] = La*Lb*Lc*sqrt( 1 - pow(cos(alpha_rad),2) - pow(cos(beta_rad),2) - pow(cos(gamma_rad),2) + 2*cos(alpha_rad)*cos(beta_rad)*cos(gamma_rad) );
	frac2car_c[2] = frac2car_c[2]/(La*Lb*sin(gamma_rad));
	//done!!!!!
    //expand the cell to the size satisfied cutoff condition
    //convert the fractional cell length to cartesian value;
    frac2car(1, 0, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_x = temp_x[0];
    frac2car(0, 1, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_y = temp_y[0];
    frac2car(0, 0, 1, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_z = temp_z[0];
    times_x[0] = (int) 2*cutoff[0]/cart_x + 1;
    times_y[0] = (int) 2*cutoff[0]/cart_y + 1;
    times_z[0] = (int) 2*cutoff[0]/cart_z + 1;
    times = times_x[0]*times_y[0]*times_z[0];
    // printf("%d\n", times);
	double epsilon_frame[N_atom_frame[0]*times], sigma_frame[N_atom_frame[0]*times], mass_frame[N_atom_frame[0]*times];
	double frac_a_frame[N_atom_frame[0]*times], frac_b_frame[N_atom_frame[0]*times], frac_c_frame[N_atom_frame[0]*times];
    for (i=0; i<N_atom_frame[0]; i++)
	{
		fscanf(fp1,"%lf %lf %lf %lf %lf %lf %lf\n", &a, &sigma_frame[i], &epsilon_frame[i], &mass_frame[i], &frac_a_frame[i], &frac_b_frame[i], &frac_c_frame[i]);
        // printf("%lf %lf %lf %lf %lf %lf %lf\n", a, sigma_frame[i], epsilon_frame[i], mass_frame[i], frac_a_frame[i], frac_b_frame[i], frac_c_frame[i]);
    	fgets(str, buffersize, fp1);
    }
    fclose(fp1);
    pbc_expand(N_atom_frame, times_x, times_y, times_z, frac_a_frame, frac_b_frame, frac_c_frame, epsilon_frame, sigma_frame, mass_frame);
    frac2car(times_x[0], 0, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_x_extended[0] = temp_x[0];
    frac2car(0, times_y[0], 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_y_extended[0] = temp_y[0];
    frac2car(0, 0, times_z[0], frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_z_extended[0] = temp_z[0];
    //---------read input file parameters------done------
    // printf("%d %d %d\n", Nmax_a, Nmax_b, Nmax_c);
    // printf("%lf %lf %lf %lf\n", La, Lb, Lc, dL);
    // printf("%lf %lf %lf\n", alpha, beta, gamma);
    // printf("%d %d %d\n", alpha_N, beta_N, gamma_N);
    // printf("%lf %d %lf %lf\n", cutoff[0], FH_signal, total_mass_adsorbate, temperature);
    // printf("N_atom_adsorbate: %d\n", N_atom_adsorbate[0]);
    // printf("center of the mass:\t%lf\t%lf\t%lf\n", center_of_mass_x[0], center_of_mass_y[0], center_of_mass_z[0]);
    // printf("N_atom_frame: %d\n", N_atom_frame[0]);
    // return 0;
    //------------------------------------------------------File I/O--------------------------------------------
    //-------------------------------------------------------done-----------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------










    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //------------------------------------GPU Variable allocation and transfer----------------------------------
    //define variables on device
    double *cart_x_extended_device, *cart_y_extended_device, *cart_z_extended_device;
    double *cutoff_device;
    int *N_atom_adsorbate_device;
    double *epsilon_adsorbate_device, *sigma_adsorbate_device;
    double *center_of_mass_x_device, *center_of_mass_y_device, *center_of_mass_z_device;
    double *vector_adsorbate_x_device, *vector_adsorbate_y_device, *vector_adsorbate_z_device;
    double *temperature_device;
    int *N_atom_frame_device;
    int *times_x_device, *times_y_device, *times_z_device;
    double *epsilon_frame_device, *sigma_frame_device, *mass_frame_device;
    double *frac_a_frame_device, *frac_b_frame_device, *frac_c_frame_device;
    double *frac2car_a_device, *frac2car_b_device, *frac2car_c_device;
    //allocate memory on device
    cudaMalloc((void **)&cart_x_extended_device, sizeof(double));
    cudaMalloc((void **)&cart_y_extended_device, sizeof(double));
    cudaMalloc((void **)&cart_z_extended_device, sizeof(double));
    cudaMalloc((void **)&cutoff_device, sizeof(double));
    cudaMalloc((void **)&N_atom_adsorbate_device, sizeof(int));
    cudaMalloc((void **)&epsilon_adsorbate_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&sigma_adsorbate_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&center_of_mass_x_device, sizeof(double));
    cudaMalloc((void **)&center_of_mass_y_device, sizeof(double));
    cudaMalloc((void **)&center_of_mass_z_device, sizeof(double));
    cudaMalloc((void **)&vector_adsorbate_x_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&vector_adsorbate_y_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&vector_adsorbate_z_device, sizeof(double)*N_atom_adsorbate[0]);
    cudaMalloc((void **)&temperature_device, sizeof(double));
    cudaMalloc((void **)&N_atom_frame_device, sizeof(int));
    cudaMalloc((void **)&times_x_device, sizeof(int));
    cudaMalloc((void **)&times_y_device, sizeof(int));
    cudaMalloc((void **)&times_z_device, sizeof(int));
    cudaMalloc((void **)&epsilon_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&sigma_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&mass_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac_a_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac_b_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac_c_frame_device, sizeof(double)*N_atom_frame[0]*times);
    cudaMalloc((void **)&frac2car_a_device, sizeof(double)*3);
    cudaMalloc((void **)&frac2car_b_device, sizeof(double)*3);
    cudaMalloc((void **)&frac2car_c_device, sizeof(double)*3);
    // //copy and transfer arrary concurrently
    // cudaMemcpy(cart_x_extended_device, cart_x_extended, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(cart_y_extended_device, cart_y_extended, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(cart_z_extended_device, cart_z_extended, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(cutoff_device, cutoff, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(N_atom_adsorbate_device, N_atom_adsorbate, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(epsilon_adsorbate_device, epsilon_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(sigma_adsorbate_device, sigma_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(center_of_mass_x_device, center_of_mass_x, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(center_of_mass_y_device, center_of_mass_y, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(center_of_mass_z_device, center_of_mass_z, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(vector_adsorbate_x_device, vector_adsorbate_x, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(vector_adsorbate_y_device, vector_adsorbate_y, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(vector_adsorbate_z_device, vector_adsorbate_z, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice);
    // cudaMemcpy(temperature_device, temperature, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(N_atom_frame_device, N_atom_frame, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(times_x_device, times_x, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(times_y_device, times_y, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(times_z_device, times_z, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(epsilon_frame_device, epsilon_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice);
    // cudaMemcpy(sigma_frame_device, sigma_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice);

    //copy and transfer arrary asynchronously
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaMemcpyAsync(cart_x_extended_device, cart_x_extended, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(cart_y_extended_device, cart_y_extended, sizeof(double), cudaMemcpyHostToDevice), stream1;
    cudaMemcpyAsync(cart_z_extended_device, cart_z_extended, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(cutoff_device, cutoff, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_atom_adsorbate_device, N_atom_adsorbate, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(epsilon_adsorbate_device, epsilon_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(sigma_adsorbate_device, sigma_adsorbate, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(center_of_mass_x_device, center_of_mass_x, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(center_of_mass_y_device, center_of_mass_y, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(center_of_mass_z_device, center_of_mass_z, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(vector_adsorbate_x_device, vector_adsorbate_x, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(vector_adsorbate_y_device, vector_adsorbate_y, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(vector_adsorbate_z_device, vector_adsorbate_z, sizeof(double)*N_atom_adsorbate[0], cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(temperature_device, &temperature, sizeof(double), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(N_atom_frame_device, N_atom_frame, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(times_x_device, times_x, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(times_y_device, times_y, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(times_z_device, times_z, sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(epsilon_frame_device, epsilon_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(sigma_frame_device, sigma_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(mass_frame_device, mass_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac_a_frame_device, frac_a_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac_b_frame_device, frac_b_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac_c_frame_device, frac_c_frame, sizeof(double)*N_atom_frame[0]*times, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac2car_a_device, frac2car_a, sizeof(double)*3, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac2car_b_device, frac2car_b, sizeof(double)*3, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(frac2car_c_device, frac2car_c, sizeof(double)*3, cudaMemcpyHostToDevice, stream1);
    //check whether data is properly transferred, uncomment only when it is debugging
    // cudaStreamSynchronize(stream1);
    // check_double<<<1,32>>>(1, cart_x_extended_device);
    // check_double<<<1,32>>>(1, cart_y_extended_device);
    // check_double<<<1,32>>>(1, cart_z_extended_device);
    // check_double<<<1,32>>>(1, cutoff_device);
    // check_int<<<1,32>>>(1, N_atom_adsorbate_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], epsilon_adsorbate_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], sigma_adsorbate_device);
    // check_double<<<1,32>>>(1, center_of_mass_x_device);
    // check_double<<<1,32>>>(1, center_of_mass_y_device);
    // check_double<<<1,32>>>(1, center_of_mass_z_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], vector_adsorbate_x_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], vector_adsorbate_y_device);
    // check_double<<<1,32>>>(N_atom_adsorbate[0], vector_adsorbate_z_device);
    // check_double<<<1,32>>>(1, temperature_device);
    // check_int<<<1,32>>>(1, N_atom_frame_device);
    // check_int<<<1,32>>>(1, times_x_device);
    // check_int<<<1,32>>>(1, times_y_device);
    // check_int<<<1,32>>>(1, times_z_device);
    // check_double<<<1,32>>>(N_atom_frame[0]*times, epsilon_frame_device);
    // check_double<<<1,32>>>(N_atom_frame[0]*times, sigma_frame_device);
    // cudaDeviceSynchronize();
    // return 0;
    //------------------------------------GPU Variable allocation and transfer----------------------------------
    //-------------------------------------------------------done-----------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------










    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------------
    //--------------------------------------------Vext calcution on GPU-----------------------------------------

    
    


    




    // calculate the number of grid
    a_N = ceil(1.0*La/dL);
    if (a_N > Nmax_a)
    {
        a_N = Nmax_a;
    }
    b_N = ceil(1.0*Lb/dL);
    if (b_N > Nmax_b)
    {
        b_N = Nmax_b;
    }
    c_N = ceil(1.0*Lc/dL);
    if (c_N > Nmax_c)
    {
        c_N = Nmax_c;
    }
    // printf("%d %d %d\n", a_N, b_N, c_N);
    int cal_a_N, cal_b_N, cal_c_N;
    int cal_alpha_N, cal_beta_N, cal_gamma_N;
    cal_a_N = 1;
    cal_b_N = 1;
    cal_c_N = 1;
    cal_alpha_N = 1;
    cal_beta_N = 1;
    cal_gamma_N = 1;
    double *V_ext_separate_single_batch;
    // double

    while ((cal_a_N*cal_b_N*cal_c_N*cal_alpha_N*cal_beta_N*cal_gamma_N)<=
        (a_N*b_N*c_N*alpha_N*beta_N*gamma_N))
    {
        // printf("rounded a: %d %d %lf %lf\n", a_N, cal_a_N, ceil(1.0*a_N/cal_a_N), 1.0*a_N/cal_a_N);
        if (cudaMalloc((void **)&V_ext_separate_single_batch, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*
            ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N)*40)==0)
        {
            cudaFree(V_ext_separate_single_batch);
            // printf("success angle\n");
            break;
        }
        else
        {
            cudaFree(V_ext_separate_single_batch);

            if (cal_a_N<a_N)
            {
                cal_a_N++;
            }
            else if (cal_b_N<b_N)
            {
                cal_b_N++;
            }
            else if (cal_c_N<c_N)
            {
                cal_c_N++;
            }
            else if (cal_gamma_N<gamma_N)
            {
                cal_gamma_N++;
            }
            else if (cal_beta_N<beta_N)
            {
                cal_beta_N++;
            }
            else if (cal_alpha_N<alpha_N)
            {
                cal_alpha_N++;
            }
            else if ((cal_a_N==a_N)&&(cal_b_N==b_N)&&(cal_c_N==c_N)&&(cal_alpha_N==alpha_N)&&(cal_beta_N==beta_N)&&(cal_gamma_N==gamma_N))
            {
                printf("limited memory issue!!!!!!!!\n");
                printf("failed!!!!!!!!\n");
                return 0;
            }
        }
    }
    // printf("%d %d %d %d %d %d\n", cal_a_N, cal_b_N, cal_c_N, cal_alpha_N, cal_beta_N, cal_gamma_N);



    

    



    


    // double alpha_rad[1], beta_rad[1], gamma_rad[1];

    double loc_a[1], loc_b[1], loc_c[1];



    // define auxilary variables to calculate the external potential
    int *N_a_device, *N_b_device, *N_c_device;
    int *N_alpha_device, *N_beta_device, *N_gamma_device;


    double *loc_a_device, *loc_b_device, *loc_c_device;
    double *a_Vext_cal_device, *b_Vext_cal_device, *c_Vext_cal_device;
    double *rot_alpha_rad_device, *rot_beta_rad_device, *rot_gamma_rad_device;
    
    double *alpha_Vext_cal_device, *beta_Vext_cal_device, *gamma_Vext_cal_device;
    int  *index_a_device, *index_b_device, *index_c_device;
    int *index_alpha_device, *index_beta_device, *index_gamma_device;
    int *index_adsorbate_device, *index_frame_device;
    double *vector_adsorbate_x_rot_device, *vector_adsorbate_y_rot_device, *vector_adsorbate_z_rot_device;
    
    double *loc_x_device, *loc_y_device, *loc_z_device;
    double *adsorbate_cart_x_rot_device, *adsorbate_cart_y_rot_device, *adsorbate_cart_z_rot_device;
    double *modify_frame_a_device, *modify_frame_b_device, *modify_frame_c_device;
    double *minimum_distance_device;
    double *V_temp_device;


    cudaMalloc((void **)&N_a_device, sizeof(int));
    cudaMalloc((void **)&N_b_device, sizeof(int));
    cudaMalloc((void **)&N_c_device, sizeof(int));
    cudaMalloc((void **)&N_alpha_device, sizeof(int));
    cudaMalloc((void **)&N_beta_device, sizeof(int));
    cudaMalloc((void **)&N_gamma_device, sizeof(int));
    cudaMalloc((void **)&loc_a_device, sizeof(double));
    cudaMalloc((void **)&loc_b_device, sizeof(double));
    cudaMalloc((void **)&loc_c_device, sizeof(double));


    size_t free_byte;
    size_t total_byte;
    double free_db;
    double total_db;
    double used_db;

    // cudaMemGetInfo(&free_byte, &total_byte);
    // free_db = (double) free_byte;
    // total_db = (double) total_byte;
    // used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    // cudaDeviceSynchronize();




    cudaMalloc((void **)&a_Vext_cal_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&b_Vext_cal_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&c_Vext_cal_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&rot_alpha_rad_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&rot_beta_rad_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&rot_gamma_rad_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&alpha_Vext_cal_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&beta_Vext_cal_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&gamma_Vext_cal_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_a_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_b_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_c_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_alpha_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_beta_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_gamma_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_adsorbate_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&index_frame_device, sizeof(int)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&loc_x_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&loc_y_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&loc_z_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&vector_adsorbate_x_rot_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&vector_adsorbate_y_rot_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&vector_adsorbate_z_rot_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&adsorbate_cart_x_rot_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&adsorbate_cart_y_rot_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&adsorbate_cart_z_rot_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&modify_frame_a_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&modify_frame_b_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&modify_frame_c_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&minimum_distance_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));
    cudaMalloc((void **)&V_temp_device, sizeof(double)*N_atom_adsorbate[0]*N_atom_frame[0]*times*ceil(1.0*a_N/cal_a_N)*ceil(1.0*b_N/cal_b_N)*ceil(1.0*c_N/cal_c_N)*
            ceil(1.0*alpha_N/cal_alpha_N)*ceil(1.0*beta_N/cal_beta_N)*ceil(1.0*gamma_N/cal_gamma_N));



    double *V_result_device;
    cudaMalloc((void **)&V_result_device, sizeof(double)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    // printf("total points: %ld\n", (long int) a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    // printf("%d %d %d %d %d %d\n", a_N, b_N, c_N, alpha_N, beta_N, gamma_N);



    
    // cudaMemGetInfo(&free_byte, &total_byte);
    // free_db = (double) free_byte;
    // total_db = (double) total_byte;
    // used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    


    // int N1[1], N2[1], N3[1];
    // N1[0] = alpha_N;
    // N2[0] = beta_N;
    // N3[0] = gamma_N;
    cudaMemcpy(N_a_device, &a_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_b_device, &b_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_c_device, &c_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_alpha_device, &alpha_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_beta_device, &beta_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_gamma_device, &gamma_N, sizeof(int), cudaMemcpyHostToDevice);
    // check_int<<<1,32>>>(1, N_a_device);
    // check_int<<<1,32>>>(1, N_b_device);
    // check_int<<<1,32>>>(1, N_c_device);
    // check_int<<<1,32>>>(1, N_alpha_device);
    // check_int<<<1,32>>>(1, N_beta_device);
    // check_int<<<1,32>>>(1, N_gamma_device);



    // cudaDeviceSynchronize();
    // return 0;


    // a_N=1;
    // b_N=0;
    // c_N=0;

    


    int i_a, i_b, i_c, i_alpha, i_beta, i_gamma;





    int N_atom_frame_extend=N_atom_frame[0]*times_x[0]*times_y[0]*times_z[0];

    int *N_atom_frame_extend_device;

    cudaMalloc((void **)&N_atom_frame_extend_device, sizeof(int));

    cudaMemcpy(N_atom_frame_extend_device, &N_atom_frame_extend, sizeof(int), cudaMemcpyHostToDevice);
    

    // printf("%d\n", N_atom_frame_extend);
    // check_int<<<1,32>>>(1, N_atom_frame_extend_device);



    // double a_start, b_start, c_start, a_end, b_end, c_end;
    // double alpha_start_rad, beta_start_rad, gamma_start_rad, alpha_end_rad, beta_end_rad, gamma_end_rad;

    // double *a_start_device, *b_start_device, *c_start_device, *a_end_device, *b_end_device, *c_end_device;
    // double *alpha_start_rad_device, *beta_start_rad_device, *gamma_start_rad_device, *alpha_end_rad_device, *beta_end_rad_device, *gamma_end_rad_device;

    // // cudaMalloc((void **)&, sizeof(double));
    // cudaMalloc((void **)&a_start_device, sizeof(double));
    // cudaMalloc((void **)&b_start_device, sizeof(double));
    // cudaMalloc((void **)&c_start_device, sizeof(double));
    // cudaMalloc((void **)&a_end_device, sizeof(double));
    // cudaMalloc((void **)&b_end_device, sizeof(double));
    // cudaMalloc((void **)&c_end_device, sizeof(double));
    // cudaMalloc((void **)&alpha_start_rad_device, sizeof(double));
    // cudaMalloc((void **)&beta_start_rad_device, sizeof(double));
    // cudaMalloc((void **)&gamma_start_rad_device, sizeof(double));
    // cudaMalloc((void **)&alpha_end_rad_device, sizeof(double));
    // cudaMalloc((void **)&beta_end_rad_device, sizeof(double));
    // cudaMalloc((void **)&gamma_end_rad_device, sizeof(double));

    // cudaMemcpy(a_start_device, &a_start, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(b_start_device, &b_start, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_start_device, &c_start, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(a_end_device, &a_end, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(b_end_device, &b_end, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_end_device, &c_end, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(alpha_start_rad_device, &alpha_start_rad, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(beta_start_rad_device, &beta_start_rad, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(gamma_start_rad_device, &gamma_start_rad, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(alpha_end_rad_device, &alpha_end_rad, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(beta_end_rad_device, &beta_end_rad, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(gamma_end_rad_device, &gamma_end_rad, sizeof(int), cudaMemcpyHostToDevice);

    int index_a_start, index_b_start, index_c_start, index_a_end, index_b_end, index_c_end;
    int index_alpha_start, index_beta_start, index_gamma_start, index_alpha_end, index_beta_end, index_gamma_end;

    int *index_a_start_device, *index_b_start_device, *index_c_start_device, *index_a_end_device, *index_b_end_device, *index_c_end_device;
    int *index_alpha_start_device, *index_beta_start_device, *index_gamma_start_device, *index_alpha_end_device, *index_beta_end_device, *index_gamma_end_device;

    cudaMalloc((void **)&index_a_start_device, sizeof(int));
    cudaMalloc((void **)&index_b_start_device, sizeof(int));
    cudaMalloc((void **)&index_c_start_device, sizeof(int));
    cudaMalloc((void **)&index_a_end_device, sizeof(int));
    cudaMalloc((void **)&index_b_end_device, sizeof(int));
    cudaMalloc((void **)&index_c_end_device, sizeof(int));
    cudaMalloc((void **)&index_alpha_start_device, sizeof(int));
    cudaMalloc((void **)&index_beta_start_device, sizeof(int));
    cudaMalloc((void **)&index_gamma_start_device, sizeof(int));
    cudaMalloc((void **)&index_alpha_end_device, sizeof(int));
    cudaMalloc((void **)&index_beta_end_device, sizeof(int));
    cudaMalloc((void **)&index_gamma_end_device, sizeof(int));


    // printf("round test: %lf\n", floor(1.0*a_N/cal_a_N));

    // define variables for sum
    long long int num_segments;
    int *h_offset, *d_offset;
    double *V_sum_temp_device;
    void *d_temp_storage;
    size_t temp_storage_bytes = 0;




    
    long long int num_total_segments = 0;
    int block_size;
    // int copy_start_index;



    // cudaError_t s;
    // const char *errorMessage;
    // for both position and angle, the end point is not included due to trapzoid rule
    for (i_a=1; i_a<=cal_a_N; i_a++)
    {

        index_a_start = (i_a-1)*ceil(1.0*a_N/cal_a_N);
        index_a_end = i_a*ceil(1.0*a_N/cal_a_N) - 1;
        if (index_a_end>(a_N-1))
        {
            index_a_end = (a_N-1);
        }
        for (i_b=1; i_b<=cal_b_N; i_b++)
        {
            index_b_start = (i_b-1)*ceil(1.0*b_N/cal_b_N);
            index_b_end = i_b*ceil(1.0*b_N/cal_b_N) - 1;
            if (index_b_end>(b_N-1))
            {
                index_b_end = (b_N-1);
            }
            for (i_c=1; i_c<=cal_c_N; i_c++)
            {
                index_c_start = (i_c-1)*ceil(1.0*c_N/cal_c_N);
                index_c_end = i_c*ceil(1.0*c_N/cal_c_N) - 1;
                if (index_c_end>(c_N-1))
                {
                    index_c_end = (c_N-1);
                }
                for (i_alpha=1; i_alpha<=cal_alpha_N; i_alpha++)
                {   
                    index_alpha_start = (i_alpha-1)*ceil(1.0*alpha_N/cal_alpha_N);
                    index_alpha_end = i_alpha*ceil(1.0*alpha_N/cal_alpha_N) - 1;
                    if (index_alpha_end>(alpha_N-1))
                    {
                        index_alpha_end = (alpha_N-1);
                    }
                    for (i_beta=1; i_beta<=cal_beta_N; i_beta++)
                    {
                        index_beta_start = (i_beta-1)*ceil(1.0*beta_N/cal_beta_N);
                        index_beta_end = i_beta*ceil(1.0*beta_N/cal_beta_N) - 1;
                        if (index_beta_end>(beta_N-1))
                        {
                            index_beta_end = (beta_N-1);
                        }
                        for (i_gamma=1; i_gamma<=cal_gamma_N; i_gamma++)
                        {
                            index_gamma_start = (i_gamma-1)*ceil(1.0*gamma_N/cal_gamma_N);
                            index_gamma_end = i_gamma*ceil(1.0*gamma_N/cal_gamma_N) - 1;
                            if (index_gamma_end>(gamma_N-1))
                            {
                                index_gamma_end = (gamma_N-1);
                            }

                            cudaMemcpy(index_a_start_device, &index_a_start, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_b_start_device, &index_b_start, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_c_start_device, &index_c_start, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_a_end_device, &index_a_end, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_b_end_device, &index_b_end, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_c_end_device, &index_c_end, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_alpha_start_device, &index_alpha_start, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_beta_start_device, &index_beta_start, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_gamma_start_device, &index_gamma_start, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_alpha_end_device, &index_alpha_end, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_beta_end_device, &index_beta_end, sizeof(int), cudaMemcpyHostToDevice);
                            cudaMemcpy(index_gamma_end_device, &index_gamma_end, sizeof(int), cudaMemcpyHostToDevice);

                            



                            // printf("%d %d %d %d %d %d %d %d %d %d %d %d\n", index_a_start, index_a_end, index_b_start, index_b_end, index_c_start, index_c_end, 
                            //     index_alpha_start, index_alpha_end, index_beta_start, index_beta_end, index_gamma_start, index_gamma_end);




                            if ((((index_a_end-index_a_start+1)*(index_b_end-index_b_start+1)*(index_c_end-index_c_start+1)
                                *(index_alpha_end-index_alpha_start+1)*(index_beta_end-index_beta_start+1)*(index_gamma_end-index_gamma_start+1)
                                *N_atom_adsorbate[0]*N_atom_frame[0]*times-1)/running_block_size+1)>running_grid_size)
                            {
                                block_size = running_grid_size;
                            }
                            else
                            {
                                block_size = (((index_a_end-index_a_start+1)*(index_b_end-index_b_start+1)*(index_c_end-index_c_start+1)
                                *(index_alpha_end-index_alpha_start+1)*(index_beta_end-index_beta_start+1)*(index_gamma_end-index_gamma_start+1)
                                *N_atom_adsorbate[0]*N_atom_frame[0]*times-1)/running_block_size+1);
                            }

                            // printf("%d %d %d %d %d %d %d %d %d\n", (index_a_end-index_a_start+1), (index_b_end-index_b_start+1), (index_c_end-index_c_start+1), 
                            //     (index_alpha_end-index_alpha_start+1), (index_beta_end-index_beta_start+1), (index_gamma_end-index_gamma_start+1), 
                            //     N_atom_adsorbate[0], N_atom_frame[0], times);

                            // printf("grid size: %d\n", block_size);
                            // printf("%d\n", );

            //                 cudaDeviceSynchronize();
                            // check_int_custom2<<<(int)((1-1)/running_block_size+1),running_block_size>>>
                            // (1, index_a_start_device, index_a_end_device, index_b_start_device, index_b_end_device, index_c_start_device, index_c_end_device, 
                            //     index_alpha_start_device, index_alpha_end_device, index_beta_start_device, index_beta_end_device, index_gamma_start_device, index_gamma_end_device);



                            

                            V_batch<<<block_size,running_block_size>>>
                            (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
                            vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                            N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                            frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                            times_x_device, times_y_device, times_z_device,
                            N_atom_frame_extend_device,
                            cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                            frac2car_a_device, frac2car_b_device, frac2car_c_device,
                            cutoff_device,

                            index_a_start_device, index_b_start_device, index_c_start_device,
                            index_a_end_device, index_b_end_device, index_c_end_device,
                            index_alpha_start_device, index_beta_start_device, index_gamma_start_device,
                            index_alpha_end_device, index_beta_end_device, index_gamma_end_device, 

                            N_a_device, N_b_device, N_c_device,
                            N_alpha_device, N_beta_device, N_gamma_device,

                            index_a_device, index_b_device, index_c_device,
                            index_alpha_device, index_beta_device, index_gamma_device,
                            index_adsorbate_device, index_frame_device,

                            a_Vext_cal_device, b_Vext_cal_device, c_Vext_cal_device,
                            rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device,
                            loc_x_device, loc_y_device, loc_z_device,
                            vector_adsorbate_x_rot_device, vector_adsorbate_y_rot_device, vector_adsorbate_z_rot_device,
                            adsorbate_cart_x_rot_device, adsorbate_cart_y_rot_device, adsorbate_cart_z_rot_device, 
                            modify_frame_a_device, modify_frame_b_device, modify_frame_c_device,
                            minimum_distance_device,
                            V_temp_device);


















                            // sum interatmoic interaction into molecule based
                            num_segments = (long long int) (index_a_end-index_a_start+1)*(index_b_end-index_b_start+1)*(index_c_end-index_c_start+1)
                                *(index_alpha_end-index_alpha_start+1)*(index_beta_end-index_beta_start+1)*(index_gamma_end-index_gamma_start+1);
                            h_offset = (int *) malloc(sizeof(int)*(num_segments+1));
                            cudaMalloc((void**)&d_offset, (num_segments+1)*sizeof(int));
                            cudaMalloc((void**)&V_sum_temp_device, sizeof(double)*num_segments);

                            h_offset[0] = 0;
                            for (j=0; j<=num_segments; j++)
                            {
                                h_offset[j] = j*N_atom_adsorbate[0]*N_atom_frame[0]*times;
                            }
                            cudaMemcpy(d_offset, h_offset, (num_segments+1)*sizeof(int), cudaMemcpyHostToDevice);

                            d_temp_storage = NULL;
                            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_temp_device, V_sum_temp_device, 
                                num_segments, d_offset, d_offset+1);
                            cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
                            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_temp_device, V_sum_temp_device, 
                                num_segments, d_offset, d_offset+1);

                            cudaFree(d_temp_storage);
                            cudaFree(d_offset);
                            free(h_offset);





                            // cudaDeviceSynchronize();
                            // check_double_k<<<1,32>>>(V_result_device);
                            // cudaDeviceSynchronize();
                            // check_double_sci<<<1,32>>>(5, V_sum_temp_device);
                            // cudaDeviceSynchronize();
                            // return 0;
                            // printf("\n");
                            // cudaMemGetInfo(&free_byte, &total_byte);
                            // free_db = (double) free_byte;
                            // total_db = (double) total_byte;
                            // used_db = total_db - free_db;
                            // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
                            //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);





                            cudaMemcpy(V_result_device+num_total_segments, V_sum_temp_device, num_segments*sizeof(double), cudaMemcpyDeviceToDevice);
                            num_total_segments = num_total_segments + num_segments;





                            // s = cudaGetLastError();
                            // errorMessage = cudaGetErrorString(s);
                            // printf("%s\n", errorMessage);

                            // printf("total_seg: %d %d\n", num_total_segments, num_segments);

                           




                            // cudaFree(V_sum_temp_device);
                            // return 0;
                        }
                    }
                }
            }
        }
    }
    // printf("%d\n", a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    // double *Vext_print;
    // Vext_print = (double *) malloc(sizeof(double)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    // printf("check: %d\n", a_N*b_N*c_N*alpha_N*beta_N*gamma_N);

    // cudaMemcpy(Vext_print, V_result_device, sizeof(double)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N, cudaMemcpyDeviceToHost);
    // for (i=0; i<a_N*b_N*c_N*alpha_N*beta_N*gamma_N; i++)
    // {
    //     printf("%lf\n", Vext_print[i]);
    // }

    // cudaMemGetInfo(&free_byte, &total_byte);
    // free_db = (double) free_byte;
    // total_db = (double) total_byte;
    // used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    // check_double<<<1,32>>>(474552, V_result_device);
    // cudaDeviceSynchronize();

    // t = clock() - t;
    // printf("%d\t", N_atom_adsorbate[0]);
    // printf("%d\t", N_atom_frame[0]);
    // printf("%d\t", times);
    // printf("%d\t", alpha_N);
    // printf("%d\t", beta_N);
    // printf("%d\t", gamma_N);
    // printf("%d\t", a_N);
    // printf("%d\t", b_N);
    // printf("%d\t", c_N);
    // printf("%d %lf\n", N_atom_adsorbate[0]*N_atom_frame[0]*times*alpha_N*beta_N*gamma_N*a_N*b_N*c_N, ((double)t)/CLOCKS_PER_SEC);

    cudaFree(a_Vext_cal_device);
    cudaFree(b_Vext_cal_device);
    cudaFree(c_Vext_cal_device);
    cudaFree(rot_alpha_rad_device);
    cudaFree(rot_beta_rad_device);
    cudaFree(rot_gamma_rad_device);
    cudaFree(alpha_Vext_cal_device);
    cudaFree(beta_Vext_cal_device);
    cudaFree(gamma_Vext_cal_device);
    cudaFree(index_a_device);
    cudaFree(index_b_device);
    cudaFree(index_c_device);
    cudaFree(index_alpha_device);
    cudaFree(index_beta_device);
    cudaFree(index_gamma_device);
    cudaFree(index_adsorbate_device);
    cudaFree(index_frame_device);
    cudaFree(loc_x_device);
    cudaFree(loc_y_device);
    cudaFree(loc_z_device);
    cudaFree(vector_adsorbate_x_rot_device);
    cudaFree(vector_adsorbate_y_rot_device);
    cudaFree(vector_adsorbate_z_rot_device);
    cudaFree(adsorbate_cart_x_rot_device);
    cudaFree(adsorbate_cart_y_rot_device);
    cudaFree(adsorbate_cart_z_rot_device);
    cudaFree(modify_frame_a_device);
    cudaFree(modify_frame_b_device);
    cudaFree(modify_frame_c_device);
    cudaFree(minimum_distance_device);
    cudaFree(V_temp_device);
    cudaDeviceSynchronize();



    // cudaMemGetInfo(&free_byte, &total_byte);
    // free_db = (double) free_byte;
    // total_db = (double) total_byte;
    // used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);



    cudaMalloc((void **)&index_a_device, sizeof(int)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    cudaMalloc((void **)&index_b_device, sizeof(int)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    cudaMalloc((void **)&index_c_device, sizeof(int)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    cudaMalloc((void **)&index_alpha_device, sizeof(int)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    cudaMalloc((void **)&index_beta_device, sizeof(int)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    cudaMalloc((void **)&index_gamma_device, sizeof(int)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    cudaMalloc((void **)&rot_beta_rad_device, sizeof(double)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);

    double *H_segment_device, *H_device;
    cudaMalloc((void **)&H_segment_device, sizeof(double)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    cudaMalloc((void **)&H_device, sizeof(double));


    // cudaMemGetInfo(&free_byte, &total_byte);
    // free_db = (double) free_byte;
    // total_db = (double) total_byte;
    // used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);







    // printf("%d\n", ((a_N*b_N*c_N*alpha_N*beta_N*gamma_N)/running_block_size+1));




    block_size = ((a_N*b_N*c_N*alpha_N*beta_N*gamma_N)/running_block_size+1);
    if (((a_N*b_N*c_N*alpha_N*beta_N*gamma_N)/running_block_size+1)>running_grid_size)
    {
        block_size = running_grid_size;
    }
    else
    {
        block_size = ((a_N*b_N*c_N*alpha_N*beta_N*gamma_N)/running_block_size+1);
    }

    //                 cudaDeviceSynchronize();
    // check_int_custom2<<<(int)((1-1)/running_block_size+1),running_block_size>>>
    // (1, index_a_start_device, index_a_end_device, index_b_start_device, index_b_end_device, index_c_start_device, index_c_end_device, 
    //     index_alpha_start_device, index_alpha_end_device, index_beta_start_device, index_beta_end_device, index_gamma_start_device, index_gamma_end_device);





    cal_H_seg_from_Vext<<<block_size,running_block_size>>>
    (V_result_device, 
        N_a_device, N_b_device, N_c_device,
        N_alpha_device, N_beta_device, N_gamma_device, 
        
        index_a_device, index_b_device, index_c_device,
        index_alpha_device, index_beta_device, index_gamma_device,

        rot_beta_rad_device, temperature_device,

        H_segment_device);

    // cudaDeviceSynchronize();

    // check_double<<<1,32>>>(1, H_segment_device);

    // cudaDeviceSynchronize();



    long long int *H_sum_offset = (long long int *) malloc(sizeof(int)*(2));
    H_sum_offset[0] = 0;
    H_sum_offset[1] = (long long int) a_N*b_N*c_N*alpha_N*beta_N*gamma_N;
    long long int *H_sum_offset_device;
    cudaMalloc((void**)&H_sum_offset_device, (2)*sizeof(long long int));
    cudaMemcpy(H_sum_offset_device, H_sum_offset, (2)*sizeof(long long int), cudaMemcpyHostToDevice);
    // printf("host: %lld\n", H_sum_offset[1]);
    free(H_sum_offset);
    
    // check_long_int<<<1,32>>>(2, H_sum_offset_device);
    // check_double<<<1,32>>>(1, H_segment_device);
    // check_int<<<1,32>>>(1, N_atom_frame_device);

    d_temp_storage = NULL;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, H_segment_device, H_device, 
        1, H_sum_offset_device, H_sum_offset_device+1);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, H_segment_device, H_device, 
        1, H_sum_offset_device, H_sum_offset_device+1);

    // check_double<<<1,32>>>(1, H_device);


    // double *temp;
    // temp = (double *) malloc(sizeof(double)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N);
    // cudaMemcpy(temp, H_segment_device, sizeof(double)*a_N*b_N*c_N*alpha_N*beta_N*gamma_N, cudaMemcpyDeviceToHost);
    // check_double_sci<<<1,32>>>(1944, H_segment_device);
    // for (i=0; i<a_N*b_N*c_N*alpha_N*beta_N*gamma_N; i++)
    // {
    //     printf("%.5e\n", temp[i]);
    // }

    





    // cudaDeviceSynchronize();
    // cudaMemGetInfo(&free_byte, &total_byte);
    // free_db = (double) free_byte;
    // total_db = (double) total_byte;
    // used_db = total_db - free_db;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //         used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);




    cudaDeviceSynchronize();
    double H_print;
    cudaMemcpy(&H_print, H_device, sizeof(double), cudaMemcpyDeviceToHost);

    t = clock() - t;

    fp1 = fopen(argv[2], "w+");
    fprintf(fp1, "%lld %lf %lf\n", (long long int) N_atom_adsorbate[0]*N_atom_frame[0]*times*alpha_N*beta_N*gamma_N*a_N*b_N*c_N, H_print, ((double)t)/CLOCKS_PER_SEC);
    fclose(fp1);
    // printf("%lld %lf %lf\n", (long long int) N_atom_adsorbate[0]*N_atom_frame[0]*times*alpha_N*beta_N*gamma_N*a_N*b_N*c_N, H_print, ((double)t)/CLOCKS_PER_SEC);
    // printf("%lld %lf %lf\n", (long long int) alpha_N*beta_N*gamma_N*a_N*b_N*c_N, H_print, ((double)t)/CLOCKS_PER_SEC);
    return 0;

}
