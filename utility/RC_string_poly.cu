#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#define PI 3.141592653589793
#define running_block_size 32




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
                printf("%d\n", x[i]);
            }
        }
    }
}

__global__
void check_int_custom(int n, int *x1, int *x2, int *x3, int *x4, int *x5, int *x6, int *x7, int *x8)
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
                printf("%d %d %d %d %d %d %d %d\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i]);
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
void check_double_ini(int n, double *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (j=0; j<(n/3); j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                // printf("index: %d\t%lf\n", i, x[i]);
                printf("%lf\t%lf\t%lf\n", x[3*i+0], x[3*i+1], x[3*i+2]);
            }
        }
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
void check_double_custom(int n, double *x1, double *x2)
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
                // printf("%.3e\t%.3e\n", x1[i], x2[i]);
                printf("%lf %lf\n", x1[i], x2[i]);
            }
        }
    }
}

__global__
void check_double_custom2(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6)
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
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i]);
            }
        }
    }
}

__global__
void check_double_custom22(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6)
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
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i], x4[i]-PI, x5[i], x6[i]);
            }
        }
    }
}

__global__
void check_double_custom3(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6)
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
                printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i], x4[i]/2/PI*360, x5[i]/2/PI*360, x6[i]/2/PI*360);
            }
        }
    }
}


__global__
void check_double_custom4(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6, double *x7)
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
                // printf("%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.3e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]);
                // printf("%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.3e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]);
                printf("%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.3e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i]);
            }
        }
    }
}


__global__
void check_double_custom5(int n, double *x1, double *x2, double *x3, double *x4, double *x5, double *x6, 
    double *y1, double *y2, double *y3, double *y4, double *y5, double *y6)
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
                printf("%.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], 
                    y1[i], y2[i], y3[i], y4[i], y5[i], y6[i]);
            }
        }
    }
}


__global__
void check_double_custom6(int n, double *x1, double *x2)
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
                printf("%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n", x1[i*3+0], x1[i*3+1], x1[i*3+2], 
                    x2[i*3+0], x2[i*3+1], x2[i*3+2]);
            }
        }
    }
}


__global__
void check_double_custom7(int n, double *x1, double *x2)
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
                printf("%.3e %.3e %.3e %.3e\n", x1[i*2], 
                    x2[i*6+0], x2[i*6+1], x2[i*6+2]);
            }
        }
    }
}

__global__
void check_double_follow(double *x1, double *x2, double *x3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    if (index==4)
    {
         printf("%lf\t%lf\t%lf\n", x1[3], x2[3], x3[3]);
    }
}



__global__
void check_double_temp(int n, double *x1, double *x2, double *x3)
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
                printf("%lf\t%lf\t%lf\n", x1[i], x2[i], x3[i]);
            }
        }
    }
}

__global__
void check_double_angle1(int n, double *x1, double *x2, double *x3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (i=index; i<n; i+=stride)
    {
        if ((i==399))
        {
            // printf("index: %d\t%lf\n", i, x[i]);
            printf("%d\t%lf\t%lf\t%lf\n", i, x1[i], x2[i], x3[i]);
        }
    }
}

__global__
void check_double_angle2(int n, double *x1, double *x2, double *x3)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (i=index; i<n; i+=stride)
    {
        if ((i==5))
        {
            // printf("index: %d\t%lf\n", i, x[i]);
            printf("%d\t%lf\t%lf\t%lf\n", i, x1[i], x2[i], x3[i]);
        }
    }
}

__global__
void check_double_angle3(int n, double *x1, double *x2, double *x3, double *x4)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j;
    for (i=index; i<n; i+=stride)
    {
        if ((i==399))
        {
            // printf("index: %d\t%lf\n", i, x[i]);
            printf("%d\t%lf\t%lf\t%lf\t%lf\n", i, x1[i], x2[i], x3[i], x4[i]);
        }
    }
}





__global__
void check_key(int n, cub::KeyValuePair<int, double> *x)
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
                printf("key: %d\tvalue: %lf\n", x[i].key, x[i].value);
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







void rotate_moleucle(int *N_atom_adsorbate, double vector_adsorbate_x[], double vector_adsorbate_y[], double vector_adsorbate_z[], 
                        double rot_alpha_angle, double rot_beta_angle, double rot_gamma_angle, double vector_adsorbate_x_rot[], 
                        double vector_adsorbate_y_rot[], double vector_adsorbate_z_rot[])
{
	double rot_alpha_rad, rot_beta_rad, rot_gamma_rad;
	int i, ii, index;

	//convert angles to rad
	rot_alpha_rad = rot_alpha_angle/180*PI;
    rot_beta_rad = rot_beta_angle/180*PI;
    rot_gamma_rad = rot_gamma_angle/180*PI;

	for (i=0; i<N_atom_adsorbate[0]; i++)
	{
		vector_adsorbate_x_rot[i] = vector_adsorbate_x[i]*cos(rot_gamma_rad)*cos(rot_beta_rad) 
			- vector_adsorbate_y[i]*sin(rot_gamma_rad)*cos(rot_alpha_rad) + vector_adsorbate_y[i]*sin(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad) 
			+ vector_adsorbate_z[i]*sin(rot_gamma_rad)*sin(rot_alpha_rad) + vector_adsorbate_z[i]*cos(rot_alpha_rad)*cos(rot_gamma_rad)*sin(rot_beta_rad);

		vector_adsorbate_y_rot[i] = vector_adsorbate_x[i]*sin(rot_gamma_rad)*cos(rot_beta_rad) 
			+ vector_adsorbate_y[i]*cos(rot_gamma_rad)*cos(rot_alpha_rad) + vector_adsorbate_y[i]*sin(rot_gamma_rad)*sin(rot_beta_rad)*sin(rot_alpha_rad)
			- vector_adsorbate_z[i]*cos(rot_gamma_rad)*sin(rot_alpha_rad) + vector_adsorbate_z[i]*sin(rot_gamma_rad)*sin(rot_beta_rad)*cos(rot_alpha_rad);

		vector_adsorbate_z_rot[i] = -vector_adsorbate_x[i]*sin(rot_beta_rad) + vector_adsorbate_y[i]*cos(rot_beta_rad)*sin(rot_alpha_rad) 
			+ vector_adsorbate_z[i]*cos(rot_alpha_rad)*cos(rot_beta_rad);
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












// solution 3 for GPU Vext on the initial plane
__global__
void Vext_cal_3(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *cal_a_device, double *cal_b_device, double *cal_c_device,
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
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        index_a_device[i] = 0;
        index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame) );
        index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
        index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
        index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_adsorbate_device[i]*temp_add_frame) );

        cal_a_device[i] = 0;
        cal_b_device[i] = index_b_device[i]*delta_grid_device[0];
        cal_c_device[i] = index_c_device[i]*delta_grid_device[0];

        rot_alpha_rad_device[i] = index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = index_gamma_device[i]*delta_angle_device[0]/180*PI;

        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);


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
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0]);
        }
        
    }

}



// upgrade solution 3 for GPU Vext on the initial plane
__global__
void Vext_cal_3_upgrade(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,

                int *direction_device, 


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *cal_a_device, double *cal_b_device, double *cal_c_device,
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
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                index_a_device[i] = 0;
                index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 2:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = 0;
                index_c_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 3:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = 0;
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
        }

        cal_a_device[i] = index_a_device[i]*delta_grid_device[0];
        cal_b_device[i] = index_b_device[i]*delta_grid_device[0];
        cal_c_device[i] = index_c_device[i]*delta_grid_device[0];

        rot_alpha_rad_device[i] = index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = index_gamma_device[i]*delta_angle_device[0]/180*PI;

        // if (i==2137136)
        // {
        //     printf("direction_device: %d\n", direction_device[0]);
        //     printf("%d %d %d\n", index_a_device[i], index_b_device[i], index_c_device[i]);
        //     printf("cal_device: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("%d %d %d\n", index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        //     printf("cal_device: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }

        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);


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
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0]);
        }
        
    }

}








//
__global__
void ini_string_1(int *N_string_device, double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 
    int *temp_add_frame_device, int *N_atom_adsorbate_device,


    int *direction_device, 
    cub::KeyValuePair<int, double> *min_value_index_device, double *s0_a_device, double *s0_b_device, double *s0_c_device,
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int i, j, jj;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<N_string_device[0]; i+=stride)
    {
        switch (direction_device[0])
        {
            // initialize string along the x-axis direction
            case 1:
                s0_a_device[i] = 1.0*i/(N_string_device[0]-1);
                s0_b_device[i] = cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_c_device[i] = cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_alpha_device[i] = rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_beta_device[i] = rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_gamma_device[i] = rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 2:
                s0_a_device[i] = cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_b_device[i] = 1.0*i/(N_string_device[0]-1);
                s0_c_device[i] = cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_alpha_device[i] = rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_beta_device[i] = rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_gamma_device[i] = rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 3:
                s0_a_device[i] = cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_b_device[i] = cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_c_device[i] = 1.0*i/(N_string_device[0]-1);
                s0_alpha_device[i] = rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_beta_device[i] = rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                s0_gamma_device[i] = rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
        }
        
        
    }

}


// copy the initila and the last point
__global__
void copy_ini(double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 

    cub::KeyValuePair<int, double> *min_value_index_device, int *temp_add_frame_device, int *N_atom_adsorbate_device,
    int *num_inidividual_ini_extra_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(2*6); i+=stride)
    {
        switch (i%6)
        {
            case 0:
                ini_minimum_string_a_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = (i/6);
                break;
            case 1:
                ini_minimum_string_b_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 2:
                ini_minimum_string_c_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 3:
                ini_minimum_string_alpha_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 4:
                ini_minimum_string_beta_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 5:
                ini_minimum_string_gamma_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
        }
        
        
    }

}



// copy the initila and the last point
__global__
void copy_ini_upgrade(double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    int *direction_device,

    double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 

    cub::KeyValuePair<int, double> *min_value_index_device, int *temp_add_frame_device, int *N_atom_adsorbate_device,
    int *num_inidividual_ini_extra_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(2*6); i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = (i/6);
                        break;
                    case 1:
                        ini_minimum_string_b_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
            case 2:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 1:
                        ini_minimum_string_b_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = (i/6);
                        break;
                    case 2:
                        ini_minimum_string_c_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;

            case 3:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 1:
                        ini_minimum_string_b_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = (i/6);
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[(i/6)*num_inidividual_ini_extra_device[0]+(i/6)] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
        }
    }

}



// copy the middle points on the initial string 
__global__
void copy_ini_middle(double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 

    int *i_cal_device,

    cub::KeyValuePair<int, double> *min_value_index_device, int *temp_add_frame_device, int *N_atom_adsorbate_device,
    int *num_inidividual_ini_extra_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(1*6); i+=stride)
    {
        switch (i%6)
        {
            case 0:
                ini_minimum_string_a_device[i_cal_device[0]] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                break;
            case 1:
                ini_minimum_string_b_device[i_cal_device[0]] = 
                cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 2:
                ini_minimum_string_c_device[i_cal_device[0]] = 
                cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 3:
                ini_minimum_string_alpha_device[i_cal_device[0]] = 
                rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 4:
                ini_minimum_string_beta_device[i_cal_device[0]] = 
                rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
            case 5:
                ini_minimum_string_gamma_device[i_cal_device[0]] = 
                rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                break;
        }
        
        
    }

}






// copy the middle points on the initial string 
__global__
void copy_ini_middle_upgrade(double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    int *direction_device,

    double *cal_a_device, double *cal_b_device, double *cal_c_device, 
    double *rot_alpha_rad_device, double *rot_beta_rad_device, double *rot_gamma_rad_device, 

    int *i_cal_device,

    cub::KeyValuePair<int, double> *min_value_index_device, int *temp_add_frame_device, int *N_atom_adsorbate_device,
    int *num_inidividual_ini_extra_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(1*6); i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[i_cal_device[0]] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                        break;
                    case 1:
                        ini_minimum_string_b_device[i_cal_device[0]] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[i_cal_device[0]] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[i_cal_device[0]] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[i_cal_device[0]] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[i_cal_device[0]] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
            case 2:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[i_cal_device[0]] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];;
                        break;
                    case 1:
                        ini_minimum_string_b_device[i_cal_device[0]] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                        break;
                    case 2:
                        ini_minimum_string_c_device[i_cal_device[0]] = 
                        cal_c_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[i_cal_device[0]] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[i_cal_device[0]] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[i_cal_device[0]] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;

            case 3:
                switch (i%6)
                {
                    case 0:
                        ini_minimum_string_a_device[i_cal_device[0]] = 
                        cal_a_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 1:
                        ini_minimum_string_b_device[i_cal_device[0]] = 
                        cal_b_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 2:
                        ini_minimum_string_c_device[i_cal_device[0]] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                        break;
                    case 3:
                        ini_minimum_string_alpha_device[i_cal_device[0]] = 
                        rot_alpha_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 4:
                        ini_minimum_string_beta_device[i_cal_device[0]] = 
                        rot_beta_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                    case 5:
                        ini_minimum_string_gamma_device[i_cal_device[0]] = 
                        rot_gamma_rad_device[min_value_index_device[0].key*temp_add_frame_device[0]*N_atom_adsorbate_device[0]];
                        break;
                }
                break;
        }
        
        
        
    }

}



// solution for GPU Vext on the initial string
__global__
void Vext_cal_ini(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *limit_transition_frac_device, double *limit_rotation_angle_device,
                double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
                double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device,

                int *i_cal_device, int *num_inidividual_ini_extra_device,



                double *cal_a_device, double *cal_b_device, double *cal_c_device,
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
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        index_a_device[i] = 0;
        index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame) );
        index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
        index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
        index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_adsorbate_device[i]*temp_add_frame) );



        // if (i==0)
        // {
        //     printf("1basic: %d %d %d %d %d %d\n", N_grid_device[0]*N_grid_device[0], N_angle_alpha_device[0], N_angle_beta_device[0], N_angle_gamma_device[0], 
        //         N_atom_adsorbate_device[0], temp_add_frame);
        // }


        
        

        cal_a_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
        cal_b_device[i] = ini_minimum_string_b_device[0] - limit_transition_frac_device[0] + index_b_device[i]*delta_grid_device[0];
        cal_c_device[i] = ini_minimum_string_c_device[0] - limit_transition_frac_device[0] + index_c_device[i]*delta_grid_device[0];

        rot_alpha_rad_device[i] = ini_minimum_string_alpha_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = ini_minimum_string_beta_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = ini_minimum_string_gamma_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_gamma_device[i]*delta_angle_device[0]/180*PI;



        // if (i==0)
        // {
        //     printf("check ori: %lf %lf\n", ini_minimum_string_b_device[0], ini_minimum_string_c_device[0]);
        //     printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("%lf %lf %lf %lf\n", ini_minimum_string_b_device[0], limit_transition_frac_device[0], index_b_device[i], delta_grid_device[0]);
        //     printf("alpha: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_alpha_device[0], rot_alpha_rad_device[i], index_alpha_device[i]);
        //     printf("beta:  %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_beta_device[0], rot_beta_rad_device[i], index_beta_device[i]);
        //     printf("gamma: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_gamma_device[0], rot_gamma_rad_device[i], index_gamma_device[i]);
        // }


        // if (i==0)
        // {
            // printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
            // printf("check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }

        // if (i==(0*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("1check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("0check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(5*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("5check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(1372*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("1372check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(7*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     printf("7check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("7check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }






        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);


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

        // if (i==0)
        // {
        //     printf("xyz: %lf %lf %lf\n", loc_x_device[i], loc_y_device[i], loc_z_device[i]);
        //     printf("rotate: %lf %lf %lf\n", vector_adsorbate_x_rot_device[i], vector_adsorbate_y_rot_device[i], vector_adsorbate_z_rot_device[i]);
        // }






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
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0]);
        }
        
        // if (i==0)
        // {
        //     printf("Vext1: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==1)
        // {
        //     printf("Vext2: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==2)
        // {
        //     printf("Vext3: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }

        // if (i==106)
        // {
        //     printf("Vext 107: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
    }

}







// solution for GPU Vext on the initial string
__global__
void Vext_cal_ini_upgrade(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,


                int *direction_device,


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *limit_transition_frac_device, double *limit_rotation_angle_device,
                double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
                double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device,

                int *i_cal_device, int *num_inidividual_ini_extra_device,



                double *cal_a_device, double *cal_b_device, double *cal_c_device,
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
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                index_a_device[i] = 0;
                index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 2:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = 0;
                index_c_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
            case 3:
                index_a_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame) );
                index_b_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_c_device[i] = 0;
                index_alpha_device[i] = (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_beta_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
                    /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
                index_gamma_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
                index_adsorbate_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
                index_frame_device[i] =  (int) ( (i-index_a_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
                    *N_atom_adsorbate_device[0]*temp_add_frame
                    - index_b_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
                    - index_adsorbate_device[i]*temp_add_frame) );
                break;
        }
        // index_a_device[i] = 0;
        // index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
        //     *N_atom_adsorbate_device[0]*temp_add_frame) );
        // index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
        //     *N_atom_adsorbate_device[0]*temp_add_frame)
        //     /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        // index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
        //     *N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
        //     /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        // index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
        //     *N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
        //     /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        // index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
        //     *N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
        // index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
        //     *N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
        // index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
        //     *N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
        //     - index_adsorbate_device[i]*temp_add_frame) );



        // if (i==0)
        // {
        //     printf("1basic: %d %d %d %d %d %d\n", N_grid_device[0]*N_grid_device[0], N_angle_alpha_device[0], N_angle_beta_device[0], N_angle_gamma_device[0], 
        //         N_atom_adsorbate_device[0], temp_add_frame);
        // }

        switch (direction_device[0])
        {
            case 1:
                cal_a_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                cal_b_device[i] = ini_minimum_string_b_device[0] - limit_transition_frac_device[0] + index_b_device[i]*delta_grid_device[0];
                cal_c_device[i] = ini_minimum_string_c_device[0] - limit_transition_frac_device[0] + index_c_device[i]*delta_grid_device[0];
                break;
            case 2:
                cal_a_device[i] = ini_minimum_string_a_device[0] - limit_transition_frac_device[0] + index_a_device[i]*delta_grid_device[0];
                cal_b_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                cal_c_device[i] = ini_minimum_string_c_device[0] - limit_transition_frac_device[0] + index_c_device[i]*delta_grid_device[0];
                break;
            case 3:
                cal_a_device[i] = ini_minimum_string_a_device[0] - limit_transition_frac_device[0] + index_a_device[i]*delta_grid_device[0];
                cal_b_device[i] = ini_minimum_string_b_device[0] - limit_transition_frac_device[0] + index_b_device[i]*delta_grid_device[0];
                cal_c_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
                break;
        }


        
        

        

        rot_alpha_rad_device[i] = ini_minimum_string_alpha_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = ini_minimum_string_beta_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = ini_minimum_string_gamma_device[0] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_gamma_device[i]*delta_angle_device[0]/180*PI;



        // if (i==0)
        // {
        //     printf("check ori: %lf %lf\n", ini_minimum_string_b_device[0], ini_minimum_string_c_device[0]);
        //     printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("%lf %lf %lf %lf\n", ini_minimum_string_b_device[0], limit_transition_frac_device[0], index_b_device[i], delta_grid_device[0]);
        //     printf("alpha: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_alpha_device[0], rot_alpha_rad_device[i], index_alpha_device[i]);
        //     printf("beta:  %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_beta_device[0], rot_beta_rad_device[i], index_beta_device[i]);
        //     printf("gamma: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_gamma_device[0], rot_gamma_rad_device[i], index_gamma_device[i]);
        // }


        // if (i==0)
        // {
            // printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
            // printf("check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }

        // if (i==(0*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("1check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("0check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(5*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("5check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(1372*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("1372check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(7*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     printf("7check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("7check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }






        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);


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

        // if (i==0)
        // {
        //     printf("xyz: %lf %lf %lf\n", loc_x_device[i], loc_y_device[i], loc_z_device[i]);
        //     printf("rotate: %lf %lf %lf\n", vector_adsorbate_x_rot_device[i], vector_adsorbate_y_rot_device[i], vector_adsorbate_z_rot_device[i]);
        // }






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
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0]);
        }
        
        // if (i==0)
        // {
        //     printf("Vext1: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==1)
        // {
        //     printf("Vext2: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==2)
        // {
        //     printf("Vext3: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }

        // if (i==106)
        // {
        //     printf("Vext 107: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
    }

}




// solution for GPU Vext on the initial string
__global__
void Vext_cal_ini_forward(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *limit_transition_frac_device, double *limit_rotation_angle_device,
                double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
                double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device,

                int *i_cal_device, int *num_inidividual_ini_extra_device,



                double *cal_a_device, double *cal_b_device, double *cal_c_device,
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
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        index_a_device[i] = 0;
        index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame) );
        index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
        index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
        index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_adsorbate_device[i]*temp_add_frame) );



        // if (i==0)
        // {
        //     printf("1basic: %d %d %d %d %d %d\n", N_grid_device[0]*N_grid_device[0], N_angle_alpha_device[0], N_angle_beta_device[0], N_angle_gamma_device[0], 
        //         N_atom_adsorbate_device[0], temp_add_frame);
        // }


        
        

        cal_a_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
        cal_b_device[i] = ini_minimum_string_b_device[i_cal_device[0]-1] - limit_transition_frac_device[0] + index_b_device[i]*delta_grid_device[0];
        cal_c_device[i] = ini_minimum_string_c_device[i_cal_device[0]-1] - limit_transition_frac_device[0] + index_c_device[i]*delta_grid_device[0];

        rot_alpha_rad_device[i] = ini_minimum_string_alpha_device[i_cal_device[0]-1] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = ini_minimum_string_beta_device[i_cal_device[0]-1] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = ini_minimum_string_gamma_device[i_cal_device[0]-1] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_gamma_device[i]*delta_angle_device[0]/180*PI;



        // if (i==0)
        // {
        //     printf("check ori: %lf %lf\n", ini_minimum_string_b_device[0], ini_minimum_string_c_device[0]);
        //     printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("%lf %lf %lf %lf\n", ini_minimum_string_b_device[0], limit_transition_frac_device[0], index_b_device[i], delta_grid_device[0]);
        //     printf("alpha: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_alpha_device[0], rot_alpha_rad_device[i], index_alpha_device[i]);
        //     printf("beta:  %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_beta_device[0], rot_beta_rad_device[i], index_beta_device[i]);
        //     printf("gamma: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_gamma_device[0], rot_gamma_rad_device[i], index_gamma_device[i]);
        // }


        // if (i==0)
        // {
            // printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
            // printf("check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }

        // if (i==(0*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("1check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("0check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(5*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("5check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(1372*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("1372check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(7*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     printf("7check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("7check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }






        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);


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

        // if (i==0)
        // {
        //     printf("xyz: %lf %lf %lf\n", loc_x_device[i], loc_y_device[i], loc_z_device[i]);
        //     printf("rotate: %lf %lf %lf\n", vector_adsorbate_x_rot_device[i], vector_adsorbate_y_rot_device[i], vector_adsorbate_z_rot_device[i]);
        // }






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
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0]);
        }
        
        // if (i==0)
        // {
        //     printf("Vext1: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==1)
        // {
        //     printf("Vext2: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==2)
        // {
        //     printf("Vext3: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }

        // if (i==106)
        // {
        //     printf("Vext 107: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
    }

}





// solution for GPU Vext on the initial string
__global__
void Vext_cal_ini_backward(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,


                int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                double *delta_grid_device, double *delta_angle_device,
                int *index_a_device, int *index_b_device, int *index_c_device,
                int *index_alpha_device, int *index_beta_device, int *index_gamma_device,
                int *index_adsorbate_device, int *index_frame_device,

                double *limit_transition_frac_device, double *limit_rotation_angle_device,
                double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
                double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device,

                int *i_cal_device, int *num_inidividual_ini_extra_device,



                double *cal_a_device, double *cal_b_device, double *cal_c_device,
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
    int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; 
        i<( N_grid_device[0]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame); 
        i+=stride)
    {
        index_a_device[i] = 0;
        index_b_device[i] = (int) ( (i)/(N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame) );
        index_c_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_alpha_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_beta_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame)
            /(N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) );
        index_gamma_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame) / (N_atom_adsorbate_device[0]*temp_add_frame) );
        index_adsorbate_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame) / (temp_add_frame) );
        index_frame_device[i] = (int) ( (i-index_b_device[i]*N_grid_device[0]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]
            *N_atom_adsorbate_device[0]*temp_add_frame
            - index_c_device[i]*N_angle_alpha_device[0]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_alpha_device[i]*N_angle_beta_device[0]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_beta_device[i]*N_angle_gamma_device[0]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_gamma_device[i]*N_atom_adsorbate_device[0]*temp_add_frame
            - index_adsorbate_device[i]*temp_add_frame) );



        // if (i==0)
        // {
        //     printf("1basic: %d %d %d %d %d %d\n", N_grid_device[0]*N_grid_device[0], N_angle_alpha_device[0], N_angle_beta_device[0], N_angle_gamma_device[0], 
        //         N_atom_adsorbate_device[0], temp_add_frame);
        // }


        
        

        cal_a_device[i] = 1.0*i_cal_device[0]/(num_inidividual_ini_extra_device[0]+1);
        cal_b_device[i] = ini_minimum_string_b_device[num_inidividual_ini_extra_device[0]+1] - limit_transition_frac_device[0] + index_b_device[i]*delta_grid_device[0];
        cal_c_device[i] = ini_minimum_string_c_device[num_inidividual_ini_extra_device[0]+1] - limit_transition_frac_device[0] + index_c_device[i]*delta_grid_device[0];

        rot_alpha_rad_device[i] = ini_minimum_string_alpha_device[num_inidividual_ini_extra_device[0]+1] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_alpha_device[i]*delta_angle_device[0]/180*PI;
        rot_beta_rad_device[i] = ini_minimum_string_beta_device[num_inidividual_ini_extra_device[0]+1] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_beta_device[i]*delta_angle_device[0]/180*PI;
        rot_gamma_rad_device[i] = ini_minimum_string_gamma_device[num_inidividual_ini_extra_device[0]+1] - 1.0*limit_rotation_angle_device[0]/180*PI + 1.0*index_gamma_device[i]*delta_angle_device[0]/180*PI;


        // rot_alpha_rad_device[i] = ini_minimum_string_alpha_device[num_inidividual_ini_extra_device[0]+1] + 1.0*index_alpha_device[i]*delta_angle_device[0]/180*PI;
        // rot_beta_rad_device[i] = ini_minimum_string_beta_device[num_inidividual_ini_extra_device[0]+1] + 1.0*index_beta_device[i]*delta_angle_device[0]/180*PI;
        // rot_gamma_rad_device[i] = ini_minimum_string_gamma_device[num_inidividual_ini_extra_device[0]+1] + 1.0*index_gamma_device[i]*delta_angle_device[0]/180*PI;




        // if (i==0)
        // {
        //     printf("check ori: %lf %lf\n", ini_minimum_string_b_device[0], ini_minimum_string_c_device[0]);
        //     printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("%lf %lf %lf %lf\n", ini_minimum_string_b_device[0], limit_transition_frac_device[0], index_b_device[i], delta_grid_device[0]);
        //     printf("alpha: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_alpha_device[0], rot_alpha_rad_device[i], index_alpha_device[i]);
        //     printf("beta:  %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_beta_device[0], rot_beta_rad_device[i], index_beta_device[i]);
        //     printf("gamma: %lf %lf %lf %lf\n", limit_rotation_angle_device[0], ini_minimum_string_gamma_device[0], rot_gamma_rad_device[i], index_gamma_device[i]);
        // }


        // if (i==0)
        // {
            // printf("check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
            // printf("check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }

        // if (i==(0*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("1check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("0check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(5*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("5check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(1372*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     // printf("6check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("1372check cal_alpha beta gamma: %lf %lf %lf\n index: %d %d %d\n", 
        //         rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i], 
        //         index_alpha_device[i], index_beta_device[i], index_gamma_device[i]);
        // }
        // if (i==(7*N_atom_adsorbate_device[0]*temp_add_frame))
        // {
        //     printf("7check cal_a b c: %lf %lf %lf\n", cal_a_device[i], cal_b_device[i], cal_c_device[i]);
        //     printf("7check cal_alpha beta gamma: %lf %lf %lf\n", rot_alpha_rad_device[i], rot_beta_rad_device[i], rot_gamma_rad_device[i]);
        // }






        loc_x_device[i] = frac2car_x_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_a_device);
        loc_y_device[i] = frac2car_y_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_b_device);
        loc_z_device[i] = frac2car_z_device(cal_a_device[i], cal_b_device[i], cal_c_device[i], frac2car_c_device);


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

        // if (i==0)
        // {
        //     printf("xyz: %lf %lf %lf\n", loc_x_device[i], loc_y_device[i], loc_z_device[i]);
        //     printf("rotate: %lf %lf %lf\n", vector_adsorbate_x_rot_device[i], vector_adsorbate_y_rot_device[i], vector_adsorbate_z_rot_device[i]);
        // }






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
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_device[i]]*epsilon_frame_device[index_frame_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_device[i]]+sigma_frame_device[index_frame_device[i]])*0.5), cutoff_device[0]);
        }
        
        // if (i==0)
        // {
        //     printf("Vext1: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==1)
        // {
        //     printf("Vext2: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
        // if (i==2)
        // {
        //     printf("Vext3: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }

        // if (i==106)
        // {
        //     printf("Vext 107: %lf %lf\n", minimum_distance_device[i], V_result_device[i]);
        // }
    }

}




__global__
void s1_frac2cart_ini(int *num_inidividual_ini_extra_device, 

    double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,

    double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device,
    double *ini_minimum_string_cart_x_device, double *ini_minimum_string_cart_y_device, double *ini_minimum_string_cart_z_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<((num_inidividual_ini_extra_device[0]+2)*3); i+=stride)
    {
        switch (i%3)
        {
            case 0:
                ini_minimum_string_cart_x_device[(i/3)] = 
                frac2car_x_device(ini_minimum_string_a_device[(i/3)], ini_minimum_string_b_device[(i/3)], ini_minimum_string_c_device[(i/3)], frac2car_a_device);
                break;
            case 1:
                ini_minimum_string_cart_y_device[(i/3)] = 
                frac2car_y_device(ini_minimum_string_a_device[(i/3)], ini_minimum_string_b_device[(i/3)], ini_minimum_string_c_device[(i/3)], frac2car_b_device);
                break;
            case 2:
                ini_minimum_string_cart_z_device[(i/3)] = 
                frac2car_z_device(ini_minimum_string_a_device[(i/3)], ini_minimum_string_b_device[(i/3)], ini_minimum_string_c_device[(i/3)], frac2car_c_device);
                break;
        }
    }
}



__global__
void cal_length_prep_ini(int *num_inidividual_ini_extra_device, 

    double *ini_minimum_string_cart_x_device, double *ini_minimum_string_cart_y_device, double *ini_minimum_string_cart_z_device,
    double *ini_minimum_length_all_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<((num_inidividual_ini_extra_device[0]+2)*3); i+=stride)
    {
        // for the first point
        if (((int) i/3)==0)
        {
            ini_minimum_length_all_device[i] = 0;
        }
        else
        {
            switch (((int) i%3))
            {
                case 0:
                    ini_minimum_length_all_device[i] = pow((ini_minimum_string_cart_x_device[(i/3)]-ini_minimum_string_cart_x_device[((i/3)-1)]),2);
                    break;
                case 1:
                    ini_minimum_length_all_device[i] = pow((ini_minimum_string_cart_y_device[(i/3)]-ini_minimum_string_cart_y_device[((i/3)-1)]),2);
                    break;
                case 2:
                    ini_minimum_length_all_device[i] = pow((ini_minimum_string_cart_z_device[(i/3)]-ini_minimum_string_cart_z_device[((i/3)-1)]),2);
                    break;
            }
        }

        

    }
}



__global__
void ini_length_sqrt_cal(int *num_inidividual_ini_extra_device, double *ini_minimum_length_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(num_inidividual_ini_extra_device[0]+2); i+=stride)
    {
        ini_minimum_length_device[i] = sqrt(ini_minimum_length_device[i]);
    }

}






__global__
void ini_2_s0(int *num_inidividual_ini_extra_device, int *N_string_device, 


    double *ini_minimum_l_abs_device, 

    double *ini_minimum_string_a_device, double *ini_minimum_string_b_device, double *ini_minimum_string_c_device, 
    double *ini_minimum_string_alpha_device, double *ini_minimum_string_beta_device, double *ini_minimum_string_gamma_device, 

    double *temp_partition_device, double *ini_minimum_length_device, 

    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        temp_partition_device[i] = 0;
        switch ((i%6))
        {
            case 0:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {

                        // if ((i/6) == 400)
                        // {
                        //     printf("check: %d %lf %lf %lf\n", ii, (1.0*(i/6)/(N_string_device[0]-1)), temp_partition_device[i], ini_minimum_length_device[ii+1]);
                        // }
                        

                        s0_a_device[(i/6)] = ini_minimum_string_a_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_a_device[ii+1]-ini_minimum_string_a_device[ii]);
                        break;
                    }
                    else
                    {
                        // if ((i/6) == 400)
                        // {
                        //     printf("check2: %lf %lf\n", (1.0*(i/6)/(N_string_device[0]-1)), (temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])));
                        // }
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }

                break;
            case 1:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_b_device[(i/6)] = ini_minimum_string_b_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_b_device[ii+1]-ini_minimum_string_b_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 2:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_c_device[(i/6)] = ini_minimum_string_c_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_c_device[ii+1]-ini_minimum_string_c_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 3:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        // if ((i/6) == 400)
                        // {
                        //     printf("check: %d %lf %lf %lf\n", ii, (1.0*(i/6)/(N_string_device[0]-1)), temp_partition_device[i], ini_minimum_length_device[ii+1]);
                        // }



                        s0_alpha_device[(i/6)] = ini_minimum_string_alpha_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_alpha_device[ii+1]-ini_minimum_string_alpha_device[ii]);
                        break;
                    }
                    else
                    {
                        // if ((i/6) == 400)
                        // {
                        //     printf("check2: %lf %lf\n", (1.0*(i/6)/(N_string_device[0]-1)), (temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])));
                        // }

                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 4:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_beta_device[(i/6)] = ini_minimum_string_beta_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_beta_device[ii+1]-ini_minimum_string_beta_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;
            case 5:
                for (ii=0; ii<(num_inidividual_ini_extra_device[0]+1); ii++)
                {
                    if ( ((1.0*(i/6)/(N_string_device[0]-1))>=temp_partition_device[i]) && 
                        (((1.0*(i/6)/(N_string_device[0]-1))<=(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0]))) || 
                        (fabs((1.0*(i/6)/(N_string_device[0]-1))-(temp_partition_device[i]+(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])))<=1e-3)) )
                    {
                        s0_gamma_device[(i/6)] = ini_minimum_string_gamma_device[ii] + (1.0*((1.0*(i/6)/(N_string_device[0]-1)) - 
                            temp_partition_device[i])/(1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0])) * 
                        (ini_minimum_string_gamma_device[ii+1]-ini_minimum_string_gamma_device[ii]);
                        break;
                    }
                    else
                    {
                        temp_partition_device[i] = temp_partition_device[i] + 1.0*ini_minimum_length_device[ii+1]/ini_minimum_l_abs_device[0];
                    }
                }
                break;

        }
    }
}


















__global__
void Vext_cal_s0(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,
                int *temp_add_frame_device,


                // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                // double *delta_grid_device, double *delta_angle_device,
                int *N_string_device,

                double *s0_a_device, double *s0_b_device, double *s0_c_device, 
                double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,


                int *index_s0_cal_Vext_s0_device,
                // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                int *index_adsorbate_cal_Vext_s0_device, int *index_frame_cal_Vext_s0_device,

                double *a_cal_Vext_s0_device, double *b_cal_Vext_s0_device, double *c_cal_Vext_s0_device,
                double *alpha_rad_cal_Vext_s0_device, double *beta_rad_cal_Vext_s0_device, double *gamma_rad_cal_Vext_s0_device,
                double *loc_x_cal_Vext_s0_device, double *loc_y_cal_Vext_s0_device, double *loc_z_cal_Vext_s0_device,
                double *vector_adsorbate_x_rot_cal_Vext_s0_device, double *vector_adsorbate_y_rot_cal_Vext_s0_device, double *vector_adsorbate_z_rot_cal_Vext_s0_device,
                double *adsorbate_cart_x_rot_cal_Vext_s0_device, double *adsorbate_cart_y_rot_cal_Vext_s0_device, double *adsorbate_cart_z_rot_cal_Vext_s0_device, 
                double *modify_frame_a_cal_Vext_s0_device, double *modify_frame_b_cal_Vext_s0_device, double *modify_frame_c_cal_Vext_s0_device,
                double *minimum_distance_cal_Vext_s0_device,
                double *V_result_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<(N_string_device[0]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]); i+=stride)
    {
        index_s0_cal_Vext_s0_device[i] = (int) ( (i)/(N_atom_adsorbate_device[0]*temp_add_frame_device[0]) );
        index_adsorbate_cal_Vext_s0_device[i] = (int) ( (i-index_s0_cal_Vext_s0_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0])
            /(temp_add_frame_device[0]) );
        index_frame_cal_Vext_s0_device[i] = (int) ( (i - index_s0_cal_Vext_s0_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -index_adsorbate_cal_Vext_s0_device[i]*temp_add_frame_device[0]) );

        a_cal_Vext_s0_device[i] = s0_a_device[index_s0_cal_Vext_s0_device[i]];
        b_cal_Vext_s0_device[i] = s0_b_device[index_s0_cal_Vext_s0_device[i]];
        c_cal_Vext_s0_device[i] = s0_c_device[index_s0_cal_Vext_s0_device[i]];

        alpha_rad_cal_Vext_s0_device[i] = s0_alpha_device[index_s0_cal_Vext_s0_device[i]];
        beta_rad_cal_Vext_s0_device[i] = s0_beta_device[index_s0_cal_Vext_s0_device[i]];
        gamma_rad_cal_Vext_s0_device[i] = s0_gamma_device[index_s0_cal_Vext_s0_device[i]];


        loc_x_cal_Vext_s0_device[i] = frac2car_x_device(a_cal_Vext_s0_device[i], b_cal_Vext_s0_device[i], c_cal_Vext_s0_device[i], frac2car_a_device);
        loc_y_cal_Vext_s0_device[i] = frac2car_y_device(a_cal_Vext_s0_device[i], b_cal_Vext_s0_device[i], c_cal_Vext_s0_device[i], frac2car_b_device);
        loc_z_cal_Vext_s0_device[i] = frac2car_z_device(a_cal_Vext_s0_device[i], b_cal_Vext_s0_device[i], c_cal_Vext_s0_device[i], frac2car_c_device);


        vector_adsorbate_x_rot_cal_Vext_s0_device[i] = rotate_moleucle_x_device(alpha_rad_cal_Vext_s0_device[i], beta_rad_cal_Vext_s0_device[i], gamma_rad_cal_Vext_s0_device[i],
            vector_adsorbate_x_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_y_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_z_device[index_adsorbate_cal_Vext_s0_device[i]]);
        vector_adsorbate_y_rot_cal_Vext_s0_device[i] = rotate_moleucle_y_device(alpha_rad_cal_Vext_s0_device[i], beta_rad_cal_Vext_s0_device[i], gamma_rad_cal_Vext_s0_device[i],
            vector_adsorbate_x_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_y_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_z_device[index_adsorbate_cal_Vext_s0_device[i]]);
        vector_adsorbate_z_rot_cal_Vext_s0_device[i] = rotate_moleucle_z_device(alpha_rad_cal_Vext_s0_device[i], beta_rad_cal_Vext_s0_device[i], gamma_rad_cal_Vext_s0_device[i],
            vector_adsorbate_x_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_y_device[index_adsorbate_cal_Vext_s0_device[i]], vector_adsorbate_z_device[index_adsorbate_cal_Vext_s0_device[i]]);

        adsorbate_cart_x_rot_cal_Vext_s0_device[i] = loc_x_cal_Vext_s0_device[i]+vector_adsorbate_x_rot_cal_Vext_s0_device[i];
        adsorbate_cart_y_rot_cal_Vext_s0_device[i] = loc_y_cal_Vext_s0_device[i]+vector_adsorbate_y_rot_cal_Vext_s0_device[i];
        adsorbate_cart_z_rot_cal_Vext_s0_device[i] = loc_z_cal_Vext_s0_device[i]+vector_adsorbate_z_rot_cal_Vext_s0_device[i];

        V_result_device[i] = 0;






        // z-direction
        if ( (adsorbate_cart_z_rot_cal_Vext_s0_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_c_device)) 
            > 0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_cal_Vext_s0_device[i] = frac_c_frame_device[index_frame_cal_Vext_s0_device[i]] + times_z_device[0];
        }
        else if ( (adsorbate_cart_z_rot_cal_Vext_s0_device[i] 
            - frac2car_z_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_c_device)) 
            < -0.5*cart_z_extended_device[0] )
        {
            modify_frame_c_cal_Vext_s0_device[i] = frac_c_frame_device[index_frame_cal_Vext_s0_device[i]] - times_z_device[0];
        }
        else
        {
            modify_frame_c_cal_Vext_s0_device[i] = frac_c_frame_device[index_frame_cal_Vext_s0_device[i]];
        }



        // y-direction
        if ( (adsorbate_cart_y_rot_cal_Vext_s0_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_b_device)) 
            > 0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_cal_Vext_s0_device[i] = frac_b_frame_device[index_frame_cal_Vext_s0_device[i]] + times_y_device[0];
        }
        else if ( (adsorbate_cart_y_rot_cal_Vext_s0_device[i] 
            - frac2car_y_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_b_device)) 
            < -0.5*cart_y_extended_device[0] )
        {
            modify_frame_b_cal_Vext_s0_device[i] = frac_b_frame_device[index_frame_cal_Vext_s0_device[i]] - times_y_device[0];
        }
        else
        {
            modify_frame_b_cal_Vext_s0_device[i] = frac_b_frame_device[index_frame_cal_Vext_s0_device[i]];
        }

        // x-direction
        if ( (adsorbate_cart_x_rot_cal_Vext_s0_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_a_device)) 
            > 0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_cal_Vext_s0_device[i] = frac_a_frame_device[index_frame_cal_Vext_s0_device[i]] + times_x_device[0];
        }
        else if ( (adsorbate_cart_x_rot_cal_Vext_s0_device[i] 
            - frac2car_x_device(frac_a_frame_device[index_frame_cal_Vext_s0_device[i]], frac_b_frame_device[index_frame_cal_Vext_s0_device[i]], 
                frac_c_frame_device[index_frame_cal_Vext_s0_device[i]], frac2car_a_device)) 
            < -0.5*cart_x_extended_device[0] )
        {
            modify_frame_a_cal_Vext_s0_device[i] = frac_a_frame_device[index_frame_cal_Vext_s0_device[i]] - times_x_device[0];
        }
        else
        {
            modify_frame_a_cal_Vext_s0_device[i] = frac_a_frame_device[index_frame_cal_Vext_s0_device[i]];
        }

        minimum_distance_cal_Vext_s0_device[i] = cal_dis_device(adsorbate_cart_x_rot_cal_Vext_s0_device[i], adsorbate_cart_y_rot_cal_Vext_s0_device[i], adsorbate_cart_z_rot_cal_Vext_s0_device[i], 
                    frac2car_x_device(modify_frame_a_cal_Vext_s0_device[i], modify_frame_b_cal_Vext_s0_device[i], modify_frame_c_cal_Vext_s0_device[i], frac2car_a_device), 
                    frac2car_y_device(modify_frame_a_cal_Vext_s0_device[i], modify_frame_b_cal_Vext_s0_device[i], modify_frame_c_cal_Vext_s0_device[i], frac2car_b_device), 
                    frac2car_z_device(modify_frame_a_cal_Vext_s0_device[i], modify_frame_b_cal_Vext_s0_device[i], modify_frame_c_cal_Vext_s0_device[i], frac2car_c_device));

        if (minimum_distance_cal_Vext_s0_device[i] < 
            (((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5)*0.1))
        {
           minimum_distance_cal_Vext_s0_device[i] = (((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5)*0.1);
        }

        if (minimum_distance_cal_Vext_s0_device[i] < cutoff_device[0])
        {
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]*epsilon_frame_device[index_frame_cal_Vext_s0_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5), minimum_distance_cal_Vext_s0_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]*epsilon_frame_device[index_frame_cal_Vext_s0_device[i]]), 
                ((sigma_adsorbate_device[index_adsorbate_cal_Vext_s0_device[i]]+sigma_frame_device[index_frame_cal_Vext_s0_device[i]])*0.5), cutoff_device[0]);
        }
        
    }

}









__global__
void remap_string_var(int *N_atom_adsorbate_device, int *temp_add_frame_device,

                int *N_string_device,

                double *s0_a_device, double *s0_b_device, double *s0_c_device,
                double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,



                double *s0_deri_a_device, double *s0_deri_b_device, double *s0_deri_c_device, 
                double *s0_deri_alpha_device, double *s0_deri_beta_device, double *s0_deri_gamma_device,

                int *s0_deri_index_string_device, int *s0_deri_index_var_device,
                int *s0_deri_index_adsorbate_device, int *s0_deri_index_frame_device,


                double *move_angle_rad_device, double *move_frac_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<(N_string_device[0]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]*7); i+=stride)
    {
        s0_deri_index_string_device[i] = (int) ( (i)/(7*N_atom_adsorbate_device[0]*temp_add_frame_device[0]) );
        s0_deri_index_var_device[i] = (int) ( (i-s0_deri_index_string_device[i]*7*N_atom_adsorbate_device[0]*temp_add_frame_device[0])
            /(N_atom_adsorbate_device[0]*temp_add_frame_device[0]) );
        s0_deri_index_adsorbate_device[i] = (int) ( (i-s0_deri_index_string_device[i]*7*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -s0_deri_index_var_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0])
            /(temp_add_frame_device[0]) );
        s0_deri_index_frame_device[i] = (int) ( (i-s0_deri_index_string_device[i]*7*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -s0_deri_index_var_device[i]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]
            -s0_deri_index_adsorbate_device[i]*temp_add_frame_device[0]) );

        s0_deri_a_device[i] = s0_a_device[s0_deri_index_string_device[i]];
        s0_deri_b_device[i] = s0_b_device[s0_deri_index_string_device[i]];
        s0_deri_c_device[i] = s0_c_device[s0_deri_index_string_device[i]];
        s0_deri_alpha_device[i] = s0_alpha_device[s0_deri_index_string_device[i]];
        s0_deri_beta_device[i] = s0_beta_device[s0_deri_index_string_device[i]];
        s0_deri_gamma_device[i] = s0_gamma_device[s0_deri_index_string_device[i]];

        // modify the variable as the derivative requests
        switch (s0_deri_index_var_device[i])
        {
            case 1:
                s0_deri_a_device[i] = s0_deri_a_device[i] + move_frac_device[0];
                break;
            case 2:
                s0_deri_b_device[i] = s0_deri_b_device[i] + move_frac_device[0];
                break;
            case 3:
                s0_deri_c_device[i] = s0_deri_c_device[i] + move_frac_device[0];
                break;
            case 4:
                s0_deri_alpha_device[i] = s0_deri_alpha_device[i] + move_angle_rad_device[0];
                break;
            case 5:
                s0_deri_beta_device[i] = s0_deri_beta_device[i] + move_angle_rad_device[0];
                break;
            case 6:
                s0_deri_gamma_device[i] = s0_deri_gamma_device[i] + move_angle_rad_device[0];
                break;

        }
    }

}









__global__
void Vext_s0_deri_cal(int *N_atom_adsorbate_device, double *epsilon_adsorbate_device, double *sigma_adsorbate_device,
    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,
                int *N_atom_frame_device, double *epsilon_frame_device, double *sigma_frame_device, 
                double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device,
                int *times_x_device, int *times_y_device, int *times_z_device,
                double *cart_x_extended_device, double *cart_y_extended_device, double *cart_z_extended_device,
                double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,
                double *cutoff_device,
                int *temp_add_frame_device,

                int *N_string_device,

                double *s0_deri_a_device, double *s0_deri_b_device, double *s0_deri_c_device,
                double *s0_deri_alpha_device, double *s0_deri_beta_device, double *s0_deri_gamma_device,

                int *s0_deri_index_adsorbate_device, int *s0_deri_index_frame_device,

                double *s0_deri_loc_x_device, double *s0_deri_loc_y_device, double *s0_deri_loc_z_device,
                double *s0_deri_vector_adsorbate_x_rot_device, double *s0_deri_vector_adsorbate_y_rot_device, double *s0_deri_vector_adsorbate_z_rot_device,
                double *s0_deri_adsorbate_cart_x_rot_device, double *s0_deri_adsorbate_cart_y_rot_device, double *s0_deri_adsorbate_cart_z_rot_device,
                double *s0_deri_modify_frame_a_device, double *s0_deri_modify_frame_b_device, double *s0_deri_modify_frame_c_device,
                double *s0_deri_minimum_distance_device,

                double *V_result_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    // int temp_add_frame = N_atom_frame_device[0]*times_x_device[0]*times_y_device[0]*times_z_device[0];
    

    for (i=index; i<(N_string_device[0]*N_atom_adsorbate_device[0]*temp_add_frame_device[0]*7); i+=stride)
    {
        s0_deri_loc_x_device[i] = frac2car_x_device(s0_deri_a_device[i], s0_deri_b_device[i], s0_deri_c_device[i], frac2car_a_device);
        s0_deri_loc_y_device[i] = frac2car_y_device(s0_deri_a_device[i], s0_deri_b_device[i], s0_deri_c_device[i], frac2car_b_device);
        s0_deri_loc_z_device[i] = frac2car_z_device(s0_deri_a_device[i], s0_deri_b_device[i], s0_deri_c_device[i], frac2car_c_device);



        s0_deri_vector_adsorbate_x_rot_device[i] = rotate_moleucle_x_device(s0_deri_alpha_device[i], s0_deri_beta_device[i], s0_deri_gamma_device[i],
            vector_adsorbate_x_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_y_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_z_device[s0_deri_index_adsorbate_device[i]]);
        s0_deri_vector_adsorbate_y_rot_device[i] = rotate_moleucle_y_device(s0_deri_alpha_device[i], s0_deri_beta_device[i], s0_deri_gamma_device[i],
            vector_adsorbate_x_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_y_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_z_device[s0_deri_index_adsorbate_device[i]]);
        s0_deri_vector_adsorbate_z_rot_device[i] = rotate_moleucle_z_device(s0_deri_alpha_device[i], s0_deri_beta_device[i], s0_deri_gamma_device[i],
            vector_adsorbate_x_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_y_device[s0_deri_index_adsorbate_device[i]], vector_adsorbate_z_device[s0_deri_index_adsorbate_device[i]]);

        s0_deri_adsorbate_cart_x_rot_device[i] = s0_deri_loc_x_device[i]+s0_deri_vector_adsorbate_x_rot_device[i];
        s0_deri_adsorbate_cart_y_rot_device[i] = s0_deri_loc_y_device[i]+s0_deri_vector_adsorbate_y_rot_device[i];
        s0_deri_adsorbate_cart_z_rot_device[i] = s0_deri_loc_z_device[i]+s0_deri_vector_adsorbate_z_rot_device[i];

        V_result_device[i] = 0;






        // z-direction
        if ( (s0_deri_adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_c_device)) 
            > 0.5*cart_z_extended_device[0] )
        {
            s0_deri_modify_frame_c_device[i] = frac_c_frame_device[s0_deri_index_frame_device[i]] + times_z_device[0];
        }
        else if ( (s0_deri_adsorbate_cart_z_rot_device[i] 
            - frac2car_z_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_c_device)) 
            < -0.5*cart_z_extended_device[0] )
        {
            s0_deri_modify_frame_c_device[i] = frac_c_frame_device[s0_deri_index_frame_device[i]] - times_z_device[0];
        }
        else
        {
            s0_deri_modify_frame_c_device[i] = frac_c_frame_device[s0_deri_index_frame_device[i]];
        }



        // y-direction
        if ( (s0_deri_adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_b_device)) 
            > 0.5*cart_y_extended_device[0] )
        {
            s0_deri_modify_frame_b_device[i] = frac_b_frame_device[s0_deri_index_frame_device[i]] + times_y_device[0];
        }
        else if ( (s0_deri_adsorbate_cart_y_rot_device[i] 
            - frac2car_y_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_b_device)) 
            < -0.5*cart_y_extended_device[0] )
        {
            s0_deri_modify_frame_b_device[i] = frac_b_frame_device[s0_deri_index_frame_device[i]] - times_y_device[0];
        }
        else
        {
            s0_deri_modify_frame_b_device[i] = frac_b_frame_device[s0_deri_index_frame_device[i]];
        }



        // x-direction
        if ( (s0_deri_adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_a_device)) 
            > 0.5*cart_x_extended_device[0] )
        {
            s0_deri_modify_frame_a_device[i] = frac_a_frame_device[s0_deri_index_frame_device[i]] + times_x_device[0];
        }
        else if ( (s0_deri_adsorbate_cart_x_rot_device[i] 
            - frac2car_x_device(frac_a_frame_device[s0_deri_index_frame_device[i]], frac_b_frame_device[s0_deri_index_frame_device[i]], 
                frac_c_frame_device[s0_deri_index_frame_device[i]], frac2car_a_device)) 
            < -0.5*cart_x_extended_device[0] )
        {
            s0_deri_modify_frame_a_device[i] = frac_a_frame_device[s0_deri_index_frame_device[i]] - times_x_device[0];
        }
        else
        {
            s0_deri_modify_frame_a_device[i] = frac_a_frame_device[s0_deri_index_frame_device[i]];
        }



        s0_deri_minimum_distance_device[i] = cal_dis_device(s0_deri_adsorbate_cart_x_rot_device[i], s0_deri_adsorbate_cart_y_rot_device[i], s0_deri_adsorbate_cart_z_rot_device[i], 
                    frac2car_x_device(s0_deri_modify_frame_a_device[i], s0_deri_modify_frame_b_device[i], s0_deri_modify_frame_c_device[i], frac2car_a_device), 
                    frac2car_y_device(s0_deri_modify_frame_a_device[i], s0_deri_modify_frame_b_device[i], s0_deri_modify_frame_c_device[i], frac2car_b_device), 
                    frac2car_z_device(s0_deri_modify_frame_a_device[i], s0_deri_modify_frame_b_device[i], s0_deri_modify_frame_c_device[i], frac2car_c_device));



        if (s0_deri_minimum_distance_device[i] < 
            (((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5)*0.1))
        {
           s0_deri_minimum_distance_device[i] = (((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5)*0.1);
        }

        if (s0_deri_minimum_distance_device[i] < cutoff_device[0])
        {
            V_result_device[i] = V_result_device[i] 
            + cal_pure_lj_device(sqrt(epsilon_adsorbate_device[s0_deri_index_adsorbate_device[i]]*epsilon_frame_device[s0_deri_index_frame_device[i]]), 
                ((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5), s0_deri_minimum_distance_device[i]) 
            - cal_pure_lj_device(sqrt(epsilon_adsorbate_device[s0_deri_index_adsorbate_device[i]]*epsilon_frame_device[s0_deri_index_frame_device[i]]), 
                ((sigma_adsorbate_device[s0_deri_index_adsorbate_device[i]]+sigma_frame_device[s0_deri_index_frame_device[i]])*0.5), cutoff_device[0]);
        }
        
    }

}

__global__
void s0_grad_cal(double *move_frac_device, double *move_angle_rad_device, double *rounding_coeff_device,
    int *N_string_device, double *s0_deri_Vext_device, double *s0_gradient_device, double *s0_gradient_square_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {


        s0_gradient_device[i] = s0_deri_Vext_device[(((int) (i%6))+1 + ((int) (i/6))*7)]-s0_deri_Vext_device[((int) (i/6))*7];
        
        switch (((int) (i%6)))
        {
            case 0:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 1:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 2:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            // case 0:
            //     s0_gradient_device[i] = 1.0*s0_gradient_device[i]/move_frac_device[0];
            //     break;
            // case 1:
            //     s0_gradient_device[i] = 1.0*s0_gradient_device[i]/move_frac_device[0];
            //     break;
            // case 2:
            //     s0_gradient_device[i] = 1.0*s0_gradient_device[i]/move_frac_device[0];
            //     break;
            //debug when the move angle is one degree
            case 3:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 4:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            case 5:
                s0_gradient_device[i] = 1.0*s0_gradient_device[i];
                break;
            // case 3:
            //     s0_gradient_device[i] = 1.0*s0_gradient_device[i]/move_angle_rad_device[0];
            //     break;
            // case 4:
            //     s0_gradient_device[i] = 1.0*s0_gradient_device[i]/move_angle_rad_device[0];
            //     break;
            // case 5:
            //     s0_gradient_device[i] = 1.0*s0_gradient_device[i]/move_angle_rad_device[0];
            //     break;
        }
        
        s0_gradient_device[i] = floor(s0_gradient_device[i]/rounding_coeff_device[0])*rounding_coeff_device[0];

        s0_gradient_square_device[i] = s0_gradient_device[i]*s0_gradient_device[i];

    }

}



__global__
void s0_grad_length_sqrt_cal(double *rounding_coeff_device, int *N_string_device, double *s0_gradient_length_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(N_string_device[0]*2); i+=stride)
    {
        s0_gradient_length_device[i] = floor(sqrt(s0_gradient_length_device[i])/rounding_coeff_device[0])*rounding_coeff_device[0];
    }

}





__global__
void s0_new_cal(double *rounding_coeff_device, int *N_string_device, 
    double *move_frac_device, double *move_angle_rad_device,




    double *s0_gradient_length_device, double *s0_gradient_device,
    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,
    double *s1_a_device, double *s1_b_device, double *s1_c_device,
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;


    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        switch (i%6)
        {
            case 0:
                if (s0_gradient_length_device[((int) (i/6))*2+0] < move_frac_device[0])
                {
                    s1_a_device[((int) (i/6))] = s0_a_device[((int) (i/6))] - move_frac_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_a_device[((int) (i/6))] = s0_a_device[((int) (i/6))] 
                    - move_frac_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+0])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 1:
                if (s0_gradient_length_device[((int) (i/6))*2+0] < move_frac_device[0])
                {
                    s1_b_device[((int) (i/6))] = s0_b_device[((int) (i/6))] - move_frac_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_b_device[((int) (i/6))] = s0_b_device[((int) (i/6))] 
                    - move_frac_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+0])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 2:
                if (s0_gradient_length_device[((int) (i/6))*2+0] < move_frac_device[0])
                {
                    s1_c_device[((int) (i/6))] = s0_c_device[((int) (i/6))] - move_frac_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_c_device[((int) (i/6))] = s0_c_device[((int) (i/6))] 
                    - move_frac_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+0])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;

            // debug when the move angle is one degree
            // case 3:
            //     if (s0_gradient_length_device[((int) (i/6))*2+1] < 5)
            //     {
            //         s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] - 1*s0_gradient_device[i]/180*PI;
            //     }
            //     else
            //     {
            //         s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] 
            //         - 1*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0]/180*PI;

            //     }
            //     break;
            // case 4:
            //     if (s0_gradient_length_device[((int) (i/6))*2+1] < 5)
            //     {
            //         s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] - 1*s0_gradient_device[i]/180*PI;
            //     }
            //     else
            //     {
            //         s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] 
            //         - 1*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0]/180*PI;
            //     }
            //     break;
            // case 5:
            //     if (s0_gradient_length_device[((int) (i/6))*2+1] < 5)
            //     {
            //         s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] - 1*s0_gradient_device[i]/180*PI;
            //     }
            //     else
            //     {
            //         s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] 
            //         - 1*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0]/180*PI;
            //     }
            //     break;

            case 3:
                // if (((int) (i/6))==399)
                // {
                //     printf("%lf\t%lf\t", s0_gradient_length_device[((int) (i/6))*2+1], s0_gradient_device[i]);
                // }
                if (s0_gradient_length_device[((int) (i/6))*2+1] < move_angle_rad_device[0])
                {
                    s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] 
                    - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 4:
                if (s0_gradient_length_device[((int) (i/6))*2+1] < move_angle_rad_device[0])
                {
                    s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] 
                    - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];
                }
                break;
            case 5:
                if (s0_gradient_length_device[((int) (i/6))*2+1] < move_angle_rad_device[0])
                {
                    s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
                }
                else
                {
                    s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] 
                    - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];

                }
                break;




            // case 3:
            //     if (((int) (i/6))==399)
            //     {
            //         printf("%lf\t%lf\t", s0_gradient_length_device[((int) (i/6))*2+1], s0_gradient_device[i]);
            //     }
            //     if (s0_gradient_length_device[((int) (i/6))*2+1] < 1.0*10/180*PI)
            //     {
            //         s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
            //     }
            //     else
            //     {
            //         s1_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))] 
            //         - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];
            //     }
            //     break;
            // case 4:
            //     if (s0_gradient_length_device[((int) (i/6))*2+1] < 1.0*10/180*PI)
            //     {
            //         s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
            //     }
            //     else
            //     {
            //         s1_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))] 
            //         - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];
            //     }
            //     break;
            // case 5:
            //     if (s0_gradient_length_device[((int) (i/6))*2+1] < 1.0*10/180*PI)
            //     {
            //         s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] - move_angle_rad_device[0]*s0_gradient_device[i];
            //     }
            //     else
            //     {
            //         s1_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))] 
            //         - move_angle_rad_device[0]*floor((s0_gradient_device[i]/s0_gradient_length_device[((int) (i/6))*2+1])/rounding_coeff_device[0])*rounding_coeff_device[0];

            //     }
            //     break;
        }




        



        

    }

}





__global__
void s1_fix_modify(int *N_string_device, 

    double *s0_gradient_length_device, double *s0_gradient_device,
    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,
    double *s1_a_device, double *s1_b_device, double *s1_c_device,
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]); i+=stride)
    {
        if (i==0)
        {
            // fix the start along the a-xis which makes sure the a good string is formed
            s1_a_device[0] = s0_a_device[0];
            
        }
        else if (i==(N_string_device[0]-1))
        {
            // fix the end along the a-xis which makes sure the a good string is formed
            s1_a_device[N_string_device[0]-1] = s0_a_device[N_string_device[0]-1];
            // make sure the start and end point are identical
            s1_b_device[N_string_device[0]-1] = s1_b_device[0];
            s1_c_device[N_string_device[0]-1] = s1_c_device[0];
            s1_alpha_device[N_string_device[0]-1] = s1_alpha_device[0];
            s1_beta_device[N_string_device[0]-1] = s1_beta_device[0];
            s1_gamma_device[N_string_device[0]-1] = s1_gamma_device[0];
        }
        else
        {
            if (s1_a_device[i]<0)
            {
                for (ii=(i+1); ii<=(N_string_device[0]+1); ii++)
                {
                    if (s1_a_device[ii]>0)
                    {
                        s1_a_device[i] = s1_a_device[ii];
                        s1_b_device[i] = s1_b_device[ii];
                        s1_c_device[i] = s1_c_device[ii];
                        s1_alpha_device[i] = s1_alpha_device[ii];
                        s1_beta_device[i] = s1_beta_device[ii];
                        s1_gamma_device[i] = s1_gamma_device[ii];
                        break;
                    }
                }
            }
            else if (s1_a_device[i]>1)
            {
                for (ii=(i-1); ii>=0; ii--)
                {
                    if (s1_a_device[ii]<1)
                    {
                        s1_a_device[i] = s1_a_device[ii];
                        s1_b_device[i] = s1_b_device[ii];
                        s1_c_device[i] = s1_c_device[ii];
                        s1_alpha_device[i] = s1_alpha_device[ii];
                        s1_beta_device[i] = s1_beta_device[ii];
                        s1_gamma_device[i] = s1_gamma_device[ii];
                        break;
                    }
                }
            }
        }
    }
}










__global__
void s1_fix_modify_upgrade(int *N_string_device, 

    int *direction_device,

    double *s0_gradient_length_device, double *s0_gradient_device,
    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,
    double *s1_a_device, double *s1_b_device, double *s1_c_device,
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]); i+=stride)
    {
        switch (direction_device[0])
        {
            case 1:
                if (i==0)
                {
                    // fix the start along the a-xis which makes sure the a good string is formed
                    s1_a_device[0] = s0_a_device[0];
                    
                }
                else if (i==(N_string_device[0]-1))
                {
                    // fix the end along the a-xis which makes sure the a good string is formed
                    s1_a_device[N_string_device[0]-1] = s0_a_device[N_string_device[0]-1];
                    // make sure the start and end point are identical
                    s1_b_device[N_string_device[0]-1] = s1_b_device[0];
                    s1_c_device[N_string_device[0]-1] = s1_c_device[0];
                    s1_alpha_device[N_string_device[0]-1] = s1_alpha_device[0];
                    s1_beta_device[N_string_device[0]-1] = s1_beta_device[0];
                    s1_gamma_device[N_string_device[0]-1] = s1_gamma_device[0];
                }
                else
                {
                    if (s1_a_device[i]<0)
                    {
                        for (ii=(i+1); ii<=(N_string_device[0]+1); ii++)
                        {
                            if (s1_a_device[ii]>0)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                    else if (s1_a_device[i]>1)
                    {
                        for (ii=(i-1); ii>=0; ii--)
                        {
                            if (s1_a_device[ii]<1)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                }
                break;

            case 2:

                if (i==0)
                {
                    // fix the start along the a-xis which makes sure the a good string is formed
                    s1_b_device[0] = s0_b_device[0];
                    
                }
                else if (i==(N_string_device[0]-1))
                {
                    // fix the end along the a-xis which makes sure the a good string is formed
                    s1_b_device[N_string_device[0]-1] = s0_b_device[N_string_device[0]-1];
                    // make sure the start and end point are identical
                    s1_a_device[N_string_device[0]-1] = s1_a_device[0];
                    s1_c_device[N_string_device[0]-1] = s1_c_device[0];
                    s1_alpha_device[N_string_device[0]-1] = s1_alpha_device[0];
                    s1_beta_device[N_string_device[0]-1] = s1_beta_device[0];
                    s1_gamma_device[N_string_device[0]-1] = s1_gamma_device[0];
                }
                else
                {
                    if (s1_b_device[i]<0)
                    {
                        for (ii=(i+1); ii<=(N_string_device[0]+1); ii++)
                        {
                            if (s1_b_device[ii]>0)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                    else if (s1_b_device[i]>1)
                    {
                        for (ii=(i-1); ii>=0; ii--)
                        {
                            if (s1_b_device[ii]<1)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                }

                break;


            case 3:

                if (i==0)
                {
                    // fix the start along the a-xis which makes sure the a good string is formed
                    s1_c_device[0] = s0_c_device[0];
                    
                }
                else if (i==(N_string_device[0]-1))
                {
                    // fix the end along the a-xis which makes sure the a good string is formed
                    s1_c_device[N_string_device[0]-1] = s0_c_device[N_string_device[0]-1];
                    // make sure the start and end point are identical
                    s1_a_device[N_string_device[0]-1] = s1_a_device[0];
                    s1_b_device[N_string_device[0]-1] = s1_b_device[0];
                    s1_alpha_device[N_string_device[0]-1] = s1_alpha_device[0];
                    s1_beta_device[N_string_device[0]-1] = s1_beta_device[0];
                    s1_gamma_device[N_string_device[0]-1] = s1_gamma_device[0];
                }
                else
                {
                    if (s1_c_device[i]<0)
                    {
                        for (ii=(i+1); ii<=(N_string_device[0]+1); ii++)
                        {
                            if (s1_c_device[ii]>0)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                    else if (s1_c_device[i]>1)
                    {
                        for (ii=(i-1); ii>=0; ii--)
                        {
                            if (s1_c_device[ii]<1)
                            {
                                s1_a_device[i] = s1_a_device[ii];
                                s1_b_device[i] = s1_b_device[ii];
                                s1_c_device[i] = s1_c_device[ii];
                                s1_alpha_device[i] = s1_alpha_device[ii];
                                s1_beta_device[i] = s1_beta_device[ii];
                                s1_gamma_device[i] = s1_gamma_device[ii];
                                break;
                            }
                        }
                    }
                }

                break;
        }
        
    }
}





__global__
void s1_frac2cart(int *N_string_device, 

    double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,

    double *s1_a_device, double *s1_b_device, double *s1_c_device,
    double *s1_cart_x_device, double *s1_cart_y_device, double *s1_cart_z_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]); i+=stride)
    {
        s1_cart_x_device[i] = frac2car_x_device(s1_a_device[i], s1_b_device[i], s1_c_device[i], frac2car_a_device);
        s1_cart_y_device[i] = frac2car_y_device(s1_a_device[i], s1_b_device[i], s1_c_device[i], frac2car_b_device);
        s1_cart_z_device[i] = frac2car_z_device(s1_a_device[i], s1_b_device[i], s1_c_device[i], frac2car_c_device);

    }
}




__global__
void s1_length_prep(int *N_string_device, 

    double *s1_cart_x_device, double *s1_cart_y_device, double *s1_cart_z_device,
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device,
    double *s1_length_coordinate_all_device, double *s1_length_orientation_all_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        // for the first point
        if (((int) i/6)==0)
        {
            s1_length_coordinate_all_device[(i/6)*3+(i%6)] = 0;

            s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = 0;
        }
        else
        {
            switch (((int) i%6))
            {
                case 0:
                    s1_length_coordinate_all_device[(i/6)*3+(i%6)] = pow((s1_cart_x_device[((int) i/6)]-s1_cart_x_device[((int) i/6)-1]),2);
                    break;
                case 1:
                    s1_length_coordinate_all_device[(i/6)*3+(i%6)] = pow((s1_cart_y_device[((int) i/6)]-s1_cart_y_device[((int) i/6)-1]),2);
                    break;
                case 2:
                    s1_length_coordinate_all_device[(i/6)*3+(i%6)] = pow((s1_cart_z_device[((int) i/6)]-s1_cart_z_device[((int) i/6)-1]),2);
                    break;

                case 3:
                    s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s1_alpha_device[((int) i/6)] -s1_alpha_device[((int) i/6)-1]),2);
                    break;
                case 4:
                    s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s1_beta_device[((int) i/6)] -s1_beta_device[((int) i/6)-1]),2);
                    break;
                case 5:
                    s1_length_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s1_gamma_device[((int) i/6)] -s1_gamma_device[((int) i/6)-1]),2);
                    break;
            }
        }

        

    }
}



__global__
void s1_length_sqrt_cal(double *rounding_coeff_device, int *N_string_device, 
    double *s1_length_coordinate_device, double *s1_length_orientation_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<(N_string_device[0]*2); i+=stride)
    {
        switch ((int) (i%2))
        {
            case 0:
                s1_length_coordinate_device[(i/2)] = sqrt(s1_length_coordinate_device[(i/2)]);
                break;
            case 1:
                s1_length_orientation_device[(i/2)] = sqrt(s1_length_orientation_device[(i/2)]);
                break;
        }
    }

}

__global__
void remap_s1_length_for_cumulation(int *N_string_device, double *s1_length_coordinate_device, double *s1_length_orientation_device,

    double *s1_length_coordinate_remap_device, double *s1_length_orientation_remap_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i;
    

    for (i=index; i<((int) (N_string_device[0]*(1+N_string_device[0])*0.5*2)); i+=stride)
    {
        switch ((int) (i%2))
        {
            case 0:
                s1_length_coordinate_remap_device[(i/2)] = 
                s1_length_coordinate_device[((int) ((i/2) - ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1)*(1+ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1))*0.5))];
                break;
            case 1:
                s1_length_orientation_remap_device[(i/2)] = 
                s1_length_orientation_device[((int) ((i/2) - ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1)*(1+ceil((sqrt(8*(((double) (i/2))+1)+1)-1)*0.5-1))*0.5))];
                break;
        }
    }

}



__global__
void s1_2_s2(int *N_string_device, double *s1_l_abs_coordinate_device, double *s1_l_abs_orientation_device,
    double *s1_length_coordinate_cumulation_device, double *s1_length_orientation_cumulation_device,

    double *s1_a_device, double *s1_b_device, double *s1_c_device, 
    double *s1_alpha_device, double *s1_beta_device, double *s1_gamma_device,

    double *s2_a_device, double *s2_b_device, double *s2_c_device,
    double *s2_alpha_device, double *s2_beta_device, double *s2_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*2); i+=stride)
    {
        switch ((int) (i%2))
        {
            case 0:
                for (ii=(i/2); ;)
                {
                    if(ii==(N_string_device[0]-1))
                    {
                        ii--;
                    }

                    // go beyond the lower boundary
                    if ((1.0*(i/2)/(N_string_device[0]-1)) < (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0]))
                    {
                        ii--;
                    }
                    // go beyond the upper boundary
                    else if ((1.0*(i/2)/(N_string_device[0]-1)) > (1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]))
                    {
                        ii++;
                    }
                    else if ( ((1.0*(i/2)/(N_string_device[0]-1)) >= (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0]))
                        && ((1.0*(i/2)/(N_string_device[0]-1)) <= (1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0])) )
                    {
                        break;
                    }
                }
                s2_a_device[(i/2)] = s1_a_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])) * 
                ((s1_a_device[ii+1]-s1_a_device[ii])/((1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]) - (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])));
                s2_b_device[(i/2)] = s1_b_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])) * 
                ((s1_b_device[ii+1]-s1_b_device[ii])/((1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]) - (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])));
                s2_c_device[(i/2)] = s1_c_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])) * 
                ((s1_c_device[ii+1]-s1_c_device[ii])/((1.0*s1_length_coordinate_cumulation_device[ii+1]/s1_l_abs_coordinate_device[0]) - (1.0*s1_length_coordinate_cumulation_device[ii]/s1_l_abs_coordinate_device[0])));
                break;

            case 1:
                // special case for orientation invariant molecule (single bead)
                // if (fabs(s1_l_abs_orientation_device[0])<1e-10)
                // {
                    s2_alpha_device[(i/2)] = s1_alpha_device[(i/2)];
                    s2_beta_device[(i/2)] = s1_beta_device[(i/2)];
                    s2_gamma_device[(i/2)] = s1_gamma_device[(i/2)];
                // }
                // else
                // {
                //     for (ii=(i/2); ;)
                //     {
                //         if(ii==(N_string_device[0]-1))
                //         {
                //             ii--;
                //         }

                //         // go beyond the lower boundary
                //         if ((1.0*(i/2)/(N_string_device[0]-1)) < (1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0]))
                //         {
                //             ii--;
                //         }
                //         // go beyond the upper boundary
                //         else if ((1.0*(i/2)/(N_string_device[0]-1)) > (1.0*s1_length_orientation_cumulation_device[ii+1]/s1_l_abs_orientation_device[0]))
                //         {
                //             ii++;
                //         }
                //         else if ( ((1.0*(i/2)/(N_string_device[0]-1)) >= (1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0]))
                //             && ((1.0*(i/2)/(N_string_device[0]-1)) <= (1.0*s1_length_orientation_cumulation_device[ii+1]/s1_l_abs_orientation_device[0])) )
                //         {
                //             break;
                //         }
                //     }
                //     s0_alpha_device[(i/2)] = s1_alpha_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0])) * 
                //     ((s1_alpha_device[ii+1]-s1_alpha_device[ii])/((1.0*s1_length_orientation_cumulation_device[ii+1]/s1_l_abs_orientation_device[0]) - (1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0])));
                //     s0_beta_device[(i/2)] = s1_beta_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0])) * 
                //     ((s1_beta_device[ii+1]-s1_beta_device[ii])/((1.0*s1_length_orientation_cumulation_device[ii+1]/s1_l_abs_orientation_device[0]) - (1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0])));
                //     s0_gamma_device[(i/2)] = s1_gamma_device[ii] + ((1.0*(i/2)/(N_string_device[0]-1))-(1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0])) * 
                //     ((s1_gamma_device[ii+1]-s1_gamma_device[ii])/((1.0*s1_length_orientation_cumulation_device[ii+1]/s1_l_abs_orientation_device[0]) - (1.0*s1_length_orientation_cumulation_device[ii]/s1_l_abs_orientation_device[0])));
                // }
                break;
        }  
    }
}







__global__
void check_s2(int *N_string_device, double *V_s0, double *V_s2,
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device, 
    double *s2_alpha_device, double *s2_beta_device, double *s2_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*3); i+=stride)
    {
        switch ((int) (i%3))
        {
            case 0:
                if (V_s2[((int) (i/3))] > V_s0[((int) (i/3))])
                {
                    // potential is not minimized after string move of angle
                    // thus overwrite the angle movement
                    s2_alpha_device[((int) (i/3))] = s0_alpha_device[((int) (i/3))];
                }
                break;

            case 1:
                if (V_s2[((int) (i/3))] > V_s0[((int) (i/3))])
                {
                    // potential is not minimized after string move of angle
                    // thus overwrite the angle movement
                    s2_beta_device[((int) (i/3))] = s0_beta_device[((int) (i/3))];
                }
                break;

            case 2:
                if (V_s2[((int) (i/3))] > V_s0[((int) (i/3))])
                {
                    // potential is not minimized after string move of angle
                    // thus overwrite the angle movement
                    s2_gamma_device[((int) (i/3))] = s0_gamma_device[((int) (i/3))];
                }
                break;
        }  
    }
}








__global__
void smooth_angle(int *N_string_device, double *smooth_coeff_device, 
    double *s2_alpha_device, double *s2_beta_device, double *s2_gamma_device, 
    double *s2_alpha_smooth_device, double *s2_beta_smooth_device, double *s2_gamma_smooth_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*3); i+=stride)
    {
        switch ((int) (i%3))
        {
            case 0:
                if ((((int) (i/3))==0)||(((int) (i/3))==(N_string_device[0]-1)))
                {
                    // first and last point along the string
                    s2_alpha_smooth_device[(i/3)] = s2_alpha_device[(i/3)];
                }
                else
                {
                    if ((s2_alpha_device[(i/3)-1]-s2_alpha_device[(i/3)]) > PI)
                    {
                        if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) > PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]-2*2*PI);
                        }
                        else
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]-1*2*PI);
                        }
                    }
                    else if ((s2_alpha_device[(i/3)-1]-s2_alpha_device[(i/3)]) < -PI)
                    {
                        if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) < -PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]+2*2*PI);
                        }
                        else
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]);
                        }
                    }
                    else
                    {
                        if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) > PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]-1*2*PI);
                        }
                        else if ((s2_alpha_device[(i/3)+1]-s2_alpha_device[(i/3)]) < -PI)
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]+1*2*PI);
                        }
                        else
                        {
                            s2_alpha_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_alpha_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_alpha_device[(i/3)-1]+s2_alpha_device[(i/3)+1]);
                        }
                    }
                }
                break;

            case 1:
                if ((((int) (i/3))==0)||(((int) (i/3))==(N_string_device[0]-1)))
                {
                    // first and last point along the string
                    s2_beta_smooth_device[(i/3)] = s2_beta_device[(i/3)];
                }
                else
                {
                    if ((s2_beta_device[(i/3)-1]-s2_beta_device[(i/3)]) > PI)
                    {
                        if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) > PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]-2*2*PI);
                        }
                        else
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]-1*2*PI);
                        }
                    }
                    else if ((s2_beta_device[(i/3)-1]-s2_beta_device[(i/3)]) < -PI)
                    {
                        if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) < -PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]+2*2*PI);
                        }
                        else
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]);
                        }
                    }
                    else
                    {
                        if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) > PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]-1*2*PI);
                        }
                        else if ((s2_beta_device[(i/3)+1]-s2_beta_device[(i/3)]) < -PI)
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]+1*2*PI);
                        }
                        else
                        {
                            s2_beta_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_beta_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_beta_device[(i/3)-1]+s2_beta_device[(i/3)+1]);
                        }
                    }
                }
                break;

            case 2:
                if ((((int) (i/3))==0)||(((int) (i/3))==(N_string_device[0]-1)))
                {
                    // first and last point along the string
                    s2_gamma_smooth_device[(i/3)] = s2_gamma_device[(i/3)];
                }
                else
                {
                    if ((s2_gamma_device[(i/3)-1]-s2_gamma_device[(i/3)]) > PI)
                    {
                        if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) > PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]-2*2*PI);
                        }
                        else
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]-1*2*PI);
                        }
                    }
                    else if ((s2_gamma_device[(i/3)-1]-s2_gamma_device[(i/3)]) < -PI)
                    {
                        if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) < -PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]+2*2*PI);
                        }
                        else
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]);
                        }
                    }
                    else
                    {
                        if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) > PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]-1*2*PI);
                        }
                        else if ((s2_gamma_device[(i/3)+1]-s2_gamma_device[(i/3)]) < -PI)
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]+1*2*PI);
                        }
                        else
                        {
                            s2_gamma_smooth_device[(i/3)] = (1-smooth_coeff_device[0])*s2_gamma_device[(i/3)] + 
                            0.5*smooth_coeff_device[0]*(s2_gamma_device[(i/3)-1]+s2_gamma_device[(i/3)+1]);
                        }
                    }
                }
                break;
        }  
    }
}





__global__
void diff_s_prep(int *N_string_device, 

    double *s0_a_device, double *s0_b_device, double *s0_c_device,
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device,

    double *s_final_a_device, double *s_final_b_device, double *s_final_c_device,
    double *s_final_alpha_device, double *s_final_beta_device, double *s_final_gamma_device,

    double *diff_s_coordinate_all_device, double *diff_s_orientation_all_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        
        switch (((int) i%6))
        {
            case 0:
                diff_s_coordinate_all_device[(i/6)*3+(i%6)] = pow((s0_a_device[((int) i/6)]-s_final_a_device[((int) i/6)]),2);
                break;
            case 1:
                diff_s_coordinate_all_device[(i/6)*3+(i%6)] = pow((s0_b_device[((int) i/6)]-s_final_b_device[((int) i/6)]),2);
                break;
            case 2:
                diff_s_coordinate_all_device[(i/6)*3+(i%6)] = pow((s0_c_device[((int) i/6)]-s_final_c_device[((int) i/6)]),2);
                break;
            case 3:
                diff_s_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s0_alpha_device[((int) i/6)]-s_final_alpha_device[((int) i/6)]),2);
                break;
            case 4:
                diff_s_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s0_beta_device[((int) i/6)]-s_final_beta_device[((int) i/6)]),2);
                break;
            case 5:
                diff_s_orientation_all_device[(i/6)*3+(i%6)-3] = pow((s0_gamma_device[((int) i/6)]-s_final_gamma_device[((int) i/6)]),2);
                break;
        }

        

    }
}



__global__
void check_signal(int *N_string_device, 
    double *total_diff_s_coordinate_device, double *total_diff_s_orientation_device,
    double *convergence_coorindate_device, double *convergence_orientation_device,
    int *signal_coordinate_device, int *signal_orientation_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<2; i+=stride)
    {
        switch (i)
        {
            case 0:
                if (signal_coordinate_device[0] == 0)
                {
                    if (total_diff_s_coordinate_device[0] < convergence_coorindate_device[0])
                    {
                        // if (total_diff_s_orientation_device[0] < convergence_orientation_device[0])
                        // {
                            signal_coordinate_device[0] = 1;
                        // }
                    }
                }
                break;
            case 1:
                if (signal_orientation_device[0] == 0)
                {
                    if (total_diff_s_orientation_device[0] < convergence_orientation_device[0])
                    {
                        // if (total_diff_s_coordinate_device[0] < convergence_coorindate_device[0])
                        // {
                            signal_orientation_device[0] = 1;
                        // }
                    }
                }
                break;
        }
    }
}





__global__
void copy2s0(int *N_string_device, 
    int *signal_coordinate_device, int *signal_orientation_device,

    double *s_copy_a_device, double *s_copy_b_device, double *s_copy_c_device, 
    double *s_copy_alpha_device, double *s_copy_beta_device, double *s_copy_gamma_device, 
    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    

    for (i=index; i<(N_string_device[0]*6); i+=stride)
    {
        switch ((int) (i%6))
        {
            case 0:
                // if (signal_coordinate_device[0] == 1)
                // {
                    // s0_a_device[((int) (i/6))] = s0_a_device[((int) (i/6))];
                // }
                // else
                // {
                    s0_a_device[((int) (i/6))] = s_copy_a_device[((int) (i/6))];
                // }
                break;
            case 1:
                // if (signal_coordinate_device[0] == 1)
                // {
                    // s0_b_device[((int) (i/6))] = s0_b_device[((int) (i/6))];
                // }
                // else
                // {
                    s0_b_device[((int) (i/6))] = s_copy_b_device[((int) (i/6))];
                // }
                break;
            case 2:
                // if (signal_coordinate_device[0] == 1)
                // {
                    // s0_c_device[((int) (i/6))] = s0_c_device[((int) (i/6))];
                // }
                // else
                // {
                    s0_c_device[((int) (i/6))] = s_copy_c_device[((int) (i/6))];
                // }
                break;
            case 3:
                // if (signal_orientation_device[0] == 1)
                // {
                    // s0_alpha_device[((int) (i/6))] = s0_alpha_device[((int) (i/6))];
                // }
                // else
                // {
                    if (s_copy_alpha_device[((int) (i/6))] > (2*PI))
                    {
                        s0_alpha_device[((int) (i/6))] = s_copy_alpha_device[((int) (i/6))] - 2*PI;
                    }
                    else if (s_copy_alpha_device[((int) (i/6))] < 0)
                    {
                        s0_alpha_device[((int) (i/6))] = s_copy_alpha_device[((int) (i/6))] + 2*PI;
                    }
                    else
                    {
                        s0_alpha_device[((int) (i/6))] = s_copy_alpha_device[((int) (i/6))];
                    }
                // }
                break;
            case 4:
                // if ((signal_orientation_device[0] == 1)||(signal_orientation_device[0] == 1))
                // {
                //     s0_beta_device[((int) (i/6))] = s0_beta_device[((int) (i/6))];
                // }
                // else
                // {
                    if (s_copy_beta_device[((int) (i/6))] > (2*PI))
                    {
                        s0_beta_device[((int) (i/6))] = s_copy_beta_device[((int) (i/6))] - 2*PI;
                    }
                    else if (s_copy_beta_device[((int) (i/6))] < 0)
                    {
                        s0_beta_device[((int) (i/6))] = s_copy_beta_device[((int) (i/6))] + 2*PI;
                    }
                    else
                    {
                        s0_beta_device[((int) (i/6))] = s_copy_beta_device[((int) (i/6))];
                    }
                // }
                break;
            case 5:
                // if ((signal_orientation_device[0] == 1)||(signal_orientation_device[0] == 1))
                // {
                //     s0_gamma_device[((int) (i/6))] = s0_gamma_device[((int) (i/6))];
                // }
                // else
                // {
                    if (s_copy_gamma_device[((int) (i/6))] > (2*PI))
                    {
                        s0_gamma_device[((int) (i/6))] = s_copy_gamma_device[((int) (i/6))] - 2*PI;
                    }
                    else if (s_copy_gamma_device[((int) (i/6))] < 0)
                    {
                        s0_gamma_device[((int) (i/6))] = s_copy_gamma_device[((int) (i/6))] + 2*PI;
                    }
                    else
                    {
                        s0_gamma_device[((int) (i/6))] = s_copy_gamma_device[((int) (i/6))];
                    }
                // }
                break;
        }  
    }
}




__global__
void print_convergence(int n, double *x1, double *x2, double *x3, double *x4)
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
                printf("%lf\t%lf\n", 1.0*x1[0]/x3[0], 1.0*x2[0]/x4[0]);
            }
        }
    }
}








































__global__
void check_double_special(int n, int *N_atom_adsorbate_device,

    double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,

    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,

    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j, ii;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                for (ii=0; ii<N_atom_adsorbate_device[0]; ii++)
                {
                    printf("%lf\t%lf\t%lf\n", 
                        frac2car_x_device(s0_a_device[i], s0_b_device[i], s0_c_device[i], frac2car_a_device)
                        +rotate_moleucle_x_device(s0_alpha_device[i], s0_beta_device[i], s0_gamma_device[i],
            vector_adsorbate_x_device[ii], vector_adsorbate_y_device[ii], vector_adsorbate_z_device[ii]),
                        frac2car_y_device(s0_a_device[i], s0_b_device[i], s0_c_device[i], frac2car_b_device)
                        +rotate_moleucle_y_device(s0_alpha_device[i], s0_beta_device[i], s0_gamma_device[i],
            vector_adsorbate_x_device[ii], vector_adsorbate_y_device[ii], vector_adsorbate_z_device[ii]),
                        frac2car_z_device(s0_a_device[i], s0_b_device[i], s0_c_device[i], frac2car_c_device)
                        +rotate_moleucle_z_device(s0_alpha_device[i], s0_beta_device[i], s0_gamma_device[i],
            vector_adsorbate_x_device[ii], vector_adsorbate_y_device[ii], vector_adsorbate_z_device[ii]) );

                }
                printf("\n");
                
                

            }
        }
    }
}


__global__
void check_double_special2(int n, int *N_atom_adsorbate_device,

    double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device,

    double *vector_adsorbate_x_device, double *vector_adsorbate_y_device, double *vector_adsorbate_z_device,

    double *s0_a_device, double *s0_b_device, double *s0_c_device, 
    double *s0_alpha_device, double *s0_beta_device, double *s0_gamma_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, j, ii;
    for (j=0; j<n; j++)
    {
        for (i=index; i<n; i+=stride)
        {
            if (i==j)
            {
                for (ii=0; ii<N_atom_adsorbate_device[0]; ii++)
                {
                    if ((i%40)==0)
                    {
                        printf("%lf\t%lf\t%lf\n", 
                        frac2car_x_device(s0_a_device[i], s0_b_device[i], s0_c_device[i], frac2car_a_device)
                        +rotate_moleucle_x_device(s0_alpha_device[i], s0_beta_device[i], s0_gamma_device[i],
            vector_adsorbate_x_device[ii], vector_adsorbate_y_device[ii], vector_adsorbate_z_device[ii]),
                        frac2car_y_device(s0_a_device[i], s0_b_device[i], s0_c_device[i], frac2car_b_device)
                        +rotate_moleucle_y_device(s0_alpha_device[i], s0_beta_device[i], s0_gamma_device[i],
            vector_adsorbate_x_device[ii], vector_adsorbate_y_device[ii], vector_adsorbate_z_device[ii]),
                        frac2car_z_device(s0_a_device[i], s0_b_device[i], s0_c_device[i], frac2car_c_device)
                        +rotate_moleucle_z_device(s0_alpha_device[i], s0_beta_device[i], s0_gamma_device[i],
            vector_adsorbate_x_device[ii], vector_adsorbate_y_device[ii], vector_adsorbate_z_device[ii]) );
                    }
                }
            }
        }
    }
}



// //convert fractional coordinate to cartesian coordinate
// void frac2car(double frac_a, double frac_b, double frac_c, double frac2car_a[], double frac2car_b[], double frac2car_c[],
//                 double cart_x[], double cart_y[], double cart_z[])
// {
//     cart_x[0] = frac_a*frac2car_a[0] + frac_b*frac2car_a[1] + frac_c*frac2car_a[2];
//     cart_y[0] = frac_a*frac2car_b[0] + frac_b*frac2car_b[1] + frac_c*frac2car_b[2];
//     cart_z[0] = frac_a*frac2car_c[0] + frac_b*frac2car_c[1] + frac_c*frac2car_c[2];
// }



double trapz(double x[], double y[], int N)
{
    int i;
    double result = 0;
    for (i=1; i<N; i++)
    {
        result = result + (x[i] - x[i-1])*(y[i] + y[i-1])/2;
    }
    return result;
}

double max(double x[], int n)
{
    int i;
    double result = x[0];
    for (i=0; i<n; i++)
    {
        if (x[i]>result)
        {
            result = x[i];
        }
    }
    return result;
}








int main(int argc, char *argv[])
{
    //To calculate the external potential field, two file strings are needed: input filename and output filename
	//define file varaiable
	FILE *fp1;
	int buffersize = 512;
	char str[buffersize];
	//define read-in parameters
	// int Nmaxa, Nmaxb, Nmaxc;
	double La, Lb, Lc, dL;
	double alpha, beta, gamma;
    double alpha_rad, beta_rad, gamma_rad;
    int FH_signal;
    double mass, temperature[1];
    int set_running_step;
	double cutoff[1];
    int N_string[1];
    int int_N_string;
    int direction[1];
    double move_angle_degree[1], move_angle_rad[1], move_frac[1];
	int N_atom_frame[1], N_atom_adsorbate[1];
    double set_conv_trans_percent, set_conv_rot_percent;
    //define ancillary parameters
    double center_of_mass_x[1], center_of_mass_y[1], center_of_mass_z[1], total_mass_adsorbate;
    double temp_x[1], temp_y[1], temp_z[1];
    double cart_x, cart_y, cart_z;
    double cart_x_extended[1], cart_y_extended[1], cart_z_extended[1];
    int times_x[1], times_y[1], times_z[1], times;
    double a;
    // int a_N, b_N, c_N;
    double shift;
    double loc_a, loc_b, loc_c, loc_x, loc_y, loc_z, loc_u;
    double temp_frame_a, temp_frame_b, temp_frame_c;
    double temp_u;
    int i, ii, iii, iiii, j, jj, jjj, k, kk;
    double dis;
    //done!!!!!

    //read input file parameters
	fp1 = fopen(argv[1], "r");
	// fp1 = fopen("AMUWIP_charged.input", "r");
	fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
	// fscanf(fp1,"%d %d %d\n", &Nmaxa, &Nmaxb, &Nmaxc);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf %lf\n", &La, &Lb, &Lc, &dL);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf\n", &alpha, &beta, &gamma);
    alpha_rad = alpha*PI/180;
    beta_rad = beta*PI/180;
    gamma_rad = gamma*PI/180;
	fgets(str, buffersize, fp1);
    fscanf(fp1,"%lf %d %lf %lf %d\n", &cutoff[0], &FH_signal, &total_mass_adsorbate, &temperature[0], &set_running_step);
    // printf("running steps: %d\n", set_running_step);
    //read string calculation setting
    fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
    fscanf(fp1,"%d\n", &direction[0]);
    fgets(str, buffersize, fp1);
    fscanf(fp1,"%d %lf %lf\n", &N_string[0], &move_frac[0], &move_angle_degree[0]);
    fgets(str, buffersize, fp1);
    fscanf(fp1,"%lf %lf\n", &set_conv_trans_percent, &set_conv_rot_percent);
    //read adsorbate information
    fgets(str, buffersize, fp1);
    fgets(str, buffersize, fp1);
    fscanf(fp1,"%d\n", &N_atom_adsorbate[0]);
    // printf("N_atom_adsorbate: %d\n", N_atom_adsorbate[0]);
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
        center_of_mass_x[0] += 1.0*x_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
        center_of_mass_y[0] += 1.0*y_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
        center_of_mass_z[0] += 1.0*z_adsorbate[i]*mass_adsorbate[i]/total_mass_adsorbate;
    }
    // printf("center of the mass:\t%lf\t%lf\t%lf\n", center_of_mass_x, center_of_mass_y, center_of_mass_z);
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
	double epsilon_frame[N_atom_frame[0]*times], sigma_frame[N_atom_frame[0]*times], mass_frame[N_atom_frame[0]*times];
	double frac_a_frame[N_atom_frame[0]*times], frac_b_frame[N_atom_frame[0]*times], frac_c_frame[N_atom_frame[0]*times];
    for (i=0; i<N_atom_frame[0]; i++)
	{
		fscanf(fp1,"%lf %lf %lf %lf %lf %lf %lf\n", &a, &sigma_frame[i], &epsilon_frame[i], &mass_frame[i], &frac_a_frame[i], &frac_b_frame[i], &frac_c_frame[i]);
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
    //done!!!!

    






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
    int *direction_device;
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
    cudaMalloc((void**)&direction_device, sizeof(int));









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
    cudaMemcpyAsync(temperature_device, temperature, sizeof(double), cudaMemcpyHostToDevice, stream1);
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
    cudaMemcpyAsync(direction_device, direction, sizeof(int), cudaMemcpyHostToDevice, stream1);
    // cudaMemcpy(direction_device, direction, sizeof(int), cudaMemcpyHostToDevice);





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
    // check_int<<<1,32>>>(1, direction_device);
    // cudaDeviceSynchronize();







    clock_t t;





    double rot_alpha_angle, rot_beta_angle, rot_gamma_angle;
    double vector_adsorbate_x_rot[N_atom_adsorbate[0]], vector_adsorbate_y_rot[N_atom_adsorbate[0]], vector_adsorbate_z_rot[N_atom_adsorbate[0]];

    double delta_angle[1];
    delta_angle[0] = 90;
    double delta_grid[1];
    delta_grid[0] = 0.1;
    int N_grid[1], N_angle_alpha[1], N_angle_beta[1], N_angle_gamma[1];
    double *ini_mapping_Vext;
    double double_variable;
    // int direction = 1;
    // direction[0] = 1;
    double local_a, local_b, local_c;
    double local_x, local_y, local_z;
    double local_alpha_angle, local_beta_angle, local_gamma_angle;
    N_grid[0] = (int) (floor(1.0/delta_grid[0])+1);
    N_angle_alpha[0] = (int) (floor(360/delta_angle[0]));
    N_angle_beta[0] = (int) (floor(180/delta_angle[0]));
    N_angle_gamma[0] = (int) (floor(360/delta_angle[0]));
    ini_mapping_Vext = (double *) malloc(sizeof(double_variable)*N_grid[0]*N_grid[0]*N_angle_alpha[0]*N_angle_beta[0]*N_angle_gamma[0]);

    double a_minimum, b_minimum, c_minimum, alpha_minimum_angle, beta_minimum_angle, gamma_minimum_angle;
    double V_min;
    int minimum_signal = 0;

    // FILE *temp_1;
    // temp_1 = fopen("nohup1.out", "w+");

    // t = clock();
    // switch (direction)
    // {
    //     case 1:
    //         local_a = 0;
    //         for (i=0; i<N_grid[0]; i++)
    //         {
    //             for (ii=0; ii<N_grid[0]; ii++)
    //             {
    //                 loc_b = delta_grid[0]*i;
    //                 loc_c = delta_grid[0]*ii;
    //                 for (j=0; j<N_angle_alpha[0]; j++)
    //                 {
    //                     local_alpha_angle = delta_angle[0]*j;
    //                     for (jj=0; jj<N_angle_beta[0]; jj++)
    //                     {
    //                         local_beta_angle = delta_angle[0]*jj;
    //                         for (jjj=0; jjj<N_angle_gamma[0]; jjj++)
    //                         {
    //                             local_gamma_angle = delta_angle[0]*jjj;
                                
    //                             //position of this calculation (center of the mass) in cartesian coordiante
    //                             frac2car(loc_a, loc_b, loc_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);

    //                             rot_alpha_angle = local_alpha_angle;
    //                             rot_beta_angle = local_beta_angle;
    //                             rot_gamma_angle = local_gamma_angle;

    //                             rotate_moleucle(N_atom_adsorbate, vector_adsorbate_x, vector_adsorbate_y, vector_adsorbate_z, rot_alpha_angle, rot_beta_angle, 
    //                                 rot_gamma_angle, vector_adsorbate_x_rot, vector_adsorbate_y_rot, vector_adsorbate_z_rot);
                                
    //                             //calculate the interaction at each site
    //                             loc_u = V_cal_molecule(N_atom_adsorbate, temp_x, temp_y, temp_z, vector_adsorbate_x_rot, vector_adsorbate_y_rot, 
    //                                         vector_adsorbate_z_rot, N_atom_frame, times, frac_a_frame, frac_b_frame, frac_c_frame, frac2car_a, frac2car_b, 
    //                                         frac2car_c, times_x, times_y, times_z, cart_x_extended, cart_y_extended, cart_z_extended, cutoff, sigma_adsorbate, 
    //                                         sigma_frame, epsilon_adsorbate, epsilon_frame);
    //                             // fprintf(temp_1, "%lf\n", loc_u);

    //                             // if (minimum_signal==0)
    //                             // {
    //                             //     loc_u = V_cal_molecule(N_atom_adsorbate, temp_x, temp_y, temp_z, vector_adsorbate_x_rot, vector_adsorbate_y_rot, 
    //                             //             vector_adsorbate_z_rot, N_atom_frame, times, frac_a_frame, frac_b_frame, frac_c_frame, frac2car_a, frac2car_b, 
    //                             //             frac2car_c, times_x, times_y, times_z, cart_x_extended, cart_y_extended, cart_z_extended, cutoff, sigma_adsorbate, 
    //                             //             sigma_frame, epsilon_adsorbate, epsilon_frame);
    //                             //     // printf("%lf\n", loc_u);
    //                             // }
                                

    //                             // if (minimum_signal==0)
    //                             // {
    //                             //     V_min = loc_u;
    //                             //     minimum_signal = 1;
    //                             //     a_minimum = loc_a;
    //                             //     b_minimum = loc_b;
    //                             //     c_minimum = loc_c;
    //                             //     alpha_minimum_angle = local_alpha_angle;
    //                             //     beta_minimum_angle = local_beta_angle;
    //                             //     gamma_minimum_angle = local_gamma_angle;
    //                             //     // printf("%lf\n", V_min);
    //                             // }
    //                             // else if (loc_u < V_min)
    //                             // {
    //                             //     V_min = loc_u;
    //                             //     a_minimum = loc_a;
    //                             //     b_minimum = loc_b;
    //                             //     c_minimum = loc_c;
    //                             //     alpha_minimum_angle = local_alpha_angle;
    //                             //     beta_minimum_angle = local_beta_angle;
    //                             //     gamma_minimum_angle = local_gamma_angle;
    //                             //     // printf("%lf\n", V_min);
    //                             // }

                                
    //                         }
    //                     }
    //                 }
    //             }
    //         }



    //         break;
    //     case 2:





    //         break;
    //     case 3:





    //         break;
    // }
    // t = clock() - t;
    // // printf("cpu time: %lf\n", ((double)t)/CLOCKS_PER_SEC);
    // printf("%lf\t", ((double)t)/CLOCKS_PER_SEC);
    // printf("cpu minimum value:\t%lf\n", V_min);
    // fclose(temp_1);

    // printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", a_minimum, b_minimum, c_minimum, alpha_minimum_angle, beta_minimum_angle, gamma_minimum_angle, V_min);
    // rotate_moleucle(N_atom_adsorbate, vector_adsorbate_x, vector_adsorbate_y, vector_adsorbate_z, alpha_minimum_angle, beta_minimum_angle, 
    //                                 gamma_minimum_angle, vector_adsorbate_x_rot, vector_adsorbate_y_rot, vector_adsorbate_z_rot);
    // frac2car(a_minimum, b_minimum, c_minimum, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    // for (k=0; k<N_atom_adsorbate[0]; k++)
    // {
    //     // cartesian coordinate of each site
    //     loc_x = temp_x[0] + vector_adsorbate_x_rot[k];
    //     loc_y = temp_y[0] + vector_adsorbate_y_rot[k];
    //     loc_z = temp_z[0] + vector_adsorbate_z_rot[k];
    //     printf("%lf\t%lf\t%lf\n", loc_x, loc_y, loc_z);
    // }








    
    // N_string[0] = 401;
    // double s0_a[N_string], s0_b[N_string], s0_c[N_string], s0_alpha_angle[N_string], s0_beta_angle[N_string], s0_gamma_angle[N_string];
    // double s0_a_new[N_string], s0_b_new[N_string], s0_c_new[N_string], s0_alpha_angle_new[N_string], s0_beta_angle_new[N_string], s0_gamma_angle_new[N_string];

    // //For this version, after finding the minimum energy state. The same state is pulling thourgh the framework.
    // switch (direction)
    // {
    //     case 1:
    //         for (i=0; i<N_string; i++)
    //         {
    //             s0_a[i] = 1.0*i/(N_string-1);
    //             s0_b[i] = b_minimum;
    //             s0_c[i] = c_minimum;
    //             s0_alpha_angle[i] = alpha_minimum_angle;
    //             s0_beta_angle[i] = beta_minimum_angle;
    //             s0_gamma_angle[i] = gamma_minimum_angle;
    //         }


    //         break;
    //     case 2:





    //         break;
    //     case 3:





    //         break;
    // }


    // for (i=0; i<N_string; i++)
    // {
    //     loc_a = s0_a[i];
    //     loc_b = s0_b[i];
    //     loc_c = s0_c[i];
    //     frac2car(loc_a, loc_b, loc_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    //     rot_alpha_angle = s0_alpha_angle[i];
    //     rot_beta_angle = s0_beta_angle[i];
    //     rot_gamma_angle = s0_gamma_angle[i];
    //     rotate_moleucle(N_atom_adsorbate, vector_adsorbate_x, vector_adsorbate_y, vector_adsorbate_z, rot_alpha_angle, rot_beta_angle, 
    //                 rot_gamma_angle, vector_adsorbate_x_rot, vector_adsorbate_y_rot, vector_adsorbate_z_rot);
    //     //calculate the interaction at each site
    //     loc_u = V_cal_molecule(N_atom_adsorbate, temp_x, temp_y, temp_z, vector_adsorbate_x_rot, vector_adsorbate_y_rot, 
    //                 vector_adsorbate_z_rot, N_atom_frame, times, frac_a_frame, frac_b_frame, frac_c_frame, frac2car_a, frac2car_b, 
    //                 frac2car_c, times_x, times_y, times_z, cart_x_extended, cart_y_extended, cart_z_extended, cutoff, sigma_adsorbate, 
    //                 sigma_frame, epsilon_adsorbate, epsilon_frame);
    //     printf("%lf %lf %lf %lf %lf %lf ", temp_x[0], temp_y[0], temp_z[0], s0_alpha_angle[i], s0_beta_angle[i], s0_gamma_angle[i]);
    //     printf("%lf\n", loc_u);
    // }














    cudaStreamSynchronize(stream1);

    // printf("\n");
    // printf("\n");
    // printf("\n");
    // printf("\n");
    // printf("\n");
    // printf("\n");
    // printf("\n");
    // printf("\n");
    // printf("\n");
    // printf("\n");



    int N_points = N_grid[0]*N_grid[0]*N_angle_alpha[0]*N_angle_beta[0]*N_angle_gamma[0];




    // solution 1:
    int *N_grid_device, *N_angle_alpha_device, *N_angle_beta_device, *N_angle_gamma_device;
    double *delta_grid_device, *delta_angle_device;
    int *index_a_device, *index_b_device, *index_c_device;
    int *index_alpha_device, *index_beta_device, *index_gamma_device;
    int *index_adsorbate_device, *index_frame_device;
    double *cal_a_device, *cal_b_device, *cal_c_device;
    double *rot_alpha_rad_device, *rot_beta_rad_device, *rot_gamma_rad_device;
    double *loc_x_device, *loc_y_device, *loc_z_device;
    double *vector_adsorbate_x_rot_device, *vector_adsorbate_y_rot_device, *vector_adsorbate_z_rot_device;
    double *adsorbate_cart_x_rot_device, *adsorbate_cart_y_rot_device, *adsorbate_cart_z_rot_device;
    double *modify_frame_a_device, *modify_frame_b_device, *modify_frame_c_device;
    double *minimum_distance_device;
    double *V_total_1;
    // allocate memory
    cudaMalloc((void **)&N_grid_device, sizeof(int));
    cudaMalloc((void **)&N_angle_alpha_device, sizeof(int));
    cudaMalloc((void **)&N_angle_beta_device, sizeof(int));
    cudaMalloc((void **)&N_angle_gamma_device, sizeof(int));
    cudaMalloc((void **)&delta_grid_device, sizeof(double));
    cudaMalloc((void **)&delta_angle_device, sizeof(double));


    cudaMalloc((void **)&index_a_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_b_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_c_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_alpha_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_beta_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_gamma_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_adsorbate_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_frame_device, sizeof(int)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);




    cudaMalloc((void **)&cal_a_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&cal_b_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&cal_c_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&rot_alpha_rad_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&rot_beta_rad_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&rot_gamma_rad_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_x_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_y_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_z_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_x_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_y_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_z_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_x_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_y_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_z_rot_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_a_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_b_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_c_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&minimum_distance_device, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&V_total_1, sizeof(double)*N_points*N_atom_adsorbate[0]*N_atom_frame[0]*times);





    // memory transfer
    cudaMemcpy(N_grid_device, N_grid, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_angle_alpha_device, N_angle_alpha, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_angle_beta_device, N_angle_beta, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(N_angle_gamma_device, N_angle_gamma, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_grid_device, delta_grid, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(delta_angle_device, delta_angle, sizeof(double), cudaMemcpyHostToDevice);

    int temp_add_frame_host[1];
    temp_add_frame_host[0] = N_atom_frame[0]*times;
    int *temp_add_frame_device;
    cudaMalloc((void **)&temp_add_frame_device, sizeof(int));
    cudaMemcpy(temp_add_frame_device, temp_add_frame_host, sizeof(int), cudaMemcpyHostToDevice);
    // check_int<<<1,32>>>(1, temp_add_frame_device);














    int num_segments = N_points;
    int *h_offset = (int *) malloc(sizeof(int)*(num_segments+1));
    h_offset[0] = 0;
    for (i=1; i<=num_segments; i++)
    {
        h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    }
    int *d_offset;
    cudaMalloc((void**)&d_offset, (num_segments+1)*sizeof(int));
    cudaMemcpy(d_offset, h_offset, (num_segments+1)*sizeof(int), cudaMemcpyHostToDevice);
    free(h_offset);
    double *V_out_test;
    double *V_out_print;
    V_out_print = (double *) malloc(sizeof(double)*num_segments);
    cudaMalloc((void**)&V_out_test, sizeof(double)*num_segments);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;



    // temp_1 = fopen("nohup.out", "w+");


    t = clock();
    // Vext_cal_3<<<(int)((N_points*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>

    // (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
    // vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
    // N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
    // frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
    // times_x_device, times_y_device, times_z_device,
    // cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
    // frac2car_a_device, frac2car_b_device, frac2car_c_device,
    // cutoff_device,


    //             N_grid_device, N_angle_alpha_device, N_angle_beta_device, N_angle_gamma_device,
    //             delta_grid_device, delta_angle_device,
    //             index_a_device, index_b_device, index_c_device,
    //             index_alpha_device, index_beta_device, index_gamma_device,
    //             index_adsorbate_device, index_frame_device,

    //             cal_a_device, cal_b_device, cal_c_device,
    //             rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device,
    //             loc_x_device, loc_y_device, loc_z_device,
    //             vector_adsorbate_x_rot_device, vector_adsorbate_y_rot_device, vector_adsorbate_z_rot_device,
    //             adsorbate_cart_x_rot_device, adsorbate_cart_y_rot_device, adsorbate_cart_z_rot_device, 
    //             modify_frame_a_device, modify_frame_b_device, modify_frame_c_device,
    //             minimum_distance_device,
    //             V_total_1);


    Vext_cal_3_upgrade<<<(int)((N_points*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>

    (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
    vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
    N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
    times_x_device, times_y_device, times_z_device,
    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
    frac2car_a_device, frac2car_b_device, frac2car_c_device,
    cutoff_device,

                direction_device, 


                N_grid_device, N_angle_alpha_device, N_angle_beta_device, N_angle_gamma_device,
                delta_grid_device, delta_angle_device,
                index_a_device, index_b_device, index_c_device,
                index_alpha_device, index_beta_device, index_gamma_device,
                index_adsorbate_device, index_frame_device,

                cal_a_device, cal_b_device, cal_c_device,
                rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device,
                loc_x_device, loc_y_device, loc_z_device,
                vector_adsorbate_x_rot_device, vector_adsorbate_y_rot_device, vector_adsorbate_z_rot_device,
                adsorbate_cart_x_rot_device, adsorbate_cart_y_rot_device, adsorbate_cart_z_rot_device, 
                modify_frame_a_device, modify_frame_b_device, modify_frame_c_device,
                minimum_distance_device,
                V_total_1);


    cudaDeviceSynchronize();



    
    // return 0;
    // check_double_custom<<<1,32>>>(times*N_atom_adsorbate[0]*N_atom_frame[0], minimum_distance_device, V_total_1);

    

    // calculate potential energy at each grid by summing over the certain range
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_out_test, 
        num_segments, d_offset, d_offset+1);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_out_test, 
        num_segments, d_offset, d_offset+1);
    cudaFree(d_offset);
    cudaFree(d_temp_storage);
    cudaDeviceSynchronize();
    t = clock() - t;
    // printf("%lf\t", ((double)t)/CLOCKS_PER_SEC);
    // printf("gpu time: %lf\n", ((double)t)/CLOCKS_PER_SEC);
    // cudaMemcpy(V_out_print, V_out_test, (num_segments)*sizeof(double), cudaMemcpyDeviceToHost);
    // printf("%d\n", N_points);


    
    // find the minimum energy value and index for the configuration on the side
    d_temp_storage = NULL;
    cub::KeyValuePair<int, double> *min_value_index_device;
    cudaMalloc((void**)&min_value_index_device, sizeof(cub::KeyValuePair<int, double>));
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_out_test, min_value_index_device, num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_out_test, min_value_index_device, num_segments);
    // check_key<<<1,32>>>(1, min_value_index_device);
    cudaFree(d_temp_storage);

    cudaFree(V_out_test);
    cudaDeviceSynchronize();



    
    // return 0;




    // copy and save the information on the minimized configuration on the side




    


    


    // set up the variable for the string
    int *N_string_device;
    double *s0_a_device, *s0_b_device, *s0_c_device;
    double *s0_alpha_device, *s0_beta_device, *s0_gamma_device;
    double *s1_a_device, *s1_b_device, *s1_c_device;
    double *s1_alpha_device, *s1_beta_device, *s1_gamma_device;
    double *s1_cart_x_device, *s1_cart_y_device, *s1_cart_z_device;
    double *s1_length_coordinate_all_device, *s1_length_orientation_all_device;
    double *s1_length_coordinate_device, *s1_length_orientation_device;
    double *s1_l_abs_coordinate_device, *s1_l_abs_orientation_device;
    double *s1_length_coordinate_remap_device, *s1_length_orientation_remap_device;
    double *s1_length_coordinate_cumulation_device, *s1_length_orientation_cumulation_device;


    double *s2_a_device, *s2_b_device, *s2_c_device;
    double *s2_alpha_device, *s2_beta_device, *s2_gamma_device;
    // double *s2_a_device, *s2_b_device, *s2_c_device;
    double *s2_alpha_smooth_device, *s2_beta_smooth_device, *s2_gamma_smooth_device;


    double *s1_length_device, *s1_length_all_device, *s1_l_abs_device;
    double *s1_legnth_remap_device, *s1_length_cumulation_device;

    int  *index_s0_cal_Vext_s0_device;
    // double *index_a_cal_Vext_s0_device, *index_b_cal_Vext_s0_device, *index_c_cal_Vext_s0_device;
    // double *index_alpha_cal_Vext_s0_device, *index_beta_cal_Vext_s0_device, *index_gamma_cal_Vext_s0_device;
    int *index_adsorbate_cal_Vext_s0_device, *index_frame_cal_Vext_s0_device;

    double *a_cal_Vext_s0_device, *b_cal_Vext_s0_device, *c_cal_Vext_s0_device;
    double *alpha_rad_cal_Vext_s0_device, *beta_rad_cal_Vext_s0_device, *gamma_rad_cal_Vext_s0_device;
    double *loc_x_cal_Vext_s0_device, *loc_y_cal_Vext_s0_device, *loc_z_cal_Vext_s0_device;
    double *vector_adsorbate_x_rot_cal_Vext_s0_device, *vector_adsorbate_y_rot_cal_Vext_s0_device, *vector_adsorbate_z_rot_cal_Vext_s0_device;
    double *adsorbate_cart_x_rot_cal_Vext_s0_device, *adsorbate_cart_y_rot_cal_Vext_s0_device, *adsorbate_cart_z_rot_cal_Vext_s0_device;
    double *modify_frame_a_cal_Vext_s0_device, *modify_frame_b_cal_Vext_s0_device, *modify_frame_c_cal_Vext_s0_device;
    double *minimum_distance_cal_Vext_s0_device;
    double *V_s0_temp, *V_s0;
    double *V_s2;



    // allocate memory
    cudaMalloc((void**)&N_string_device, sizeof(int));
    cudaMalloc((void**)&s0_a_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_b_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_alpha_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_beta_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s0_gamma_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_a_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_b_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_alpha_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_beta_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_gamma_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_cart_x_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_cart_y_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_cart_z_device, sizeof(double)*N_string[0]);

    cudaMalloc((void**)&s1_length_coordinate_all_device, sizeof(double)*N_string[0]*3);
    cudaMalloc((void**)&s1_length_orientation_all_device, sizeof(double)*N_string[0]*3);
    cudaMalloc((void**)&s1_length_coordinate_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_length_orientation_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_l_abs_coordinate_device, sizeof(double));
    cudaMalloc((void**)&s1_l_abs_orientation_device, sizeof(double));
    cudaMalloc((void**)&s1_length_coordinate_remap_device, sizeof(double)*((int) (N_string[0]*(1+N_string[0])*0.5)));
    cudaMalloc((void**)&s1_length_orientation_remap_device, sizeof(double)*((int) (N_string[0]*(1+N_string[0])*0.5)));
    cudaMalloc((void**)&s1_length_coordinate_cumulation_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_length_orientation_cumulation_device, sizeof(double)*N_string[0]);


    cudaMalloc((void**)&s2_a_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_b_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_alpha_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_beta_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_gamma_device, sizeof(double)*N_string[0]);
    // cudaMalloc((void**)&s2_a_device, sizeof(double)*N_string[0]);
    // cudaMalloc((void**)&s2_b_device, sizeof(double)*N_string[0]);
    // cudaMalloc((void**)&s2_c_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_alpha_smooth_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_beta_smooth_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s2_gamma_smooth_device, sizeof(double)*N_string[0]);






    cudaMalloc((void**)&s1_length_device, sizeof(double)*N_string[0]);
    cudaMalloc((void**)&s1_length_all_device, sizeof(double)*N_string[0]*6);
    cudaMalloc((void**)&s1_l_abs_device, sizeof(double));
    cudaMalloc((void**)&s1_legnth_remap_device, sizeof(double)*((int) (N_string[0]*(1+N_string[0])*0.5)));
    cudaMalloc((void**)&s1_length_cumulation_device, sizeof(double)*N_string[0]);




    // copy and transfer memory
    cudaMemcpy(N_string_device, N_string, sizeof(int), cudaMemcpyHostToDevice);

    // check_int<<<1,32>>>(1, N_string_device);
    // check_int<<<1,32>>>(1, direction_device);


    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // initialize the string
    // printf("check\n");
    cudaDeviceSynchronize();
    int signal_straight_line = 0;
    int num_inidividual_ini_extra[1];
    num_inidividual_ini_extra[0] = 5;
    double limit_transition_frac[1], limit_rotation_angle[1];
    limit_transition_frac[0] = 0.15;
    limit_rotation_angle[0] = 0;

    int *i_cal_device;



    
    

    








    // printf("%d\n", argc);
    if (argc==4)
    {
        // there is also input of initital string

        // check the compatibility of the current input string
        fp1 = fopen(argv[2], "r");
        i=0;
        while (1)
        {
            if ( fgets(str, buffersize, fp1) != NULL)
            {
                i++;
            }
            else
            {
                break;
            }
        }
        fclose(fp1);
        if (i==N_string[0])
        {
            double *temp_input_load_a, *temp_input_load_b, *temp_input_load_c;
            double *temp_input_load_alpha, *temp_input_load_beta, *temp_input_load_gamma;

            temp_input_load_a = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_b = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_c = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_alpha = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_beta = (double *) malloc(N_string[0]*sizeof(double));
            temp_input_load_gamma = (double *) malloc(N_string[0]*sizeof(double));


            fp1 = fopen(argv[2], "r");
            for (ii=0; ii<i; ii++)
            {
                fscanf(fp1, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf", &temp_input_load_a[ii], &temp_input_load_b[ii], &temp_input_load_c[ii], 
                    &temp_input_load_alpha[ii], &temp_input_load_beta[ii], &temp_input_load_gamma[ii]);
                fgets(str, buffersize, fp1);
                // printf("%.10e\t%.10e\t%.10e\t%.10e\t%.10e\t%.10e\n", temp_input_load_a[ii], temp_input_load_b[ii], temp_input_load_c[ii], 
                //     temp_input_load_alpha[ii], temp_input_load_beta[ii], temp_input_load_gamma[ii]);
                // printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", temp_input_load_a[ii], temp_input_load_b[ii], temp_input_load_c[ii], 
                //     temp_input_load_alpha[ii], temp_input_load_beta[ii], temp_input_load_gamma[ii]);


                // // debug non-stop rotation
                // temp_input_load_alpha[ii] = 0;
                // temp_input_load_beta[ii] = 0;
                // temp_input_load_gamma[ii] = 0;
                
            }
            fclose(fp1);
            cudaMemcpy(s0_a_device, temp_input_load_a, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_b_device, temp_input_load_b, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_c_device, temp_input_load_c, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_alpha_device, temp_input_load_alpha, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_beta_device, temp_input_load_beta, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(s0_gamma_device, temp_input_load_gamma, N_string[0]*sizeof(double), cudaMemcpyHostToDevice);

        }
        else
        {
            printf("Warning!!!!\n");
            printf("Incompatible input string!!!\n");
            printf("Wrong line number!!!\n");
        }


    }
    else
    {
        if (signal_straight_line == 1)
        {
            // use straight line throught connecting the minimum energy point along the material
            ini_string_1<<<(int)((N_string[0]-1)/running_block_size+1),running_block_size>>>
            (N_string_device, cal_a_device, cal_b_device, cal_c_device, rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 
                temp_add_frame_device, N_atom_adsorbate_device, direction_device, min_value_index_device, 
                s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device);
        }
        else if (signal_straight_line == 0)
        {

            int *num_inidividual_ini_extra_device;
            double *ini_minimum_string_a_device, *ini_minimum_string_b_device, *ini_minimum_string_c_device;
            double *ini_minimum_string_alpha_device, *ini_minimum_string_beta_device, *ini_minimum_string_gamma_device;

            double *limit_transition_frac_device, *limit_rotation_angle_device;

            double *temp_partition_device;


            cudaMalloc((void**)&num_inidividual_ini_extra_device, sizeof(int));
            cudaMalloc((void**)&ini_minimum_string_a_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_b_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_c_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_alpha_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_beta_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_gamma_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&limit_transition_frac_device, sizeof(double));
            cudaMalloc((void**)&limit_rotation_angle_device, sizeof(double));
            cudaMalloc((void**)&i_cal_device, sizeof(double));
            cudaMalloc((void**)&temp_partition_device, sizeof(double)*N_string[0]*6);



            cudaMemcpy(num_inidividual_ini_extra_device, num_inidividual_ini_extra, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(limit_transition_frac_device, limit_transition_frac, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(limit_rotation_angle_device, limit_rotation_angle, sizeof(double), cudaMemcpyHostToDevice);







            // copy the start and end point
            // copy_ini<<<(int)((2*6)/running_block_size+1),running_block_size>>>
            // (ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
            // ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

            // cal_a_device, cal_b_device, cal_c_device, 
            // rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 

            // min_value_index_device, temp_add_frame_device, N_atom_adsorbate_device,
            // num_inidividual_ini_extra_device);




            copy_ini_upgrade<<<(int)((2*6)/running_block_size+1),running_block_size>>>
            (ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
            ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

            direction_device,

            cal_a_device, cal_b_device, cal_c_device, 
            rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 

            min_value_index_device, temp_add_frame_device, N_atom_adsorbate_device,
            num_inidividual_ini_extra_device);



            delta_angle[0] = 10;
            delta_grid[0] = 0.05;

            N_grid[0] = (int) (floor(2*limit_transition_frac[0]/delta_grid[0])+1);
            N_angle_alpha[0] = (int) (floor(2*limit_rotation_angle[0]/delta_angle[0]+1));
            N_angle_beta[0] = (int) (floor(2*limit_rotation_angle[0]/delta_angle[0]+1));
            N_angle_gamma[0] = (int) (floor(2*limit_rotation_angle[0]/delta_angle[0]+1));
            // N_angle_alpha[0] = (int) (floor(limit_rotation_angle[0]/delta_angle[0]+1));
            // N_angle_beta[0] = (int) (floor(limit_rotation_angle[0]/delta_angle[0]+1));
            // N_angle_gamma[0] = (int) (floor(limit_rotation_angle[0]/delta_angle[0]+1));





            cudaMemcpy(N_grid_device, N_grid, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(N_angle_alpha_device, N_angle_alpha, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(N_angle_beta_device, N_angle_beta, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(N_angle_gamma_device, N_angle_gamma, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(delta_grid_device, delta_grid, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(delta_angle_device, delta_angle, sizeof(double), cudaMemcpyHostToDevice);



            // printf("N_points: %d\n", N_points);
            N_points = N_grid[0]*N_grid[0]*N_angle_alpha[0]*N_angle_beta[0]*N_angle_gamma[0];
            // printf("N_points: %d\n", N_points);
            // printf("N_grid: %d\n", N_grid[0]);
            // printf("N_alpha: %d\n", N_angle_alpha[0]);
            // printf("N_beta: %d\n", N_angle_beta[0]);
            // printf("N_gamma: %d\n", N_angle_gamma[0]);
            int *ini_h_offset = (int *) malloc(sizeof(int)*(N_points+1));
            ini_h_offset[0] = 0;
            for (i=1; i<=N_points; i++)
            {
                ini_h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
            }
            int *ini_d_offset;
            cudaMalloc((void**)&ini_d_offset, (N_points+1)*sizeof(int));
            cudaMemcpy(ini_d_offset, ini_h_offset, (N_points+1)*sizeof(int), cudaMemcpyHostToDevice);
            free(ini_h_offset);
            double *V_ini_test;
            cudaMalloc((void**)&V_ini_test, sizeof(double)*N_points);



            for (i=1; i<=num_inidividual_ini_extra[0]; i++)
            {
                cudaMemcpy(i_cal_device, &i, sizeof(int), cudaMemcpyHostToDevice);
                // check_int<<<1,32>>>(1, i_cal_device);
                // printf("cpu: %d %d %d\n", N_points, times, N_atom_adsorbate[0]*N_atom_frame[0]);
               // Vext_cal_ini<<<(int)((N_points*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>

               //  (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
               //  vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
               //  N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
               //  frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
               //  times_x_device, times_y_device, times_z_device,
               //  cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
               //  frac2car_a_device, frac2car_b_device, frac2car_c_device,
               //  cutoff_device,


               //              N_grid_device, N_angle_alpha_device, N_angle_beta_device, N_angle_gamma_device,
               //              delta_grid_device, delta_angle_device,
               //              index_a_device, index_b_device, index_c_device,
               //              index_alpha_device, index_beta_device, index_gamma_device,
               //              index_adsorbate_device, index_frame_device,

               //              limit_transition_frac_device, limit_rotation_angle_device,
               //              ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
               //              ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device,

               //              i_cal_device, num_inidividual_ini_extra_device,

               //              cal_a_device, cal_b_device, cal_c_device,
               //              rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device,
               //              loc_x_device, loc_y_device, loc_z_device,
               //              vector_adsorbate_x_rot_device, vector_adsorbate_y_rot_device, vector_adsorbate_z_rot_device,
               //              adsorbate_cart_x_rot_device, adsorbate_cart_y_rot_device, adsorbate_cart_z_rot_device, 
               //              modify_frame_a_device, modify_frame_b_device, modify_frame_c_device,
               //              minimum_distance_device,
               //              V_total_1);



                Vext_cal_ini_upgrade<<<(int)((N_points*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>

                (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
                vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                times_x_device, times_y_device, times_z_device,
                cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                frac2car_a_device, frac2car_b_device, frac2car_c_device,
                cutoff_device,


                            direction_device,


                            N_grid_device, N_angle_alpha_device, N_angle_beta_device, N_angle_gamma_device,
                            delta_grid_device, delta_angle_device,
                            index_a_device, index_b_device, index_c_device,
                            index_alpha_device, index_beta_device, index_gamma_device,
                            index_adsorbate_device, index_frame_device,

                            limit_transition_frac_device, limit_rotation_angle_device,
                            ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
                            ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device,

                            i_cal_device, num_inidividual_ini_extra_device,

                            cal_a_device, cal_b_device, cal_c_device,
                            rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device,
                            loc_x_device, loc_y_device, loc_z_device,
                            vector_adsorbate_x_rot_device, vector_adsorbate_y_rot_device, vector_adsorbate_z_rot_device,
                            adsorbate_cart_x_rot_device, adsorbate_cart_y_rot_device, adsorbate_cart_z_rot_device, 
                            modify_frame_a_device, modify_frame_b_device, modify_frame_c_device,
                            minimum_distance_device,
                            V_total_1);

                cudaDeviceSynchronize();




                d_temp_storage = NULL;
                cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_ini_test, 
                    N_points, ini_d_offset, ini_d_offset+1);
                cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
                cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_total_1, V_ini_test, 
                    N_points, ini_d_offset, ini_d_offset+1);
                cudaFree(d_temp_storage);
                // check_double<<<1,32>>>(N_points, V_ini_test);
                // check_double<<<1,32>>>(1, V_ini_test);
                d_temp_storage = NULL;
                cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_ini_test, min_value_index_device, N_points);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, V_ini_test, min_value_index_device, N_points);
                // check_key<<<1,32>>>(1, min_value_index_device);
                cudaFree(d_temp_storage);


                // copy_ini_middle<<<(int)((1*6)/running_block_size+1),running_block_size>>>
                // (ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
                // ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

                // cal_a_device, cal_b_device, cal_c_device, 
                // rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 

                // i_cal_device, 

                // min_value_index_device, temp_add_frame_device, N_atom_adsorbate_device,
                // num_inidividual_ini_extra_device);


                copy_ini_middle_upgrade<<<(int)((1*6)/running_block_size+1),running_block_size>>>
                (ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
                ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

                direction_device,
                
                cal_a_device, cal_b_device, cal_c_device, 
                rot_alpha_rad_device, rot_beta_rad_device, rot_gamma_rad_device, 

                i_cal_device, 

                min_value_index_device, temp_add_frame_device, N_atom_adsorbate_device,
                num_inidividual_ini_extra_device);
            }

            cudaFree(V_ini_test);

            double *ini_minimum_string_cart_x_device, *ini_minimum_string_cart_y_device, *ini_minimum_string_cart_z_device;
            double *ini_minimum_length_all_device, *ini_minimum_length_device, *ini_minimum_l_abs_device;



            cudaMalloc((void**)&ini_minimum_string_cart_x_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_cart_y_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_string_cart_z_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_length_all_device, sizeof(double)*(num_inidividual_ini_extra[0]+2)*3);
            cudaMalloc((void**)&ini_minimum_length_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));
            cudaMalloc((void**)&ini_minimum_l_abs_device, sizeof(double)*1);
            // double *ini_length_a_device, *ini_length_b_device, *ini_length_c_device;
            // sizeof(double)*(num_inidividual_ini_extra[0]+2)

            // cudaMalloc((void**)&ini_length_a_device, sizeof(double)*(num_inidividual_ini_extra[0]+2));


            s1_frac2cart_ini<<<(int)(((num_inidividual_ini_extra[0]+2)*3-1)/running_block_size+1),running_block_size>>>
            (num_inidividual_ini_extra_device, 

            frac2car_a_device, frac2car_b_device, frac2car_c_device,

            ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device,
            ini_minimum_string_cart_x_device, ini_minimum_string_cart_y_device, ini_minimum_string_cart_z_device);

            // check_double_custom2<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
            // ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device);


            // check_double_custom2<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_string_cart_x_device, ini_minimum_string_cart_y_device, ini_minimum_string_cart_z_device, 
            // ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device);


            cal_length_prep_ini<<<(int)(((num_inidividual_ini_extra[0]+2)*3-1)/running_block_size+1),running_block_size>>>

            (num_inidividual_ini_extra_device, 

            ini_minimum_string_cart_x_device, ini_minimum_string_cart_y_device, ini_minimum_string_cart_z_device,
            ini_minimum_length_all_device);

            // check_double_ini<<<1,32>>>(((num_inidividual_ini_extra[0]+2)*3), ini_minimum_length_all_device);


            int *add_ini_h_offset = (int *) malloc(sizeof(int)*((num_inidividual_ini_extra[0]+2)+1));
            add_ini_h_offset[0] = 0;
            for (i=1; i<=(num_inidividual_ini_extra[0]+2); i++)
            {
                add_ini_h_offset[i] = i*3;
            }
            int *add_ini_d_offset;
            cudaMalloc((void**)&add_ini_d_offset, ((num_inidividual_ini_extra[0]+2)+1)*sizeof(int));
            cudaMemcpy(add_ini_d_offset, add_ini_h_offset, ((num_inidividual_ini_extra[0]+2)+1)*sizeof(int), cudaMemcpyHostToDevice);
            free(add_ini_h_offset);


            d_temp_storage = NULL;
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_all_device, ini_minimum_length_device, 
                (num_inidividual_ini_extra[0]+2), add_ini_d_offset, add_ini_d_offset+1);
            cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_all_device, ini_minimum_length_device, 
                (num_inidividual_ini_extra[0]+2), add_ini_d_offset, add_ini_d_offset+1);
            cudaFree(d_temp_storage);


            // check_double<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_length_device);


            ini_length_sqrt_cal<<<(int)(((num_inidividual_ini_extra[0]+2)-1)/running_block_size+1),running_block_size>>>

            (num_inidividual_ini_extra_device, ini_minimum_length_device);


            // check_double<<<1,32>>>((num_inidividual_ini_extra[0]+2), ini_minimum_length_device);

            int *sum_ini_h_offset = (int *) malloc(sizeof(int)*((1)+1));
            sum_ini_h_offset[0] = 0;
            for (i=1; i<=(1); i++)
            {
                sum_ini_h_offset[i] = (num_inidividual_ini_extra[0]+2);
            }
            int *sum_ini_d_offset;
            cudaMalloc((void**)&sum_ini_d_offset, ((1)+1)*sizeof(int));
            cudaMemcpy(sum_ini_d_offset, sum_ini_h_offset, ((1)+1)*sizeof(int), cudaMemcpyHostToDevice);
            free(sum_ini_h_offset);

            // check_int<<<1,32>>>(2, sum_ini_d_offset);

            d_temp_storage = NULL;
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_device, ini_minimum_l_abs_device, 
                (1), sum_ini_d_offset, sum_ini_d_offset+1);
            cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, ini_minimum_length_device, ini_minimum_l_abs_device, 
                (1), sum_ini_d_offset, sum_ini_d_offset+1);
            cudaFree(d_temp_storage);

            // check_double<<<1,32>>>(1, ini_minimum_l_abs_device);







            ini_2_s0<<<(int)(((N_string[0]*6)-1)/running_block_size+1),running_block_size>>>

            (num_inidividual_ini_extra_device, N_string_device, 


            ini_minimum_l_abs_device, 

            ini_minimum_string_a_device, ini_minimum_string_b_device, ini_minimum_string_c_device, 
            ini_minimum_string_alpha_device, ini_minimum_string_beta_device, ini_minimum_string_gamma_device, 

            temp_partition_device, ini_minimum_length_device, 

            s0_a_device, s0_b_device, s0_c_device, 
            s0_alpha_device, s0_beta_device, s0_gamma_device);










        }
        else
        {
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            printf("fatal error!!!!!\n");
            return 0;
        }
        
    }
    // check_double<<<1,32>>>(N_points, V_ini_test);
    // check_double_custom<<<1,32>>>(1272, minimum_distance_device, V_total_1);
    

    

    // check_double_custom2<<<1,32>>>(401, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device);
    // // print xyz
    // check_double_special<<<1,32>>>
    // (1, N_atom_adsorbate_device,

    // frac2car_a_device, frac2car_b_device, frac2car_c_device,

    // vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,

    // s0_a_device, s0_b_device, s0_c_device, 
    // s0_alpha_device, s0_beta_device, s0_gamma_device);
    // cudaDeviceSynchronize();
    // return 0;



    













    // free memory space used to calculate the potential energy on the side
    cudaFree(N_grid_device);
    cudaFree(N_angle_alpha_device);
    cudaFree(N_angle_beta_device);
    cudaFree(N_angle_gamma_device);
    cudaFree(delta_grid_device);
    cudaFree(delta_angle_device);

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
    cudaFree(V_total_1);





    
    // free memory space used to calculate the potential energy on the side
    cudaFree(cal_a_device);
    cudaFree(cal_b_device);
    cudaFree(cal_c_device);
    cudaFree(rot_alpha_rad_device);
    cudaFree(rot_beta_rad_device);
    cudaFree(rot_gamma_rad_device);



    // calculate energy for the potential along the string without calculating anything extra related to the derivative
    cudaMalloc((void **)&index_s0_cal_Vext_s0_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_adsorbate_cal_Vext_s0_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&index_frame_cal_Vext_s0_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&a_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&b_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&c_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&alpha_rad_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&beta_rad_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&gamma_rad_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_x_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_y_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&loc_z_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_x_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_y_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&vector_adsorbate_z_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_x_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_y_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&adsorbate_cart_z_rot_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_a_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_b_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&modify_frame_c_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&minimum_distance_cal_Vext_s0_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&V_s0_temp, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times);
    cudaMalloc((void **)&V_s0, sizeof(double)*N_string[0]);
    cudaMalloc((void **)&V_s2, sizeof(double)*N_string[0]);





    Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
    (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
    vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                times_x_device, times_y_device, times_z_device,
                cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                frac2car_a_device, frac2car_b_device, frac2car_c_device,
                cutoff_device,
                temp_add_frame_device,


                // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                // double *delta_grid_device, double *delta_angle_device,
                N_string_device,

                s0_a_device, s0_b_device, s0_c_device, 
                s0_alpha_device, s0_beta_device, s0_gamma_device,


                index_s0_cal_Vext_s0_device,
                // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                minimum_distance_cal_Vext_s0_device,
                V_s0_temp);

    h_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    h_offset[0] = 0;
    for (i=1; i<=N_string[0]; i++)
    {
        h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    }
    cudaMalloc((void**)&d_offset, (N_string[0]+1)*sizeof(int));
    cudaMemcpy(d_offset, h_offset, (N_string[0]+1)*sizeof(int), cudaMemcpyHostToDevice);
    free(h_offset);

    d_temp_storage = NULL;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaFree(d_temp_storage);
    // cudaFree(d_offset);

    // check_double<<<1,32>>>(N_string[0], V_s0);
    // check_double_custom4<<<1,32>>>(401, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);
    // cudaDeviceSynchronize();
    // return 0;


    // double *temp = (double *) malloc(sizeof(double)*N_string[0]);
    // cudaMemcpy(temp, V_s0, (N_string[0])*sizeof(double), cudaMemcpyDeviceToHost);
    // for (i=0; i<N_string[0]; i++)
    // {
    //     printf("%lf\n", temp[i]);
    // }

    // cudaFree(index_s0_cal_Vext_s0_device);
    // cudaFree(index_adsorbate_cal_Vext_s0_device);
    // cudaFree(index_frame_cal_Vext_s0_device);
    // cudaFree(a_cal_Vext_s0_device);
    // cudaFree(b_cal_Vext_s0_device);
    // cudaFree(c_cal_Vext_s0_device);
    // cudaFree(alpha_rad_cal_Vext_s0_device);
    // cudaFree(beta_rad_cal_Vext_s0_device);
    // cudaFree(gamma_rad_cal_Vext_s0_device);
    // cudaFree(loc_x_cal_Vext_s0_device);
    // cudaFree(loc_y_cal_Vext_s0_device);
    // cudaFree(loc_z_cal_Vext_s0_device);
    // cudaFree(vector_adsorbate_x_rot_cal_Vext_s0_device);
    // cudaFree(vector_adsorbate_y_rot_cal_Vext_s0_device);
    // cudaFree(vector_adsorbate_z_rot_cal_Vext_s0_device);
    // cudaFree(adsorbate_cart_x_rot_cal_Vext_s0_device);
    // cudaFree(adsorbate_cart_y_rot_cal_Vext_s0_device);
    // cudaFree(adsorbate_cart_z_rot_cal_Vext_s0_device);
    // cudaFree(modify_frame_a_cal_Vext_s0_device);
    // cudaFree(modify_frame_b_cal_Vext_s0_device);
    // cudaFree(modify_frame_c_cal_Vext_s0_device);
    // cudaFree(minimum_distance_cal_Vext_s0_device);
    // cudaFree(V_s0_temp);
    // cudaFree(V_s0);
    double s0_cart_x[1], s0_cart_y[1], s0_cart_z[1];
    double *s0_a_ini, *s0_b_ini, *s0_c_ini;
    double *s0_alpha_ini, *s0_beta_ini, *s0_gamma_ini;
    double *s0_a_final, *s0_b_final, *s0_c_final;
    double *s0_alpha_final, *s0_beta_final, *s0_gamma_final;
    double *s0_x, *s0_y, *s0_z;
    s0_a_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_b_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_c_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_alpha_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_beta_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_gamma_ini = (double *) malloc(sizeof(double)*N_string[0]);
    s0_a_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_b_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_c_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_alpha_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_beta_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_gamma_final = (double *) malloc(sizeof(double)*N_string[0]);
    s0_x = (double *) malloc(sizeof(double)*N_string[0]);
    s0_y = (double *) malloc(sizeof(double)*N_string[0]);
    s0_z = (double *) malloc(sizeof(double)*N_string[0]);

    cudaMemcpy(s0_a_ini, s0_a_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_b_ini, s0_b_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_c_ini, s0_c_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_alpha_ini, s0_alpha_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_beta_ini, s0_beta_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_gamma_ini, s0_gamma_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);

    double kb = 1.38e-23, T = 300;
    double *V_s0_1, *V_s0_2;
    double *V_s0_treated, *s0;
    double D_1, D_2;
    V_s0_1 = (double *) malloc(sizeof(double)*N_string[0]);
    V_s0_2 = (double *) malloc(sizeof(double)*N_string[0]);
    V_s0_treated = (double *) malloc(sizeof(double)*N_string[0]);
    s0 = (double *) malloc(sizeof(double)*N_string[0]);
    cudaMemcpy(V_s0_1, V_s0, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (i=0; i<N_string[0]; i++)
    {
        frac2car(s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], frac2car_a, frac2car_b, frac2car_c, s0_cart_x, s0_cart_y, s0_cart_z);
        s0_x[i] = s0_cart_x[0]*1e-10;
        s0_y[i] = s0_cart_y[0]*1e-10;
        s0_z[i] = s0_cart_z[0]*1e-10;
        // printf("%.5e %.5e %.5e\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i]);
        // printf("%.5e %.5e %.5e\n", s0_x[i], s0_y[i], s0_z[i]);
        // printf("%lf %lf %lf %lf %lf %lf\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], s0_alpha_ini[i], s0_beta_ini[i], s0_gamma_ini[i]);
    }
    for (i=0; i<N_string[0]; i++)
    {
        if (i==0)
        {
            s0[i] = 0;
        }
        else
        {
            s0[i] = s0[i-1] + sqrt( pow((s0_x[i]-s0_x[i-1]), 2) + pow((s0_y[i]-s0_y[i-1]), 2) + pow((s0_z[i]-s0_z[i-1]), 2) );
        }
         // = 1.0*i/(N_string[0]-1)*1e-10;
        if ((V_s0_1[i]/T)>6e2)
        {
            V_s0_treated[i] = exp(-6e2);
        }
        else
        {
            V_s0_treated[i] = exp(-V_s0_1[i]/T);
        }
        // printf("%.5e\n", V_s0_1[i]);
        
    }
    // printf("length: %.5e\n", s0[N_string[0]-1]);
    switch (direction[0])
    {
        case 1:
            D_1 = 0.5 * pow((La*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_1, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 2:
            D_1 = 0.5 * pow((Lb*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_1, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 3:
            D_1 = 0.5 * pow((Lc*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_1, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
    }
    // printf("%.5e\n", pow((Lb*1e-10), 2));
    // printf("%.5e\n", sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))));
    // printf("%.5e\n", exp( -max(V_s0_1, N_string[0])/T ));
    // printf("%.5e\n", trapz(s0, V_s0_treated, N_string[0]));


    // cudaDeviceSynchronize();
    // printf("test_D: %.5e\n", D_1);
    // return 0;





    // remap the string to the set that can calculate the partial derivative
    double *s0_deri_a_device, *s0_deri_b_device, *s0_deri_c_device;
    double *s0_deri_alpha_device, *s0_deri_beta_device, *s0_deri_gamma_device;
    int *s0_deri_index_string_device, *s0_deri_index_var_device;
    int *s0_deri_index_adsorbate_device, *s0_deri_index_frame_device;

    double *s0_deri_loc_x_device, *s0_deri_loc_y_device, *s0_deri_loc_z_device;
    double *s0_deri_vector_adsorbate_x_rot_device, *s0_deri_vector_adsorbate_y_rot_device, *s0_deri_vector_adsorbate_z_rot_device;
    double *s0_deri_adsorbate_cart_x_rot_device, *s0_deri_adsorbate_cart_y_rot_device, *s0_deri_adsorbate_cart_z_rot_device;
    double *s0_deri_modify_frame_a_device, *s0_deri_modify_frame_b_device, *s0_deri_modify_frame_c_device;
    double *s0_deri_minimum_distance_device;
    double *s0_deri_total_Vext_device, *s0_deri_Vext_device;

    double *s0_gradient_device, *s0_gradient_square_device, *s0_gradient_length_device;

    



    double *diff_s_coordinate_all_device, *diff_s_orientation_all_device;
    double *diff_s_coordinate_device, *diff_s_orientation_device;
    double *total_diff_s_coordinate_device, *total_diff_s_orientation_device;


    // double

    // double *a_s0_cal_device, *b_s0_cal_device, *c_s0_cal_device;
    // double *alpha_rad_s0_device, *beta_rad_s0_device, *gamma_rad_s0_device;




    cudaMalloc((void **)&s0_deri_a_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_b_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_c_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_alpha_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_beta_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_gamma_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_string_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_var_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_adsorbate_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_index_frame_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_loc_x_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_loc_y_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_loc_z_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_vector_adsorbate_x_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_vector_adsorbate_y_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_vector_adsorbate_z_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_adsorbate_cart_x_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_adsorbate_cart_y_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_adsorbate_cart_z_rot_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_modify_frame_a_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_modify_frame_b_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_modify_frame_c_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_minimum_distance_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_total_Vext_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    cudaMalloc((void **)&s0_deri_Vext_device, sizeof(double)*N_string[0]*7);
    cudaMalloc((void **)&s0_gradient_device, sizeof(double)*N_string[0]*6);
    cudaMalloc((void **)&s0_gradient_square_device, sizeof(double)*N_string[0]*6);
    cudaMalloc((void **)&s0_gradient_length_device, sizeof(double)*N_string[0]*2);



    cudaMalloc((void **)&diff_s_coordinate_all_device, sizeof(double)*N_string[0]*3);
    cudaMalloc((void **)&diff_s_orientation_all_device, sizeof(double)*N_string[0]*3);

    cudaMalloc((void **)&diff_s_coordinate_device, sizeof(double)*N_string[0]);
    cudaMalloc((void **)&diff_s_orientation_device, sizeof(double)*N_string[0]);
    cudaMalloc((void **)&total_diff_s_coordinate_device, sizeof(double));
    cudaMalloc((void **)&total_diff_s_orientation_device, sizeof(double));



    // parameter used for string method
    double rounding_coeff[1];
    double smooth_coeff[1];
    double *move_angle_rad_device, *move_frac_device;
    double *rounding_coeff_device;
    double *smooth_coeff_device;
    cudaMalloc((void **)&move_angle_rad_device, sizeof(double));
    cudaMalloc((void **)&move_frac_device, sizeof(double));
    cudaMalloc((void **)&rounding_coeff_device, sizeof(double));
    cudaMalloc((void **)&smooth_coeff_device, sizeof(double));
    // move_angle_degree[0] = 1.0;
    // move_frac[0] = 1e-4;
    move_angle_rad[0] = 1.0*move_angle_degree[0]/180*PI;
    rounding_coeff[0] = 1e-15;
    smooth_coeff[0] = 1e-4;

    cudaMemcpy(move_angle_rad_device, move_angle_rad, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(move_frac_device, move_frac, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rounding_coeff_device, rounding_coeff, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(smooth_coeff_device, smooth_coeff, sizeof(double), cudaMemcpyHostToDevice);
    // check_double<<<1,32>>>(1, move_angle_rad_device);
    // check_double<<<1,32>>>(1, move_frac_device);
    // check_double_sci<<<1,32>>>(1, rounding_coeff_device);
    // check_double_sci<<<1,32>>>(1, smooth_coeff_device);
    // cudaDeviceSynchronize();
    // return 0;



    int *V_deri_offset = (int *) malloc(sizeof(int)*(N_string[0]*7+1));
    V_deri_offset[0] = 0;
    for (i=1; i<=(N_string[0]*7); i++)
    {
        V_deri_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    }
    int *V_deri_offset_device;
    cudaMalloc((void**)&V_deri_offset_device, (N_string[0]*7+1)*sizeof(int));
    cudaMemcpy(V_deri_offset_device, V_deri_offset, (N_string[0]*7+1)*sizeof(int), cudaMemcpyHostToDevice);
    free(V_deri_offset);

    int *s0_gradient_offset = (int *) malloc(sizeof(int)*(N_string[0]*2+1));
    s0_gradient_offset[0] = 0;
    for (i=1; i<=(N_string[0]*2); i++)
    {
        s0_gradient_offset[i] = i*3;
    }
    int *s0_gradient_offset_device;
    cudaMalloc((void**)&s0_gradient_offset_device, sizeof(int)*(N_string[0]*2+1));
    cudaMemcpy(s0_gradient_offset_device, s0_gradient_offset, sizeof(int)*(N_string[0]*2+1), cudaMemcpyHostToDevice);
    free(s0_gradient_offset);


    int *s1_l_sum_1_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    s1_l_sum_1_offset[0] = 0;
    for (i=1; i<=(N_string[0]); i++)
    {
        s1_l_sum_1_offset[i] = i*6;
    }
    int *s1_l_sum_1_offset_device;
    cudaMalloc((void**)&s1_l_sum_1_offset_device, sizeof(int)*(N_string[0]+1));
    cudaMemcpy(s1_l_sum_1_offset_device, s1_l_sum_1_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    free(s1_l_sum_1_offset);

    int *s1_l_sum_2_offset = (int *) malloc(sizeof(int)*2);
    s1_l_sum_2_offset[0] = 0;
    s1_l_sum_2_offset[1] = N_string[0];
    int *s1_l_sum_2_offset_device;
    cudaMalloc((void**)&s1_l_sum_2_offset_device, sizeof(int)*2);
    cudaMemcpy(s1_l_sum_2_offset_device, s1_l_sum_2_offset, sizeof(int)*2, cudaMemcpyHostToDevice);
    free(s1_l_sum_2_offset);

    // int *s1_l_cumulation_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    // s1_l_cumulation_offset[0] = 0;
    // for (i=1; i<=(N_string[0]); i++)
    // {
    //     s1_l_cumulation_offset[i] = ((int) (i*(i+1)*0.5));
    // }
    // int *s1_l_cumulation_offset_device;
    // cudaMalloc((void**)&s1_l_cumulation_offset_device, sizeof(int)*(N_string[0]+1));
    // cudaMemcpy(s1_l_cumulation_offset_device, s1_l_cumulation_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    // free(s1_l_cumulation_offset);








    int *s1_l_sum_separate_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    s1_l_sum_separate_offset[0] = 0;
    for (i=1; i<=(N_string[0]); i++)
    {
        s1_l_sum_separate_offset[i] = i*3;
    }
    int *s1_l_sum_separate_offset_device;
    cudaMalloc((void**)&s1_l_sum_separate_offset_device, sizeof(int)*(N_string[0]+1));
    cudaMemcpy(s1_l_sum_separate_offset_device, s1_l_sum_separate_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    free(s1_l_sum_separate_offset);



    int *s1_l_sum_total_offset = (int *) malloc(sizeof(int)*2);
    s1_l_sum_total_offset[0] = 0;
    s1_l_sum_total_offset[1] = N_string[0];
    int *s1_l_sum_total_offset_device;
    cudaMalloc((void**)&s1_l_sum_total_offset_device, sizeof(int)*2);
    cudaMemcpy(s1_l_sum_total_offset_device, s1_l_sum_total_offset, sizeof(int)*2, cudaMemcpyHostToDevice);
    free(s1_l_sum_total_offset);



    int *s1_l_cumulation_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    s1_l_cumulation_offset[0] = 0;
    for (i=1; i<=(N_string[0]); i++)
    {
        s1_l_cumulation_offset[i] = ((int) (i*(i+1)*0.5));
    }
    int *s1_l_cumulation_offset_device;
    cudaMalloc((void**)&s1_l_cumulation_offset_device, sizeof(int)*(N_string[0]+1));
    cudaMemcpy(s1_l_cumulation_offset_device, s1_l_cumulation_offset, sizeof(int)*(N_string[0]+1), cudaMemcpyHostToDevice);
    free(s1_l_cumulation_offset);






    double convergence_coorindate[1];
    double convergence_orientation[1];
    int signal_coordinate[1];
    int signal_orientation[1];
    // printf("check: %lf %lf\n", 1.0*set_conv_trans_percent/100, 1.0*set_conv_rot_percent/100);
    convergence_coorindate[0] = 1.0*set_conv_trans_percent/100*move_frac[0]*N_string[0]*sqrt(3);
    convergence_orientation[0] = 1.0*set_conv_rot_percent/100*move_angle_rad[0]*N_string[0]*sqrt(3);
    signal_coordinate[0] = 0;
    signal_orientation[0] = 0;

    double *convergence_coorindate_device;
    double *convergence_orientation_device;
    int *signal_coordinate_device;
    int *signal_orientation_device;
    cudaMalloc((void **)&convergence_coorindate_device, sizeof(double));
    cudaMalloc((void **)&convergence_orientation_device, sizeof(double));
    cudaMalloc((void **)&signal_coordinate_device, sizeof(int));
    cudaMalloc((void **)&signal_orientation_device, sizeof(int));
    cudaMemcpy(convergence_coorindate_device, convergence_coorindate, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(convergence_orientation_device, convergence_orientation, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(signal_coordinate_device, signal_coordinate, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(signal_orientation_device, signal_orientation, sizeof(int), cudaMemcpyHostToDevice);
    // check_double<<<1,32>>>(1, convergence_coorindate_device);
    // check_double<<<1,32>>>(1, convergence_orientation_device);
    // check_int<<<1,32>>>(1, signal_coordinate_device);
    // check_int<<<1,32>>>(1, signal_orientation_device);
    // cudaDeviceSynchronize();
    // return 0;










    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    // all the following items would be included in the iteration/loop
    cudaDeviceSynchronize();
    // printf("start\n");
    t = clock();

    int time_set = set_running_step;
    int i_time;
    for (i_time=0; i_time<time_set; i_time++)
    {
        remap_string_var<<<(int)((N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, temp_add_frame_device,

                    N_string_device,

                    s0_a_device, s0_b_device, s0_c_device,
                    s0_alpha_device, s0_beta_device, s0_gamma_device,



                    s0_deri_a_device, s0_deri_b_device, s0_deri_c_device, 
                    s0_deri_alpha_device, s0_deri_beta_device, s0_deri_gamma_device,

                    s0_deri_index_string_device, s0_deri_index_var_device,
                    s0_deri_index_adsorbate_device, s0_deri_index_frame_device,


                    move_angle_rad_device, move_frac_device);







        Vext_s0_deri_cal<<<(int)((N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
        vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                    N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                    times_x_device, times_y_device, times_z_device,
                    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                    frac2car_a_device, frac2car_b_device, frac2car_c_device,
                    cutoff_device,
                    temp_add_frame_device,

                    N_string_device,

                    s0_deri_a_device, s0_deri_b_device, s0_deri_c_device,
                    s0_deri_alpha_device, s0_deri_beta_device, s0_deri_gamma_device,

                    s0_deri_index_adsorbate_device, s0_deri_index_frame_device,

                    s0_deri_loc_x_device, s0_deri_loc_y_device, s0_deri_loc_z_device,
                    s0_deri_vector_adsorbate_x_rot_device, s0_deri_vector_adsorbate_y_rot_device, s0_deri_vector_adsorbate_z_rot_device,
                    s0_deri_adsorbate_cart_x_rot_device, s0_deri_adsorbate_cart_y_rot_device, s0_deri_adsorbate_cart_z_rot_device,
                    s0_deri_modify_frame_a_device, s0_deri_modify_frame_b_device, s0_deri_modify_frame_c_device,
                    s0_deri_minimum_distance_device,

                    s0_deri_total_Vext_device);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_deri_total_Vext_device, s0_deri_Vext_device, 
            N_string[0]*7, V_deri_offset_device, V_deri_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_deri_total_Vext_device, s0_deri_Vext_device, 
            N_string[0]*7, V_deri_offset_device, V_deri_offset_device+1);
        cudaFree(d_temp_storage);







        s0_grad_cal<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (move_frac_device, move_angle_rad_device, rounding_coeff_device,
        N_string_device, s0_deri_Vext_device, s0_gradient_device, s0_gradient_square_device);

        // double *temp;
        // temp = (double *) malloc(sizeof(double)*N_string[0]*6);
        // cudaMemcpy(temp, s0_gradient_device, (N_string[0]*6)*sizeof(double), cudaMemcpyDeviceToHost);
        // for (i=0; i<N_string[0]; i++)
        // {
        //     printf("%lf %lf %lf %lf %lf %lf\n", temp[i*6+0], temp[i*6+1], temp[i*6+2], temp[i*6+3], temp[i*6+4], temp[i*6+5]);
        // }
        // return 0;


        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_gradient_square_device, s0_gradient_length_device, 
            N_string[0]*2, s0_gradient_offset_device, s0_gradient_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s0_gradient_square_device, s0_gradient_length_device, 
            N_string[0]*2, s0_gradient_offset_device, s0_gradient_offset_device+1);
        cudaFree(d_temp_storage);



        s0_grad_length_sqrt_cal<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, s0_gradient_length_device);

        // double *temp;
        // temp = (double *) malloc(sizeof(double)*N_string[0]*2);
        // cudaMemcpy(temp, s0_gradient_length_device, (N_string[0]*2)*sizeof(double), cudaMemcpyDeviceToHost);
        // for (i=0; i<N_string[0]; i++)
        // {
        //     printf("%lf %lf\n", temp[i*2+0], temp[i*2+1]);
        // }
        // return 0;





        s0_new_cal<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, 
        move_frac_device, move_angle_rad_device,




        s0_gradient_length_device, s0_gradient_device,
        s0_a_device, s0_b_device, s0_c_device, 
        s0_alpha_device, s0_beta_device, s0_gamma_device,
        s1_a_device, s1_b_device, s1_c_device,
        s1_alpha_device, s1_beta_device, s1_gamma_device);








        // check_double_custom2<<<1,32>>>(301, s1_a_device, s1_b_device, s1_c_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;


        s1_fix_modify_upgrade<<<(int)((N_string[0]-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        direction_device,

        s0_gradient_length_device, s0_gradient_device,
        s0_a_device, s0_b_device, s0_c_device, 
        s0_alpha_device, s0_beta_device, s0_gamma_device,
        s1_a_device, s1_b_device, s1_c_device,
        s1_alpha_device, s1_beta_device, s1_gamma_device);


        s1_frac2cart<<<(int)((N_string[0]-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        frac2car_a_device, frac2car_b_device, frac2car_c_device,

        s1_a_device, s1_b_device, s1_c_device,
        s1_cart_x_device, s1_cart_y_device, s1_cart_z_device);



        // check_double_custom2<<<1,32>>>(301, s1_cart_x_device, s1_cart_y_device, s1_cart_z_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // check_double_custom2<<<1,32>>>(401, s1_a_device, s1_b_device, s1_c_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;


        
        s1_length_prep<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        s1_cart_x_device, s1_cart_y_device, s1_cart_z_device,
        s1_alpha_device, s1_beta_device, s1_gamma_device,
        s1_length_coordinate_all_device, s1_length_orientation_all_device);



        // double *temp1, *temp2;
        // temp1 = (double *) malloc(sizeof(double)*N_string[0]*3);
        // temp2 = (double *) malloc(sizeof(double)*N_string[0]*3);
        // cudaMemcpy(temp1, s1_length_coordinate_all_device, (N_string[0]*3)*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(temp2, s1_length_orientation_all_device, (N_string[0]*3)*sizeof(double), cudaMemcpyDeviceToHost);
        // for (i=0; i<N_string[0]; i++)
        // {
        //     printf("%.3e %.3e %.3e %.3e %.3e %.3e\n", temp1[i*3+0], temp1[i*3+1], temp1[i*3+2], temp2[i*3+0], temp2[i*3+1], temp2[i*3+2]);
        // }
        // return 0;



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_all_device, s1_length_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_all_device, s1_length_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_all_device, s1_length_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_all_device, s1_length_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);



        // check_double_custom<<<1,32>>>(301, s1_length_coordinate_device, s1_length_orientation_device);
        // cudaDeviceSynchronize();
        // return 0;



        s1_length_sqrt_cal<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, s1_length_coordinate_device, s1_length_orientation_device);



        // check_double_custom<<<1,32>>>(301, s1_length_coordinate_device, s1_length_orientation_device);
        // cudaDeviceSynchronize();
        // return 0;


        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_device, s1_l_abs_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_device, s1_l_abs_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_device, s1_l_abs_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_device, s1_l_abs_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);



        // check_double_custom<<<1,32>>>(1, s1_l_abs_coordinate_device, s1_l_abs_orientation_device);
        // cudaDeviceSynchronize();
        // return 0;



        remap_s1_length_for_cumulation<<<(int)(((N_string[0]*(1+N_string[0])*0.5*2)-1)/running_block_size+1),running_block_size>>>
        (N_string_device, s1_length_coordinate_device, s1_length_orientation_device, 

        s1_length_coordinate_remap_device, s1_length_orientation_remap_device);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_remap_device, s1_length_coordinate_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_coordinate_remap_device, s1_length_coordinate_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_remap_device, s1_length_orientation_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, s1_length_orientation_remap_device, s1_length_orientation_cumulation_device, 
            N_string[0], s1_l_cumulation_offset_device, s1_l_cumulation_offset_device+1);
        cudaFree(d_temp_storage);



        // check_double_custom<<<1,32>>>(301, s1_length_coordinate_cumulation_device, s1_length_orientation_cumulation_device);
        // cudaDeviceSynchronize();
        // return 0;











        s1_2_s2<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (N_string_device, s1_l_abs_coordinate_device, s1_l_abs_orientation_device,
        s1_length_coordinate_cumulation_device, s1_length_orientation_cumulation_device, 

        s1_a_device, s1_b_device, s1_c_device, 
        s1_alpha_device, s1_beta_device, s1_gamma_device,

        s2_a_device, s2_b_device, s2_c_device,
        s2_alpha_device, s2_beta_device, s2_gamma_device);

        // check_double_temp<<<1,32>>>(401, s2_a_device, s2_b_device, s2_c_device);
        // check_double_temp<<<1,32>>>(401, s2_alpha_device, s2_beta_device, s2_gamma_device);
        // check_double_custom2<<<1,32>>>(401, s1_a_device, s1_b_device, s1_c_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;

        // check_double_custom2<<<1,32>>>(401, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device);
        // cudaDeviceSynchronize();
        // return 0;

        // calculate the potential of initial string
        Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
        vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                    N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                    times_x_device, times_y_device, times_z_device,
                    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                    frac2car_a_device, frac2car_b_device, frac2car_c_device,
                    cutoff_device,
                    temp_add_frame_device,


                    // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                    // double *delta_grid_device, double *delta_angle_device,
                    N_string_device,

                    s0_a_device, s0_b_device, s0_c_device, 
                    s0_alpha_device, s0_beta_device, s0_gamma_device,


                    index_s0_cal_Vext_s0_device,
                    // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                    // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                    index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                    a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                    alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                    loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                    vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                    adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                    modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                    minimum_distance_cal_Vext_s0_device,
                    V_s0_temp);
        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
            N_string[0], d_offset, d_offset+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
            N_string[0], d_offset, d_offset+1);
        cudaFree(d_temp_storage);

        // calculate the potential of moved string
        Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
        (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
        vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                    N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                    frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                    times_x_device, times_y_device, times_z_device,
                    cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                    frac2car_a_device, frac2car_b_device, frac2car_c_device,
                    cutoff_device,
                    temp_add_frame_device,


                    // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                    // double *delta_grid_device, double *delta_angle_device,
                    N_string_device,

                    s2_a_device, s2_b_device, s2_c_device, 
                    s2_alpha_device, s2_beta_device, s2_gamma_device,


                    index_s0_cal_Vext_s0_device,
                    // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                    // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                    index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                    a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                    alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                    loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                    vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                    adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                    modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                    minimum_distance_cal_Vext_s0_device,
                    V_s0_temp);
        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s2, 
            N_string[0], d_offset, d_offset+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s2, 
            N_string[0], d_offset, d_offset+1);
        cudaFree(d_temp_storage);



        

        check_s2<<<(int)((N_string[0]*3-1)/running_block_size+1),running_block_size>>>
        (N_string_device, V_s0, V_s2,
        s0_alpha_device, s0_beta_device, s0_gamma_device, 
        s2_alpha_device, s2_beta_device, s2_gamma_device);










        smooth_angle<<<(int)((N_string[0]*3-1)/running_block_size+1),running_block_size>>>
        (N_string_device, smooth_coeff_device, 
        s2_alpha_device, s2_beta_device, s2_gamma_device, 
        s2_alpha_smooth_device, s2_beta_smooth_device, s2_gamma_smooth_device);

        cudaDeviceSynchronize();
        // return 0;









        // calculate the difference in coordinate and orientation after one iteration
        diff_s_prep<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 

        s0_a_device, s0_b_device, s0_c_device,
        s0_alpha_device, s0_beta_device, s0_gamma_device,

        s2_a_device, s2_b_device, s2_c_device,
        s2_alpha_device, s2_beta_device, s2_gamma_device,

        diff_s_coordinate_all_device, diff_s_orientation_all_device);

        // if (i_time==8000)
        // {
        //     // check_double_custom5<<<1,32>>>
        //     // (401, s0_a_device, s0_b_device, s0_c_device,
        //     // s0_alpha_device, s0_beta_device, s0_gamma_device,

        //     // s2_a_device, s2_b_device, s2_c_device,
        //     // s2_alpha_device, s2_beta_device, s2_gamma_device);


        //     // check_double_custom7<<<1,32>>>
        //     // (401, s0_gradient_length_device, s0_gradient_device);


            

        


        //     cudaDeviceSynchronize();
        //     return 0;
        // }

        // if (i_time==8001)
        // {
        //     check_double_custom5<<<1,32>>>
        //     (401, s0_a_device, s0_b_device, s0_c_device,
        //     s0_alpha_device, s0_beta_device, s0_gamma_device,

        //     s2_a_device, s2_b_device, s2_c_device,
        //     s2_alpha_device, s2_beta_device, s2_gamma_device);


        //     check_double_custom7<<<1,32>>>
        //     (401, s0_gradient_length_device, s0_gradient_device);


            

        


        //     cudaDeviceSynchronize();
        //     return 0;
        // }


        // check_double_custom5<<<1,32>>>
        // (401, s0_a_device, s0_b_device, s0_c_device,
        // s0_alpha_device, s0_beta_device, s0_gamma_device,

        // s2_a_device, s2_b_device, s2_c_device,
        // s2_alpha_device, s2_beta_device, s2_gamma_device);

        // check_double_custom6<<<1,32>>>
        // (401, diff_s_coordinate_all_device, diff_s_orientation_all_device);

        // cudaDeviceSynchronize();
        // return 0;



        
        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_all_device, diff_s_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_all_device, diff_s_coordinate_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_all_device, diff_s_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_all_device, diff_s_orientation_device, 
            N_string[0], s1_l_sum_separate_offset_device, s1_l_sum_separate_offset_device+1);
        cudaFree(d_temp_storage);

        cudaDeviceSynchronize();
        // check_double_custom<<<1,32>>>
        // (401, diff_s_coordinate_device, diff_s_orientation_device);


        s1_length_sqrt_cal<<<(int)((N_string[0]*2-1)/running_block_size+1),running_block_size>>>
        (rounding_coeff_device, N_string_device, diff_s_coordinate_device, diff_s_orientation_device);


        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_device, total_diff_s_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_coordinate_device, total_diff_s_coordinate_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);



        d_temp_storage = NULL;
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_device, total_diff_s_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, diff_s_orientation_device, total_diff_s_orientation_device, 
            1, s1_l_sum_total_offset_device, s1_l_sum_total_offset_device+1);
        cudaFree(d_temp_storage);

        if ((i_time%200)==0)
        {
            // // string evolve output
            // check_double_special2<<<1,32>>>
            // (401, N_atom_adsorbate_device,

            // frac2car_a_device, frac2car_b_device, frac2car_c_device,

            // vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,

            // s0_a_device, s0_b_device, s0_c_device, 
            // s0_alpha_device, s0_beta_device, s0_gamma_device);
            // cudaDeviceSynchronize();
            // printf("\n");


            // // string evolve energy output
            // Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
            // (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
            // vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
            //             N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
            //             frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
            //             times_x_device, times_y_device, times_z_device,
            //             cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
            //             frac2car_a_device, frac2car_b_device, frac2car_c_device,
            //             cutoff_device,
            //             temp_add_frame_device,


            //             // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
            //             // double *delta_grid_device, double *delta_angle_device,
            //             N_string_device,

            //             s0_a_device, s0_b_device, s0_c_device, 
            //             s0_alpha_device, s0_beta_device, s0_gamma_device,


            //             index_s0_cal_Vext_s0_device,
            //             // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
            //             // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
            //             index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

            //             a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
            //             alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
            //             loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
            //             vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
            //             adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
            //             modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
            //             minimum_distance_cal_Vext_s0_device,
            //             V_s0_temp);

            // d_temp_storage = NULL;
            // cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
            //     N_string[0], d_offset, d_offset+1);
            // cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
            // cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
            //     N_string[0], d_offset, d_offset+1);
            // cudaFree(d_temp_storage);
            // // cudaFree(d_offset);
            // cudaDeviceSynchronize();

            // check_double_custom4<<<1,32>>>(401, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);
            // cudaDeviceSynchronize();

            // printf("\n");






















            // printf("%d\t", i_time);
            // print_convergence<<<1,32>>>
            // (1, total_diff_s_coordinate_device, total_diff_s_orientation_device, 
            //     convergence_coorindate_device, convergence_orientation_device);
            // check_double<<<1,32>>>(1, s1_l_abs_coordinate_device);
            // cudaDeviceSynchronize();
            // int_N_string = N_string[0];
            // check_double_custom4<<<1,32>>>(int_N_string, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);










            

            check_signal<<<(int)((2-1)/running_block_size+1),running_block_size>>>
            (N_string_device, 
            total_diff_s_coordinate_device, total_diff_s_orientation_device,
            convergence_coorindate_device, convergence_orientation_device,
            signal_coordinate_device, signal_orientation_device);

            cudaMemcpy(signal_coordinate, signal_coordinate_device, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(signal_orientation, signal_orientation_device, sizeof(int), cudaMemcpyDeviceToHost);



            // if (i_time==8000)
            // {
            //     check_double_custom4<<<1,32>>>(401, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);

            //     check_double_custom4<<<1,32>>>(401, s2_a_device, s2_b_device, s2_c_device, s2_alpha_device, s2_beta_device, s2_gamma_device, V_s0);

            //     cudaDeviceSynchronize();
            //     return 0;
            // }

            if ((signal_coordinate[0]==1)&&(signal_orientation[0]==1))
            {
                // printf("converged!\n");
                // printf("info:\t1\t%d\t%d\t", i_time, N_atom_frame[0]*times);
                break;
            }
            // check_double_angle1<<<1,32>>>
            // (401, s0_alpha_device, s0_beta_device, s0_gamma_device);
            // cudaDeviceSynchronize();
        }
        // check_double_angle1<<<1,32>>>
        // (401, s0_alpha_device, s0_beta_device, s0_gamma_device);
        // cudaDeviceSynchronize();
        // check_double_custom<<<1,32>>>
        // (1, total_diff_s_coordinate_device, total_diff_s_orientation_device);


        











        // check_double_temp<<<1,32>>>(401, s2_alpha_smooth_device, s2_beta_smooth_device, s2_gamma_smooth_device);
        // cudaDeviceSynchronize();
        // return 0;

        // check_signal<<<(int)((2-1)/running_block_size+1),running_block_size>>>
        // (N_string_device, 
        // total_diff_s_coordinate_device, total_diff_s_orientation_device,
        // convergence_coorindate_device, convergence_orientation_device,
        // signal_coordinate_device, signal_orientation_device);




        copy2s0<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        (N_string_device, 
        signal_coordinate_device, signal_orientation_device,

        s2_a_device, s2_b_device, s2_c_device, 
        s2_alpha_smooth_device, s2_beta_smooth_device, s2_gamma_smooth_device, 
        s0_a_device, s0_b_device, s0_c_device, 
        s0_alpha_device, s0_beta_device, s0_gamma_device);

        // copy2s0<<<(int)((N_string[0]*6-1)/running_block_size+1),running_block_size>>>
        // (N_string_device, 
        // signal_coordinate_device, signal_orientation_device,

        // s2_a_device, s2_b_device, s2_c_device, 
        // s2_alpha_device, s2_beta_device, s2_gamma_device, 
        // s0_a_device, s0_b_device, s0_c_device, 
        // s0_alpha_device, s0_beta_device, s0_gamma_device);

        // check_double_angle1<<<1,32>>>
        // (401, s0_alpha_device, s0_beta_device, s0_gamma_device);
        // cudaDeviceSynchronize();

        // printf("time: %d\n", i_time);
        // check_double_angle1<<<1,32>>>
        // (401, s0_alpha_device, s0_beta_device, s0_gamma_device);
        // cudaDeviceSynchronize();

        // check_double_angle2<<<1,32>>>
        // (401, s0_alpha_device, s0_beta_device, s0_gamma_device);
        // cudaDeviceSynchronize();






        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        




        // Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
        // (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
        // vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
        //             N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
        //             frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
        //             times_x_device, times_y_device, times_z_device,
        //             cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
        //             frac2car_a_device, frac2car_b_device, frac2car_c_device,
        //             cutoff_device,
        //             temp_add_frame_device,


        //             // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
        //             // double *delta_grid_device, double *delta_angle_device,
        //             N_string_device,

        //             s0_a_device, s0_b_device, s0_c_device, 
        //             s0_alpha_device, s0_beta_device, s0_gamma_device,


        //             index_s0_cal_Vext_s0_device,
        //             // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
        //             // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
        //             index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

        //             a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
        //             alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
        //             loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
        //             vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
        //             adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
        //             modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
        //             minimum_distance_cal_Vext_s0_device,
        //             V_s0_temp);
        // d_temp_storage = NULL;
        // cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        //     N_string[0], d_offset, d_offset+1);
        // cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
        // cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        //     N_string[0], d_offset, d_offset+1);
        // cudaFree(d_temp_storage);



        // check_double_angle3<<<1,32>>>
        // (401, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);
        // cudaDeviceSynchronize();












        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        // BIGGEST DEBUG
        cudaDeviceSynchronize();
        // if ((i_time%500)==0)
        // {
        //     printf("%d\n", i_time);
        // }
        
    }

    if ((signal_coordinate[0]==1)&&(signal_orientation[0]==1))
    {
        // printf("converged!\n");
        printf("info:\t1\t%d\t%d\t", i_time, N_atom_frame[0]*times);
    }
    else
    {
        // printf("timed out\n");
        printf("info:\t0\t%d\t%d\t", i_time, N_atom_frame[0]*times);
    }

    // return 0;


    Vext_cal_s0<<<(int)((N_string[0]*times*N_atom_adsorbate[0]*N_atom_frame[0]-1)/running_block_size+1),running_block_size>>>
    (N_atom_adsorbate_device, epsilon_adsorbate_device, sigma_adsorbate_device,
    vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,
                N_atom_frame_device, epsilon_frame_device, sigma_frame_device, 
                frac_a_frame_device, frac_b_frame_device, frac_c_frame_device,
                times_x_device, times_y_device, times_z_device,
                cart_x_extended_device, cart_y_extended_device, cart_z_extended_device,
                frac2car_a_device, frac2car_b_device, frac2car_c_device,
                cutoff_device,
                temp_add_frame_device,


                // int *N_grid_device, int *N_angle_alpha_device, int *N_angle_beta_device, int *N_angle_gamma_device,
                // double *delta_grid_device, double *delta_angle_device,
                N_string_device,

                s0_a_device, s0_b_device, s0_c_device, 
                s0_alpha_device, s0_beta_device, s0_gamma_device,


                index_s0_cal_Vext_s0_device,
                // int *index_a_cal_Vext_s0_device, int *index_b_cal_Vext_s0_device, int *index_c_cal_Vext_s0_device,
                // int *index_alpha_cal_Vext_s0_device, int *index_beta_cal_Vext_s0_device, int *index_gamma_cal_Vext_s0_device,
                index_adsorbate_cal_Vext_s0_device, index_frame_cal_Vext_s0_device,

                a_cal_Vext_s0_device, b_cal_Vext_s0_device, c_cal_Vext_s0_device,
                alpha_rad_cal_Vext_s0_device, beta_rad_cal_Vext_s0_device, gamma_rad_cal_Vext_s0_device,
                loc_x_cal_Vext_s0_device, loc_y_cal_Vext_s0_device, loc_z_cal_Vext_s0_device,
                vector_adsorbate_x_rot_cal_Vext_s0_device, vector_adsorbate_y_rot_cal_Vext_s0_device, vector_adsorbate_z_rot_cal_Vext_s0_device,
                adsorbate_cart_x_rot_cal_Vext_s0_device, adsorbate_cart_y_rot_cal_Vext_s0_device, adsorbate_cart_z_rot_cal_Vext_s0_device, 
                modify_frame_a_cal_Vext_s0_device, modify_frame_b_cal_Vext_s0_device, modify_frame_c_cal_Vext_s0_device,
                minimum_distance_cal_Vext_s0_device,
                V_s0_temp);

    // h_offset = (int *) malloc(sizeof(int)*(N_string[0]+1));
    // h_offset[0] = 0;
    // for (i=1; i<=N_string[0]; i++)
    // {
    //     h_offset[i] = i*N_atom_adsorbate[0]*N_atom_frame[0]*times;
    // }
    // cudaMalloc((void**)&d_offset, (N_string[0]+1)*sizeof(int));
    // cudaMemcpy(d_offset, h_offset, (N_string[0]+1)*sizeof(int), cudaMemcpyHostToDevice);
    // free(h_offset);

    d_temp_storage = NULL;
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, V_s0_temp, V_s0, 
        N_string[0], d_offset, d_offset+1);
    cudaFree(d_temp_storage);
    // cudaFree(d_offset);


    










    cudaDeviceSynchronize();
    // check_double_custom3<<<1,32>>>(201, s1_cart_x_device, s1_cart_y_device, s1_cart_z_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
    // check_double<<<1,32>>>(1, s1_l_abs_device);
    // check_double<<<1,32>>>(201, s1_length_device);
    // check_double<<<1,32>>>((N_string[0]*(N_string[0]+1)*0.5), s1_legnth_remap_device);
    // check_double<<<1,32>>>(201, s1_length_cumulation_device);
    // check_double_custom2<<<1,32>>>(201, s1_a_device, s1_b_device, s1_c_device, s1_alpha_device, s1_beta_device, s1_gamma_device);
    // check_double_custom2<<<1,32>>>(201, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device);

    check_double_custom4<<<1,32>>>(int_N_string, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);






    cudaDeviceSynchronize();
    // // print xyz
    // check_double_special<<<1,32>>>
    // (401, N_atom_adsorbate_device,

    // frac2car_a_device, frac2car_b_device, frac2car_c_device,

    // vector_adsorbate_x_device, vector_adsorbate_y_device, vector_adsorbate_z_device,

    // s0_a_device, s0_b_device, s0_c_device, 
    // s0_alpha_device, s0_beta_device, s0_gamma_device);
    // cudaDeviceSynchronize();

    t = clock() - t;

    // double *print_a, *print_b, *print_c;
    // double *print_alpha, *print_beta, *print_gamma;
    // double *print_Vext;



    // print_a = (double *) malloc(sizeof(double)*N_string[0]);
    // print_b = (double *) malloc(sizeof(double)*N_string[0]);
    // print_c = (double *) malloc(sizeof(double)*N_string[0]);
    // print_alpha = (double *) malloc(sizeof(double)*N_string[0]);
    // print_beta = (double *) malloc(sizeof(double)*N_string[0]);
    // print_gamma = (double *) malloc(sizeof(double)*N_string[0]);
    // print_Vext = (double *) malloc(sizeof(double)*N_string[0]);
    // cudaMemcpy(print_a, s0_a_device, N_string[0]*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(print_b, s0_b_device, N_string[0]*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(print_c, s0_c_device, N_string[0]*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(print_alpha, s0_alpha_device, N_string[0]*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(print_beta, s0_beta_device, N_string[0]*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(print_gamma, s0_gamma_device, N_string[0]*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(print_Vext, V_s0, N_string[0]*sizeof(double), cudaMemcpyDeviceToHost);








    
    cudaMemcpy(s0_a_final, s0_a_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_b_final, s0_b_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_c_final, s0_c_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_alpha_final, s0_alpha_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_beta_final, s0_beta_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(s0_gamma_final, s0_gamma_device, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(V_s0_2, V_s0, sizeof(double)*N_string[0], cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    for (i=0; i<N_string[0]; i++)
    {
        frac2car(s0_a_final[i], s0_b_final[i], s0_c_final[i], frac2car_a, frac2car_b, frac2car_c, s0_cart_x, s0_cart_y, s0_cart_z);
        s0_x[i] = s0_cart_x[0]*1e-10;
        s0_y[i] = s0_cart_y[0]*1e-10;
        s0_z[i] = s0_cart_z[0]*1e-10;
    }
    for (i=0; i<N_string[0]; i++)
    {
        if (i==0)
        {
            s0[i] = 0;
        }
        else
        {
            s0[i] = s0[i-1] + sqrt( pow((s0_x[i]-s0_x[i-1]), 2) + pow((s0_y[i]-s0_y[i-1]), 2) + pow((s0_z[i]-s0_z[i-1]), 2) );
        }

        if ((V_s0_2[i]/T)>6e2)
        {
            V_s0_treated[i] = exp(-6e2);
        }
        else
        {
            V_s0_treated[i] = exp(-V_s0_2[i]/T);
        }   
    }
    // printf("length: %.5e\n", s0[N_string[0]-1]);
    switch (direction[0])
    {
        case 1:
            D_2 = 0.5 * pow((La*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_2, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 2:
            D_2 = 0.5 * pow((Lb*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_2, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
        case 3:
            D_2 = 0.5 * pow((Lc*1e-10), 2) * sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))) *  
            (  exp( -max(V_s0_2, N_string[0])/T ) / trapz(s0, V_s0_treated, N_string[0]) );
            break;
    }
    // printf("%.5e\n", pow((Lb*1e-10), 2));
    // printf("%.5e\n", sqrt((kb*T)/(2*PI*(total_mass_adsorbate/1e3/6.02214076e23))));
    // printf("%.5e\n", exp( -max(V_s0_2, N_string[0])/T ));
    // printf("%.5e\n", trapz(s0, V_s0_treated, N_string[0]));

    // printf("test_D: %.5e\n", D_2);


    // return 0;







































    if (argc==3)
    {
        fp1 =fopen(argv[2], "w+");
        if (D_1>D_2)
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], s0_alpha_ini[i], s0_beta_ini[i], s0_gamma_ini[i], V_s0_1[i]);
            }
        }
        else
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_final[i], s0_b_final[i], s0_c_final[i], s0_alpha_final[i], s0_beta_final[i], s0_gamma_final[i], V_s0_2[i]);
            }
        }
        
        fclose(fp1);
    }
    else if (argc==4)
    {
        fp1 =fopen(argv[3], "w+");
        if (D_1>D_2)
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_ini[i], s0_b_ini[i], s0_c_ini[i], s0_alpha_ini[i], s0_beta_ini[i], s0_gamma_ini[i], V_s0_1[i]);
            }
        }
        else
        {
            for (i=0; i<N_string[0]; i++)
            {
                fprintf(fp1, "%lf %lf %lf %lf %lf %lf %lf\n", s0_a_final[i], s0_b_final[i], s0_c_final[i], s0_alpha_final[i], s0_beta_final[i], s0_gamma_final[i], V_s0_2[i]);
            }
        }
        fclose(fp1);
    }
    






    
    // check_double_custom4<<<1,32>>>(int_N_string, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device, V_s0);
    // printf("gpu time: %lf\n", ((double)t)/CLOCKS_PER_SEC);
    printf("%lf\n", ((double)t)/CLOCKS_PER_SEC);
    // double *temp;
    // temp = (double *) malloc(sizeof(double)*N_string[0]*2);
    // cudaMemcpy(temp, s0_gradient_length_device, (N_string[0]*2)*sizeof(double), cudaMemcpyDeviceToHost);
    // for (i=0; i<N_string[0]; i++)
    // {
    // //     // printf("%lf %lf %lf %lf %lf %lf\n", temp[i*6+0], temp[i*6+1], temp[i*6+2], temp[i*6+3], temp[i*6+4], temp[i*6+5]);
    //     printf("%lf %lf\n", temp[i*2+0], temp[i*2+1]);
    // }




    // check_double_custom2<<<1,32>>>(201, s0_a_device, s0_b_device, s0_c_device, s0_alpha_device, s0_beta_device, s0_gamma_device);
    // double *temp1, *temp2, *temp3, *temp4, *temp5, *temp6;
    // int *temp7, *temp8;
    // temp1 = (double *) malloc(sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // temp2 = (double *) malloc(sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // temp3 = (double *) malloc(sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // temp4 = (double *) malloc(sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // temp5 = (double *) malloc(sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // temp6 = (double *) malloc(sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // temp7 = (int *) malloc(sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // temp8 = (int *) malloc(sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7);
    // cudaMemcpy(temp1, s0_deri_a_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp2, s0_deri_b_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp3, s0_deri_c_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp4, s0_deri_alpha_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp5, s0_deri_beta_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp6, s0_deri_gamma_device, sizeof(double)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp7, s0_deri_index_adsorbate_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // cudaMemcpy(temp8, s0_deri_index_frame_device, sizeof(int)*N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7, cudaMemcpyDeviceToHost);
    // for (i=0; i<(N_string[0]*N_atom_adsorbate[0]*N_atom_frame[0]*times*7); i++)
    // {
    //     printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%d\t%d\n", temp1[i], temp2[i], temp3[i], temp4[i], temp5[i], temp6[i], temp7[i], temp8[i]);
    // }

}
