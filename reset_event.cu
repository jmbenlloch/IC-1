#include <stdio.h>

struct voxel{
	float x;
	float y;
	float E;
};

struct segmented_scan{
	float value;
	int active;
};

struct mlem_scan{
	float eff;
	float projection;
	int flag;
};

struct correction{
	float x;
	float y;
	float factor;
};

// Launch: block <1024, 1, 1>, grid <nslices,1>
// All voxels are initialized with "charge" and only those within 
// fiducial volume are set to be active
__global__ void create_voxels(voxel * voxels, 
		int * slice_start, float * xmins, float * xmaxs, 
		float * ymins, float * ymaxs, float * charges, float xsize, 
		float ysize, float rmax, bool * actives, int * address){

	int offset = slice_start[blockIdx.x];
	float charge = charges[blockIdx.x];
	if(threadIdx.x == 0){
		printf("[%d][%d]: start: %d\n", blockIdx.x, threadIdx.x, offset);
	}

	int xmin = xmins[blockIdx.x];
	int xmax = xmaxs[blockIdx.x];
	int ymin = ymins[blockIdx.x];
	int ymax = ymaxs[blockIdx.x];
	int xsteps = (xmax - xmin) / xsize;
	int ysteps = (ymax - ymin) / ysize;

	int iterations = ceilf(1.f*(xsteps*ysteps)/blockDim.x);
//	printf("iterations: %d\n", iterations);

	for(int i=0; i<iterations; i++){
		int vid = threadIdx.x + i*blockDim.x;
		float x = xmin + (vid / ysteps) * xsize;
		float y = ymin + (vid % ysteps) * xsize;

		//TODO Check boundary condition
		if(x < xmax && y < ymax){
			bool active = sqrtf(x*x + y*y) < rmax;
			voxel * v = voxels + offset + vid;
//			printf("[%d][%d][%d]: offset %d, vid %d\n", blockIdx.x, threadIdx.x, i, offset, vid);
			v->x = x;
			v->y = y;
			v->E = charge;
			//printf("[%d][%d][%d]: pos: (%f, %f), steps: (%d, %d)\n", blockIdx.x, threadIdx.x, i, x, y, xsteps, ysteps);
//			printf("[%d][%d][%d]: pos: (%f, %f), steps: (%d, %d)\n", blockIdx.x, threadIdx.x, i, v->x, v->y, xsteps, ysteps);
			actives[offset + vid] = active;
			address[offset + vid] = active;
		}
	}

}
