#include <stdio.h>

struct voxel{
	float x;
	float y;
	float E;
};

// due to packing this is 12 bytes instead of 9, to be changed
struct sensor{
	int id;
	float charge;
	bool active;
};


// Launch: block < (xmax - xmin)/xsize, (ymax - ymin)/ysize >, grid <1,1>
// All voxels are initialized with "charge" and only those within 
// fiducial volume are set to be active

// Has to be one block and not one grid. We loose synchronization.
// Maximum 2k threads. We have up to ~30k voxels. How to map them to threads?
__global__ void create_voxels_compact(voxel * voxels, int * address,
		bool * actives, float xmin, float xmax, float ymin, float ymax, 
		float xsize, float ysize, float rmax, float charge){

	int offset = blockIdx.x * blockDim.x;
	float x = xmin + xsize * blockIdx.x;
	float y = ymin + ysize * threadIdx.x;
	bool active = sqrtf(x*x + y*y) < rmax;

	// Mem layout: voxels[x][y]
	int pos = offset + threadIdx.x;
	address[pos] = active;
	actives[pos] = active;
//	printf("[%d, %d] address[%d]: %d\n", blockIdx.x, threadIdx.x, pos, address[pos]);
	__syncthreads();

	// Scan algoritm (Hillis-Steele)
	for(int idx=1; idx <= threadIdx.x; idx <<= 1){
		int new_value = address[pos] + address[pos - idx];
		__syncthreads();
		address[pos] = new_value;
//		printf("#########\n");
//		printf("[%d, %d] value: %d, idx: %d, pos: %d\n" , blockIdx.x, threadIdx.x, address[pos], idx, pos);
	}

	//Write active voxels in their address
	// Addresses are shifted 1 position due to scan algorithm
	if(active){
		voxel * v = voxels + offset + address[pos] - 1;
		v->x = x;
		v->y = y;
		v->E = charge;
	}
}


// Launch block < xdim, 1, 1>, grid <1,1>
__global__ void compact_voxels(voxel * voxels, voxel * voxels_compact,
		int * address, bool * actives, int ydim){
	extern __shared__ int offset[];

	int pos = threadIdx.x * ydim - 1;
	if(threadIdx.x == 0){
		pos += 1;
	}

//	printf("[%d]: address: %d\n", threadIdx.x, pos);

	offset[threadIdx.x] = address[pos];
//	printf("[%d]: %d\n", threadIdx.x, offset[threadIdx.x]);

	//Scan offset vector
	for(int idx=1; idx <= threadIdx.x; idx<<=1){
		int value = offset[threadIdx.x] + offset[threadIdx.x - idx];
		__syncthreads();
		offset[threadIdx.x] = value;

//		printf("-[%d]: idx: %d, value: %d\n", threadIdx.x, idx, offset[threadIdx.x]);
	}

//	printf("scan [%d]: %d\n", threadIdx.x, offset[threadIdx.x]);

	// Compact vector
	for(int i=0; i<ydim; i++){
		int offset_in = threadIdx.x * ydim + i;
//		printf("[%d]: offset_in: %d, active: %d\n", threadIdx.x, offset_in, actives[offset_in]);
		if(actives[offset_in]){
			voxel * v_out = voxels_compact + offset[threadIdx.x] - 1 + i;
			voxel * v_in  = voxels + offset_in;
//			printf("[%d]: compact: %d\tnon_compact: %d\n", threadIdx.x, offset[threadIdx.x] - 1 + i, threadIdx.x + i);
			v_out->x = v_in->x;
			v_out->y = v_in->y;
			v_out->E = v_in->E;
		}
	}
}

// Launch <<< nsensors = 1792 >>>
__global__ void initialize_anode(sensor * sensors, float xmin, float xmax, float * xs, float ymin, float ymax, float * ys, float sipm_dist){
	sensor * s = sensors + blockIdx.x;
	int id = blockIdx.x;

	s->active = xs[id] > (xmin - sipm_dist) && xs[id] < (xmax + sipm_dist) &&
		ys[id] > (ymin - sipm_dist) && ys[id] < (ymax + sipm_dist);
	s->charge = 0;
	s->id = id; 

	//printf("[%d]: id=%d, charge=%f, active=%d\n", blockIdx.x, s->id, s->charge, s->active);
}

// Launch < #sensors in slice >
__global__ void create_anode_response(sensor * sensors, int * ids, float * charges){
	int id = ids[blockIdx.x];
	sensor * s = sensors + id; 
	s->charge = charges[blockIdx.x];

	//  printf("[%d]: id=%d, %d, charge=%f, active=%d\n", blockIdx.x, id, s->id, s->charge, s->active);
}

