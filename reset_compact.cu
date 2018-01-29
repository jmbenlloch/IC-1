#include <stdio.h>

struct voxel{
	float x;
	float y;
	float E;
};

struct correction{
	float x;
	float y;
	float factor;
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
	for(int idx=1; idx <= blockDim.x; idx <<= 1){
		int new_value; 
		if (idx <= threadIdx.x) new_value = address[pos] + address[pos - idx];
		__syncthreads();
		if (idx <= threadIdx.x) address[pos] = new_value;
//		printf("#########\n");
//		printf("[%d, %d] value: %d, idx: %d, pos: %d\n" , blockIdx.x, threadIdx.x, address[pos], idx, pos);
	}

	__syncthreads();

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
	for(int idx=1; idx < blockDim.x; idx<<=1){
		int value;
		if (idx <= threadIdx.x) value = offset[threadIdx.x] + offset[threadIdx.x - idx];
		__syncthreads();
		if (idx <= threadIdx.x) offset[threadIdx.x] = value;

//		printf("-[%d]: idx: %d, value: %d\n", threadIdx.x, idx, offset[threadIdx.x]);
	}

	__syncthreads();

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
__global__ void initialize_anode(float * sensors, float xmin, float xmax, float * xs, float ymin, float ymax, float * ys, float sipm_dist){
//	s->active = xs[id] > (xmin - sipm_dist) && xs[id] < (xmax + sipm_dist) &&
//		ys[id] > (ymin - sipm_dist) && ys[id] < (ymax + sipm_dist);
	sensors[blockIdx.x] = 0;

	//printf("[%d]: id=%d, charge=%f, active=%d\n", blockIdx.x, s->id, s->charge, s->active);
}

// Launch < #sensors in slice >
__global__ void create_anode_response(float * sensors, int * ids, float * charges){
	int id = ids[blockIdx.x];
	sensors[id] = charges[blockIdx.x];

	//  printf("[%d]: id=%d, %d, charge=%f, active=%d\n", blockIdx.x, id, s->id, s->charge, s->active);
}


// Launch < nvoxels >
// active[nvoxels][nsensors]  / probabilities[nvoxels][nsensors]
// scan[nvoxels][nsensors]
//TODO: Use threads within the block to avoid the for loop
__global__ void compute_active_sensors(bool * active, float * probabilities,
		int * scan, voxel * voxels, float * xs, float * ys,
		int nsensors, float sensor_dist,
		float step, int nbins, float xmin, float ymin, correction * corrections){
	int count = 0;
	for(int i=0; i<nsensors; i++){
		int idx = blockIdx.x * nsensors + i;
		float xdist = voxels[blockIdx.x].x - xs[i];
		float ydist = voxels[blockIdx.x].y - ys[i];

		bool voxel_sensor = ((abs(xdist) <= sensor_dist) &&
				(abs(ydist) <= sensor_dist));
		active[idx] = voxel_sensor;
		count += voxel_sensor;
		scan[idx] = count;

		// Compute probability
		// In order to avoid accesing wrong parts of the memory 
		// if the sensor is not active for a particular voxel,
		// then we will use index 0.
		// Rounding: plus 0.5 and round down
		int xindex = __float2int_rd((xdist - xmin) / step * voxel_sensor + 0.5f);
		int yindex = __float2int_rd((ydist - ymin) / step * voxel_sensor + 0.5f);
		int prob_idx = xindex * nbins + yindex;
		probabilities[idx] = corrections[prob_idx].factor;

		//      printf("[%d]: idx=%d, p=%f, pidx=%d, xindex=%d, %f, nbins=%d, yindex=%d, %f\n", blockIdx.x, idx, probabilities[idx], prob_idx, xindex, xdist, nbins, yindex, ydist);
		//      printf("[%d]: id=%d, xdist=%f, ydist=%f, active=%d\n", blockIdx.x, id, xdist, ydist, active[idx], probabilities[idx]);
	}
}

// Launch <nvoxels>
// offset[nvoxels]
__global__ void compact_probabilities(float * probs_in, float * probs_out,
		int * address, int * offset, int nsensors){

	int pos = blockIdx.x * nsensors - 1;
	if(blockIdx.x == 0){
		pos += 1;
	}

	offset[blockIdx.x] = address[pos];
	printf("[%d]: offset: %d, address: %d\n", blockIdx.x, pos, offset[pos]);

	//Scan offset vector
	for(int idx=1; idx < gridDim.x; idx<<=1){
		int value;
		if (idx <= blockIdx.x) value = offset[blockIdx.x] + offset[blockIdx.x - idx];
		__syncthreads();
		if (idx <= blockIdx.x) offset[blockIdx.x] = value;
//		printf("-[%d]: idx: %d, value: %d\n", threadIdx.x, idx, offset[threadIdx.x]);
	}

	__syncthreads();
	printf("[%d]: offset: %d\n", blockIdx.x, offset[pos]);

	for(int i=0; i<nsensors; i++){
		int in_idx = blockIdx.x * nsensors + i;
	}

}
