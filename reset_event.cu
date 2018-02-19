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
//		printf("[%d][%d]: start: %d\n", blockIdx.x, threadIdx.x, offset);
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


// Launch block < xdim, 1, 1>, grid <1,1>
__global__ void compact_voxels(voxel * voxels_nc, voxel * voxels,
		int * address, bool * actives, int * slice_start_nc, int * slice_start){
	int start = slice_start_nc[blockIdx.x];
	int end   = slice_start_nc[blockIdx.x+1];
	int steps = end - start;
	int iterations = ceilf(1.f*steps/blockDim.x);
	if(threadIdx.x == 0){
//		printf("[%d], start %d, end %d, steps %d, iterations: %d\n", blockDim.x, start, end, steps, iterations);
		slice_start[blockIdx.x] = address[start] - 1;
		if (blockIdx.x == 0){
			int lastSlice = slice_start_nc[gridDim.x]-1;
			slice_start[gridDim.x] = address[lastSlice];
		}
	}

	// Compact vector
	for(int i=0; i<iterations; i++){
		int vidx = threadIdx.x + i*blockDim.x;
		int offset = start + vidx;
		if(offset < end && actives[offset]){
			voxel * v_out = voxels + address[offset] - 1;
			voxel * v_in  = voxels_nc + offset;
//			printf("[%d]: offset %d, out %d\n", blockIdx.x, offset, address[offset]);
			v_out->x = v_in->x;
			v_out->y = v_in->y;
			v_out->E = v_in->E;
		}   
	}   
}

// Launch grid <nslices, 1>, block <1024, 1, 1>
__global__ void create_anode_response(float * anode_response, int nsensors,
		int * sensors_ids, float * charges, int * slices_start){
	int start = slices_start[blockIdx.x];
	int end   = slices_start[blockIdx.x+1];
	int steps = end - start;
	int offset = nsensors * blockIdx.x;
	int iterations = ceilf(1.f*steps/blockDim.x);
//	printf("[%d]: iterations=%d, start=%d, end=%d, steps=%d, offset=%d\n", blockIdx.x, iterations, start, end, steps, offset);

	for(int i=0; i<iterations; i++){
		int step = threadIdx.x + i*blockDim.x;
		int sidx = start + step;
		if (step < steps){
			int sensor_pos = offset + sensors_ids[sidx];
//			printf("[%d]: iterations=%d, sidx=%d, sid=%d, charge=%f, pos=%d\n", blockIdx.x, iterations, sidx, sensors_ids[sidx], charges[sidx], sensor_pos);
			anode_response[sensor_pos] = charges[sidx];
		}
	}
}


// Launch block <1024,1,1>, grid < ceil(nvoxels/1024), 1>
__global__ void compute_active_sensors(float * probs, bool * active, int * address,
		int nvoxels, int nsensors, int sensors_per_voxel, voxel * voxels, float sensor_dist, 
		float * xs, float * ys, float step, int nbins, float xmin, float ymin,
	   	correction * corrections){
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("[%d][%d] id: %d, nsensors: %d\n", blockIdx.x, threadIdx.x, vidx, nsensors);

	int base_idx = vidx * sensors_per_voxel;
	int active_count = 0;

	//Check bounds
	if(vidx < nvoxels){
		for(int sidx=0; sidx<nsensors; sidx++){
			int idx = base_idx + active_count;
			float xdist = voxels[vidx].x - xs[sidx];
			float ydist = voxels[vidx].y - ys[sidx];

			bool voxel_sensor = ((abs(xdist) <= sensor_dist) &&
					(abs(ydist) <= sensor_dist));
			active_count += voxel_sensor;

			//Compute index
			active[idx]  = voxel_sensor;
			address[idx] = voxel_sensor;

			// Compute probability
			// In order to avoid accesing wrong parts of the memory 
			// if the sensor is not active for a particular voxel,
			// then we will use index 0.
			// Rounding: plus 0.5 and round down
			int xindex = __float2int_rd((xdist - xmin) / step * voxel_sensor + 0.5f);
			int yindex = __float2int_rd((ydist - ymin) / step * voxel_sensor + 0.5f);
			int prob_idx = xindex * nbins + yindex;

			probs[idx] = corrections[prob_idx].factor;
		}
	}
}
