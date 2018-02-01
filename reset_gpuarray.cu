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

	//Write active voxels in their address
	if(active){
		voxel * v = voxels + offset + threadIdx.x;
		v->x = x;
		v->y = y;
		v->E = charge;
	}
}

// Launch block < xdim, 1, 1>, grid <1,1>
__global__ void compact_voxels(voxel * voxels, voxel * voxels_compact,
		int * address, bool * actives, int ydim){
	// Compact vector
	for(int i=0; i<ydim; i++){
		int offset  = threadIdx.x * ydim + i;
		if(actives[offset]){
			voxel * v_out = voxels_compact + address[offset] - 1;
			voxel * v_in  = voxels + offset;
//			printf("[%d]: compact: %d\tnon_compact: %d\n", threadIdx.x, offset[threadIdx.x] - 1, threadIdx.x + i);
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

// Launch block: < nsensors || nsensors//2, 1, 1) 
// grid < nvoxels, 1 >
// active[nvoxels][nsensors]  / probabilities[nvoxels][nsensors]
// scan[nvoxels][nsensors]
__global__ void compute_active_sensors_block(bool * active, float * probabilities,
		int * scan, voxel * voxels, float * xs, float * ys,
		int nsensors, float sensor_dist,
		float step, int nbins, float xmin, float ymin, correction * corrections){

	int base_addr = blockIdx.x * nsensors;
	for(int i=0; i<(nsensors/blockDim.x); i++){
		int sidx = threadIdx.x + i * blockDim.x;
		//now thread i will take i and i+1
		// probably due to cache is better to take i and i+blockDim.x
		int idx  = base_addr + sidx;
		//int id   = sensors[sidx].id;

		float xdist = voxels[blockIdx.x].x - xs[sidx];
		float ydist = voxels[blockIdx.x].y - ys[sidx];

		bool voxel_sensor = ((abs(xdist) <= sensor_dist) &&
				(abs(ydist) <= sensor_dist));
		active[idx] = voxel_sensor;
		scan[idx]   = voxel_sensor;

//		printf("[b: %d, t:%d]: i:%d, sidx:%d, active: %d\n", blockIdx.x, threadIdx.x, i, sidx, voxel_sensor);

		// Compute probability
		// In order to avoid accesing wrong parts of the memory 
		// if the sensor is not active for a particular voxel,
		// then we will use index 0.
		// Rounding: plus 0.5 and round down
		int xindex = __float2int_rd((xdist - xmin) / step * voxel_sensor + 0.5f);
		int yindex = __float2int_rd((ydist - ymin) / step * voxel_sensor + 0.5f);
		int prob_idx = xindex * nbins + yindex;

		probabilities[idx] = corrections[prob_idx].factor;

		//printf("[%d]: idx=%d, p=%f, pidx=%d, xindex=%d, %f, nbins=%d, yindex=%d, %f\n", blockIdx.x, idx, probabilities[idx], prob_idx, xindex, xdist, nbins, yindex, ydist);

//		printf("[%d]: id=%d, v=(%f, %f), s=(%f, %f), xd=%f, yd=%f, active=%d\n", blockIdx.x, sidx, voxels[blockIdx.x].x, voxels[blockIdx.x].y, xs[sidx], ys[sidx],xdist, ydist, active[idx], probabilities[idx]);

	}
}


// Launch block: < nsensors || nsensors//2, 1, 1) 
// grid < nvoxels, 1 >
__global__ void compact_probabilities(bool * active, int * address,
	   	float * probs_in, float * probs_out, int * voxel_start, 
		int * sensor_ids, int nvoxels, int nsensors){

	int base_addr = blockIdx.x * nsensors;
	if(blockIdx.x == 0){
		voxel_start[0] = 0;
		voxel_start[nvoxels] = address[nvoxels * nsensors-1];
	}else{
		voxel_start[blockIdx.x] = address[base_addr-1];
	}

//	printf("[%d, %d]: nsensors: %d, blockdim: %d, for %d\n", blockIdx.x, threadIdx.x, nsensors, blockDim.x, (nsensors/blockDim.x));
	for(int i=0; i<(nsensors/blockDim.x); i++){
		int sidx = threadIdx.x + i * blockDim.x;
		int idx = base_addr + sidx;
//		printf("[%d, %d]: sidx: %d, idx: %d\n", blockIdx.x, threadIdx.x, sidx, idx);
		if(active[idx]){
			// addresses are shifted one position due to scan
			int offset = address[idx] - 1;
			probs_out[offset] = probs_in[idx];
			sensor_ids[offset] = sidx;
		}
	}

}

// Launch block: < nvoxels || 1024, 1, 1) 
// grid < nvoxels, 1 >
__global__ void compact_probs_sensor(bool * active, int * address, 
		float * probs_in, float * probs_out,
	   	int * sensor_start, int * voxel_ids, int nvoxels, int nsensors){

	int base_addr = blockIdx.x * nvoxels;
	if(blockIdx.x == 0){
		sensor_start[0] = 0;
		sensor_start[nsensors] = address[nvoxels * nsensors-1];
	}else{
		sensor_start[blockIdx.x] = address[base_addr-1];
	}

	// If nvoxels % blockDim.x != 0 -> one more iteration
	int iterations = nvoxels / blockDim.x;
	iterations += (iterations * blockDim.x) < nvoxels;
	for(int i=0; i<iterations; i++){
		int vidx = threadIdx.x + i * blockDim.x;
		int idx = base_addr + vidx;
		if (vidx < nvoxels){
			if(active[idx]){
				// addresses are shifted one position due to scan
				int offset = address[idx] - 1;
				probs_out[offset] = probs_in[idx];
				voxel_ids[offset] = vidx;
			}
		}
	}
}

// Launch block: < nsensors || nsensors//2, 1, 1) 
// grid < nvoxels, 1 >
__global__ void transpose_probabilities(float * probs_in, float * probs_out,
		bool * active_in, bool * active_out,
		int * address_in, int * address_out,
		int nvoxels, int nsensors){
	int base_addr_in = blockIdx.x * nsensors;

	for(int i=0; i<(nsensors/blockDim.x); i++){
		int sidx = threadIdx.x + i*blockDim.x;
		int offset_in = base_addr_in + sidx;
		int offset_out = sidx * nvoxels + blockIdx.x;
		
		probs_out[offset_out]   = probs_in[offset_in];
		active_out[offset_out]  = active_in[offset_in];
		address_out[offset_out] = address_in[offset_in];
	}
}

// Arrays dimensions
// forward_projection[nsensors], voxels[nvoxels]
// probs[sensors, voxel]
// Launch block <1 , 1, 1>, grid <nsensors, 1>
__global__ void forward_projection(float * forward_projection,
		voxel * voxels, float * sensor_probs, int * sensor_start,
		int * voxel_ids){

	float denom = 0;
	// Parallelize this for
	//printf("[%d] start: %d, end: %d\n", blockIdx.x, sensor_start[blockIdx.x], sensor_start[blockIdx.x+1]);
	for(int i=sensor_start[blockIdx.x]; 
			i<sensor_start[blockIdx.x+1]; i++){
		int vidx = voxel_ids[i];
//		printf("[%d] i: %d, voxel: %d\n", blockIdx.x, i, vidx);
		denom += voxels[vidx].E * sensor_probs[i];
	}   
	forward_projection[blockIdx.x] = denom;
	//printf("forward[%d] = %f\n", blockIdx.x, forward_projection[blockIdx.x]);
}


__global__ void forward_projection_d(int sensor,
		float * forward_projection,	voxel * voxels, float * sensor_probs,
	    int * sensor_start,	int * voxel_ids){
	float denom = 0;
	// Parallelize this for
	for(int i=sensor_start[sensor]; 
			i<sensor_start[sensor+1]; i++){
		int vidx = voxel_ids[i];
		denom += voxels[vidx].E * sensor_probs[i];
	}   
	forward_projection[sensor] = denom;

}

// Arrays dimensions
// forward_projection[nsensors], voxels[nvoxels]
// probs[sensors, voxel]
// Launch block <1 , 1, 1>, grid <nsensors, 1>
__global__ void forward_projection_dynamic(float * forward_projection,
		voxel * voxels, float * sensor_probs, int * sensor_start,
		int * voxel_ids){
	forward_projection_d<<<1,1>>>(blockIdx.x, forward_projection, voxels, sensor_probs, sensor_start, voxel_ids);
}

// Launch block <1 , 1, 1>, grid <nvoxels, 1>
__global__ void mlem_step(voxel * voxels, voxel * voxels_out,
		float * sipm_forward, float * anode_response,
		float * sipm_probs, int * sipm_voxel_start, int * sipm_ids,
		float * pmt_forward, float * cath_response,
		float * pmt_probs, int * pmt_voxel_start, int * pmt_ids){

	float eff = 0;
	float anode_forward = 0;
	float cath_forward = 0;

	for(int i=sipm_voxel_start[blockIdx.x]; i<sipm_voxel_start[blockIdx.x+1]; i++){
		float prob = sipm_probs[i];
		eff += prob;

		int sensor_id = sipm_ids[i];
		float denom = sipm_forward[sensor_id];
		float num = anode_response[sensor_id] * prob;
		anode_forward += num/denom;
	}

	for(int i=pmt_voxel_start[blockIdx.x]; i<pmt_voxel_start[blockIdx.x+1]; i++){
		float prob = pmt_probs[i];
		eff += prob;

		int sensor_id = pmt_ids[i];
		float denom = pmt_forward[sensor_id];
		float num = cath_response[sensor_id] * prob;
		cath_forward += num/denom;
	}

	float result = voxels[blockIdx.x].E/eff * (anode_forward + cath_forward);
	voxels_out[blockIdx.x].E = result;

}
