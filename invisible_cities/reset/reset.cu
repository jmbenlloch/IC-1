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

// Launch: block <1024, 1, 1>, grid <nslices,1>
// All voxels are initialized with "charge" and only those within 
// fiducial volume are set to be active
__global__ void create_voxels(voxel * voxels, 
		int * slice_start, float * xmins, float * xmaxs, 
		float * ymins, float * ymaxs, float * charges, float xsize, 
		float ysize, float rmax, bool * actives, int * address,
		int * slices_ids){

	int offset = slice_start[blockIdx.x];
	float charge = charges[blockIdx.x];
//	if(threadIdx.x == 0){
//		printf("[%d][%d]: start: %d\n", blockIdx.x, threadIdx.x, offset);
//	}

	int xmin = xmins[blockIdx.x];
	int xmax = xmaxs[blockIdx.x];
	int ymin = ymins[blockIdx.x];
	int ymax = ymaxs[blockIdx.x];
	// +1 to include last column/row
	int xsteps = (xmax - xmin) / xsize + 1;
	int ysteps = (ymax - ymin) / ysize + 1;

	int iterations = ceilf(1.f*(xsteps*ysteps)/blockDim.x);
//	printf("iterations: %d\n", iterations);

	for(int i=0; i<iterations; i++){
		int vid = threadIdx.x + i*blockDim.x;
		float x = xmin + (vid / ysteps) * xsize;
		float y = ymin + (vid % ysteps) * xsize;

		if(x <= xmax && y <= ymax){
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
			slices_ids[offset + vid] = blockIdx.x;
		}
	}
}


// Launch block < xdim, 1, 1>, grid <1,1>
__global__ void compact_voxels(voxel * voxels_nc, voxel * voxels, 
		int * slice_ids_nc,	int * slice_ids, int * address, bool * actives,
	   	int * slice_start_nc, int * slice_start){
	int start = slice_start_nc[blockIdx.x];
	int end   = slice_start_nc[blockIdx.x+1];
	int steps = end - start;
	int iterations = ceilf(1.f*steps/blockDim.x);
	if(threadIdx.x == 0){
//		printf("[%d], start %d, end %d, steps %d, iterations: %d\n", blockDim.x, start, end, steps, iterations);
		//slice_start[blockIdx.x] = address[start] - 1;
		slice_start[blockIdx.x] = address[start];
		if (blockIdx.x == 0){
//			int lastSlice = slice_start_nc[gridDim.x]-1;
			int lastSlice = slice_start_nc[gridDim.x];
			slice_start[gridDim.x] = address[lastSlice];
		}
	}

	// Compact vector
	for(int i=0; i<iterations; i++){
		int vidx = threadIdx.x + i*blockDim.x;
		int offset_in = start + vidx;
		if(offset_in < end && actives[offset_in]){
//			int offset_out = address[offset_in] - 1;
			int offset_out = address[offset_in];
			voxel * v_out = voxels + offset_out;
			voxel * v_in  = voxels_nc + offset_in;
//			printf("[%d]: offset %d, out %d\n", blockIdx.x, offset, address[offset]);
			v_out->x = v_in->x;
			v_out->y = v_in->y;
			v_out->E = v_in->E;
			slice_ids[offset_out] = slice_ids_nc[offset_in];
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

__device__ void get_probability(float * prob, bool * active, voxel * voxels, 
		int vidx, int sidx,	float * xs, float * ys, float sensor_dist,
	    float xmin, float ymin,	float step,	int nbins, correction * corrections){

	//Compute distance and check condition
	float xdist = voxels[vidx].x - xs[sidx];
	float ydist = voxels[vidx].y - ys[sidx];
	*active = ((abs(xdist) <= sensor_dist) && (abs(ydist) <= sensor_dist));

	// Compute probability
	// In order to avoid accesing wrong parts of the memory 
	// if the sensor is not active for a particular voxel,
	// then we will use index 0.
	// Rounding: plus 0.5 and round down
	int xindex = __float2int_rd((xdist - xmin) / step * (*active) + 0.5f);
	int yindex = __float2int_rd((ydist - ymin) / step * (*active) + 0.5f);
	int prob_idx = xindex * nbins + yindex;
	*prob = corrections[prob_idx].factor;
}

// Launch block <1024,1,1>, grid < ceil(nvoxels/1024), 1>
__global__ void compute_active_sensors(float * probs, bool * active, int * address, int * sensor_ids,
		int * slice_ids, int * sensor_starts, bool * sensor_actives, int * sensor_starts_addr, 
		int * voxel_start, int nvoxels, int nsensors, int sensors_per_voxel, voxel * voxels,
	   	float sensor_dist, float * xs, float * ys, float step, int nbins, float xmin, float ymin,
	   	correction * corrections){
	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("[%d][%d] id: %d, nsensors: %d\n", blockIdx.x, threadIdx.x, vidx, nsensors);

	int base_idx = vidx * sensors_per_voxel;
	int active_count = 0;

	if(blockIdx.x == 0 && threadIdx.x == 0)
		printf("nvoxels: %d\n", nvoxels);

	//Check bounds
	if(vidx < nvoxels){
		int slice_id = slice_ids[vidx];

		for(int sidx=0; sidx<nsensors; sidx++){
			int global_sidx = nsensors * slice_id + sidx;
			int idx = base_idx + active_count;
			//Compute distance and get probability
			float prob;
			bool voxel_sensor;
			get_probability(&prob, &voxel_sensor, voxels, vidx, sidx, 
					xs, ys, sensor_dist, xmin, ymin, step, nbins, corrections);

			active_count += voxel_sensor;

			//Compute index
			active[idx]  = voxel_sensor;
			address[idx] = voxel_sensor;

			//Avoid extra read/write
			if(voxel_sensor){
				probs[idx] = prob;
				//sensor_ids[idx] = sidx;
				sensor_ids[idx] = global_sidx;
				//Increase the next one in order to get addresses after scan

				if(sidx == 1352 && vidx == 2){
					printf("voxels: [%d, %d] slice: %d, sidx: %d, voxel: %d, prob: %f\n", blockIdx.x, threadIdx.x, slice_id, sidx, vidx, prob);
				}

//				atomicAdd(last_position + nsensors*slice_id + sidx + 1, 1);
				//The scan is excluvise (starting with 0 as neutral element)
				//atomicAdd(sensor_starts + global_sidx + 1, 1);
				atomicAdd(sensor_starts + global_sidx, 1);
				sensor_starts_addr[global_sidx] = 1;
				sensor_actives[global_sidx] = 1;
			}

			// Stop if all relevant sensors for current voxel has been found
			if(active_count >= sensors_per_voxel){
				break;
			}
		}
//	voxel_start[vidx+1] = active_count;
	//The scan is excluvise (starting with 0 as neutral element)
	voxel_start[vidx] = active_count;
	}
}

__global__ void sensor_voxel_probs(float * probs, int * sensor_starts, 
		int * voxel_ids, int nsensors, int nslices, float * x_sensors,
	   	float * y_sensors, voxel * voxels, int * slice_start_nc,
	   	int * address_voxels, float sensor_dist, float * xmins, 
		float * xmaxs, float * ymins, float * ymaxs, float xsize, 
		float ysize, float p_xmin, float p_ymin, float step, int nbins,
	   	correction * corrections){

	//Compute sensor id and slice id
	int sid, slice;
	// Two blocks for 1792 sipms. 
	// First 896 in the even block, 2nd 896 in the odd block
	if(gridDim.x > nslices){
//	if(1){
		slice = blockIdx.x / 4;
		sid = (blockIdx.x % 4) * blockDim.x + threadIdx.x;
		// Hack to set slice 1
//		slice = 3;
//		sid = (blockIdx.x % 4) * blockDim.x + threadIdx.x;
	}else{
		slice = blockIdx.x;
		sid = threadIdx.x;
	}

	//Compute limits to iterate over voxels
	int xsteps = (xmaxs[slice] - xmins[slice]) / xsize + 1;
	int ysteps = (ymaxs[slice] - ymins[slice]) / ysize + 1;
//	printf("xsteps: %d, ysteps: %d\n", xsteps, ysteps);

	int xstart = (x_sensors[sid] - sensor_dist - xmins[slice]) / xsize;
	int xend   = (x_sensors[sid] + sensor_dist - xmins[slice]) / xsize;
	int ystart = (y_sensors[sid] - sensor_dist - ymins[slice]) / ysize;
	int yend   = (y_sensors[sid] + sensor_dist - ymins[slice]) / ysize;

	if(blockIdx.x == 263 && threadIdx.x == 38){
		printf("xs: %f, ys: %f, dist: %f, xmin: %f, ymin: %f, xsize: %f, ysize: %f\n", x_sensors[sid], y_sensors[sid], sensor_dist, xmins[slice], ymins[slice], xsize, ysize);
		printf("x: (%d, %d), y: (%d, %d), steps: (%d, %d)\n", xstart, xend, ystart, yend, xsteps, ysteps);
	}

	//Correct limit if we are past the borders
	if(xstart < 0){
		xstart = 0;
	}
	if(xend >= xsteps){
		xend = xsteps - 1;
	}
	if(ystart < 0){
		ystart = 0;
	}
	if(yend >= ysteps){
		yend = ysteps - 1;
	}

	if(blockIdx.x == 263 && threadIdx.x == 38){
		printf("x: (%d, %d), y: (%d, %d), steps: (%d, %d)\n", xstart, xend, ystart, yend, xsteps, ysteps);
	}

	//Compute actual addresses
	int offset = slice_start_nc[slice];
	int start = offset + xstart * ysteps + ystart;
	int end   = offset + xend   * ysteps + yend;
	if(blockIdx.x == 263 && threadIdx.x == 38){
		printf("sensor [%d, %d] start: %d, end: %d\n", blockIdx.x, threadIdx.x, start, end);
	}

	if(start <= end){
		start = address_voxels[start];
		end   = address_voxels[end];
	}

	int count = 0;
//	printf("[%d, %d] slice: %d, sidx: %d, voxels: %d, %d\n", blockIdx.x, threadIdx.x, slice, sid, start, end);
	int start_pos = sensor_starts[nsensors * slice + sid];
	for(int vidx = start; vidx <= end; vidx++){
		//Compute distance and get probability
		float prob;
		bool voxel_sensor;
		get_probability(&prob, &voxel_sensor, voxels, vidx, sid, 
				x_sensors, y_sensors, sensor_dist, 
				p_xmin, p_ymin, step, nbins, corrections);

			if(blockIdx.x == 263 && threadIdx.x == 38){
				printf("sensor [%d, %d] count: %d, slice: %d, sidx: %d, voxel: %d\n", blockIdx.x, threadIdx.x, count, slice, sid, vidx);
				printf("sensor [%d, %d] start: %d, end: %d\n", blockIdx.x, threadIdx.x, start, end);
			}

		if(voxel_sensor){
			int pos = start_pos + count;
			probs[pos] = prob;
			voxel_ids[pos] = vidx;			

//			if(sid == 1351){
//				printf("sensor [%d, %d] count: %d, slice: %d, sidx: %d, voxel: %d\n", blockIdx.x, threadIdx.x, count, slice, sid, vidx);
//			}
			if(sid == 1352 && vidx == 2){
				printf("sensor: [%d, %d] slice: %d, sidx: %d, voxel: %d, prob: %f\n", blockIdx.x, threadIdx.x, slice, sid, vidx, prob);
			}

			count++;
		}
	}
//	printf("[%d, %d] count: %d, slice: %d, sidx: %d, voxels: %d, %d\n", blockIdx.x, threadIdx.x, count, slice, sid, start, end);
}

// Launch grid<1,1> block <nslices+1, 1, 1>
__global__ void compact_slices(int * slice_start,
	   int * slice_start_nc, int * address, int sensors_per_voxel){
	int idx = slice_start_nc[threadIdx.x] * sensors_per_voxel;
	slice_start[threadIdx.x] = address[idx];
}

// Launch grid<100,1> block <1024, 1, 1>
__global__ void compact_probs(float * probs_in, float * probs_out, 
		float * forward_num, int * ids_in, int * ids_out, int * slice_ids,
	   	int * address, bool * actives, int size,
		int nsensors, int sensors_per_voxel, float * sensors_response){

	int iterations = ceilf(1.f*size / (blockDim.x*gridDim.x));
	int grid_base = blockIdx.x * blockDim.x * iterations;

	for(int i=0; i<iterations; i++){
		int block_base = i * blockDim.x;
		int offset = grid_base + block_base + threadIdx.x;
		if(offset < size){
			int addr = address[offset];  

			if(actives[offset]){
				float prob = probs_in[offset];
				int sensor_id = ids_in[offset];
				probs_out[addr] = prob;
				ids_out[addr]   = sensor_id;
				
				float response = sensors_response[sensor_id];
				forward_num[addr] = response * prob;
			}
		}
	}
}

__global__ void compact_sensor_start(int * starts_in, int * starts_out, 
		int * ids, int * address, bool * actives, int size){
	int iterations = ceilf(1.f*size / (blockDim.x*gridDim.x));
	int grid_base = blockIdx.x * blockDim.x * iterations;

	// First thread will write also the last item
	// (end position for last active sensor)
	if(threadIdx.x == 0 && blockIdx.x ==0){
		int addr = address[size-1];
		starts_out[addr] = starts_in[size-1];
		ids[addr] = size-1;
	}

	for(int i=0; i<iterations; i++){
		int block_base = i * blockDim.x;
		int offset = grid_base + block_base + threadIdx.x;
		if(offset < size){
			//int addr = address[offset] - 1;  
			int addr = address[offset];
			if(actives[offset]){
				starts_out[addr] = starts_in[offset];
				ids[addr] = offset;
			}
		}
	}
}

__global__ void forward_denom(float * denoms, int * sensor_starts,
	   	int * sensor_start_ids, float * sensor_probs,
	   	int * voxel_ids, voxel * voxels, int size){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size){
		int start = sensor_starts[id];
		int end = sensor_starts[id+1];

		float denom = 0;

		for(int i=start; i<end; i++){
			int vidx = voxel_ids[i];
			denom += voxels[vidx].E * sensor_probs[i];
		}

		int sidx = sensor_start_ids[id];
		denoms[sidx] = denom;
	}
}

__global__ void mlem_step(voxel * voxels, int * pmt_voxel_starts,
	   	float * pmt_probs, int * pmt_sensor_ids, float * pmt_nums, 
		float * pmt_denoms, int * sipm_voxel_starts, float * sipm_probs,
	   	int * sipm_sensor_ids, float * sipm_nums, float * sipm_denoms,
	   	int size){

	int vidx = blockIdx.x * blockDim.x + threadIdx.x;

	float pmt_eff      = 0;
	float pmt_fwd  = 0;
	float sipm_eff     = 0;
	float sipm_fwd = 0;

	if(vidx < size){
		int sipm_start = sipm_voxel_starts[vidx];
		int sipm_end   = sipm_voxel_starts[vidx+1];

		for(int i=sipm_start; i<sipm_end; i++){
			sipm_eff += sipm_probs[i];
			int sidx = sipm_sensor_ids[i];

			// Check for nans
			float value = sipm_nums[i] / sipm_denoms[sidx];
			if(isfinite(value)){
				sipm_fwd += value;
			}
		}

		int pmt_start = pmt_voxel_starts[vidx];
		int pmt_end   = pmt_voxel_starts[vidx+1];

		for(int i=pmt_start; i<pmt_end; i++){
			pmt_eff += pmt_probs[i];
			int sidx = pmt_sensor_ids[i];

			// Check for nans
			float value = pmt_nums[i] / pmt_denoms[sidx];
			if(isfinite(value)){
				pmt_fwd += value;
			}
		}

		voxels[vidx].E = voxels[vidx].E / (pmt_eff + sipm_eff ) * (pmt_fwd + sipm_fwd);
	}
}
