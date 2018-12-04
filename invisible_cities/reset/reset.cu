#include <stdio.h>

struct voxel{
	double x;
	double y;
	double E;
};

struct correction{
	double x;
	double y;
	double factor;
};

// Launch: block <1024, 1, 1>, grid <nslices,1>
// All voxels are initialized with "charge" and only those within 
// fiducial volume are set to be active
__global__ void create_voxels(voxel * voxels, 
		int * slice_start, double * xmins, double * xmaxs, 
		double * ymins, double * ymaxs, double * charges, double xsize, 
		double ysize, double rmax, bool * actives, int * address,
		int * slices_ids){

	int offset = slice_start[blockIdx.x];
	double charge = charges[blockIdx.x];

	int xmin = xmins[blockIdx.x];
	int xmax = xmaxs[blockIdx.x];
	int ymin = ymins[blockIdx.x];
	int ymax = ymaxs[blockIdx.x];
	// +1 to include last column/row
	int xsteps = (xmax - xmin) / xsize + 1;
	int ysteps = (ymax - ymin) / ysize + 1;

	int iterations = ceilf(1.f*(xsteps*ysteps)/blockDim.x);

	for(int i=0; i<iterations; i++){
		int vid = threadIdx.x + i*blockDim.x;
		double x = xmin + (vid / ysteps) * xsize;
		double y = ymin + (vid % ysteps) * xsize;

		if(x <= xmax && y <= ymax){
			bool active = sqrtf(x*x + y*y) < rmax;
			voxel * v = voxels + offset + vid;
			v->x = x;
			v->y = y;
			v->E = charge;
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
		slice_start[blockIdx.x] = address[start];
		if (blockIdx.x == 0){
			int lastSlice = slice_start_nc[gridDim.x];
			slice_start[gridDim.x] = address[lastSlice];
		}
	}

	// Compact vector
	for(int i=0; i<iterations; i++){
		int vidx = threadIdx.x + i*blockDim.x;
		int offset_in = start + vidx;
		if(offset_in < end && actives[offset_in]){
			int offset_out = address[offset_in];
			voxel * v_out = voxels + offset_out;
			voxel * v_in  = voxels_nc + offset_in;
			v_out->x = v_in->x;
			v_out->y = v_in->y;
			v_out->E = v_in->E;
			slice_ids[offset_out] = slice_ids_nc[offset_in];
		}   
	}   
}

// Launch grid <nslices, 1>, block <1024, 1, 1>
__global__ void create_anode_response(double * anode_response, int nsensors,
		int * sensors_ids, double * charges, int * slices_start){
	int start = slices_start[blockIdx.x];
	int end   = slices_start[blockIdx.x+1];
	int steps = end - start;
	int offset = nsensors * blockIdx.x;
	int iterations = ceilf(1.f*steps/blockDim.x);

	for(int i=0; i<iterations; i++){
		int step = threadIdx.x + i*blockDim.x;
		int sidx = start + step;
		if (step < steps){
			int sensor_pos = offset + sensors_ids[sidx];
			anode_response[sensor_pos] = charges[sidx];
		}
	}
}

__device__ void get_probability(double * prob, bool * active, voxel * voxels, 
		int vidx, int sidx,	double * xs, double * ys, double sensor_dist,
	    double xmin, double ymin,	double step,	int nbins, correction * corrections){

	//Compute distance and check condition
	double xdist = voxels[vidx].x - xs[sidx];
	double ydist = voxels[vidx].y - ys[sidx];
	*active = ((abs(xdist) <= sensor_dist) && (abs(ydist) <= sensor_dist));

	// Compute probability
	// In order to avoid accesing wrong parts of the memory 
	// if the sensor is not active for a particular voxel,
	// then we will use index 0.
	// Rounding: plus 0.5 and round down
	int xindex = __double2int_rd((xdist - xmin) / step * (*active) + 0.5f);
	int yindex = __double2int_rd((ydist - ymin) / step * (*active) + 0.5f);
	int prob_idx = xindex * nbins + yindex;
	*prob = corrections[prob_idx].factor;
}

// Launch block <1024,1,1>, grid < ceil(nvoxels/1024), 1>
__global__ void compute_active_sensors(double * probs, bool * active, int * address, int * sensor_ids,
		int * slice_ids, int * sensor_starts, bool * sensor_actives, int * sensor_starts_addr, 
		int * voxel_start, int nvoxels, int nsensors, int sensors_per_voxel, voxel * voxels,
	   	double sensor_dist, double * xs, double * ys, double step, int nbins, double xmin, double ymin,
	   	correction * corrections){

	int vidx = blockIdx.x * blockDim.x + threadIdx.x;
	int base_idx = vidx * sensors_per_voxel;
	int active_count = 0;

	//Check bounds
	if(vidx < nvoxels){
		int slice_id = slice_ids[vidx];

		for(int sidx=0; sidx<nsensors; sidx++){
			int global_sidx = nsensors * slice_id + sidx;
			int idx = base_idx + active_count;
			//Compute distance and get probability
			double prob;
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
	//The scan is excluvise (starting with 0 as neutral element)
	voxel_start[vidx] = active_count;
	}
}

__global__ void sensor_voxel_probs(double * probs, int * sensor_starts, 
		int * voxel_ids, int nsensors, int nslices, double * x_sensors,
	   	double * y_sensors, voxel * voxels, int * slice_start_nc,
	   	int * address_voxels, double sensor_dist, double * xmins, 
		double * xmaxs, double * ymins, double * ymaxs, double xsize, 
		double ysize, double p_xmin, double p_ymin, double step, int nbins,
	   	correction * corrections){

	//Compute sensor id and slice id
	int sid, slice;
	// Two blocks for 1792 sipms. 
	// First 896 in the even block, 2nd 896 in the odd block
	if(gridDim.x > nslices){
		slice = blockIdx.x / 4;
		sid = (blockIdx.x % 4) * blockDim.x + threadIdx.x;
	}else{
		slice = blockIdx.x;
		sid = threadIdx.x;
	}

	//Compute limits to iterate over voxels
	int xsteps = (xmaxs[slice] - xmins[slice]) / xsize + 1;
	int ysteps = (ymaxs[slice] - ymins[slice]) / ysize + 1;

	int xstart = (x_sensors[sid] - sensor_dist - xmins[slice]) / xsize;
	int xend   = (x_sensors[sid] + sensor_dist - xmins[slice]) / xsize;
	int ystart = (y_sensors[sid] - sensor_dist - ymins[slice]) / ysize;
	int yend   = (y_sensors[sid] + sensor_dist - ymins[slice]) / ysize;

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

	//Compute actual addresses
	int offset = slice_start_nc[slice];
	int start = offset + xstart * ysteps + ystart;
	int end   = offset + xend   * ysteps + yend;

	if(start <= end){
		start = address_voxels[start];
		end   = address_voxels[end];
	}

	int count = 0;
	int start_pos = sensor_starts[nsensors * slice + sid];
	for(int vidx = start; vidx <= end; vidx++){
		//Compute distance and get probability
		double prob;
		bool voxel_sensor;
		get_probability(&prob, &voxel_sensor, voxels, vidx, sid, 
				x_sensors, y_sensors, sensor_dist, 
				p_xmin, p_ymin, step, nbins, corrections);

		if(voxel_sensor){
			int pos = start_pos + count;
			probs[pos] = prob;
			voxel_ids[pos] = vidx;			

			count++;
		}
	}
}

// Launch grid<1,1> block <nslices+1, 1, 1>
__global__ void compact_slices(int * slice_start,
	   int * slice_start_nc, int * address, int sensors_per_voxel){
	int idx = slice_start_nc[threadIdx.x] * sensors_per_voxel;
	slice_start[threadIdx.x] = address[idx];
}

// Launch grid<100,1> block <1024, 1, 1>
__global__ void compact_probs(double * probs_in, double * probs_out, 
		double * forward_num, int * ids_in, int * ids_out, int * slice_ids,
	   	int * address, bool * actives, int size,
		int nsensors, int sensors_per_voxel, double * sensors_response){

	int iterations = ceilf(1.f*size / (blockDim.x*gridDim.x));
	int grid_base = blockIdx.x * blockDim.x * iterations;

	for(int i=0; i<iterations; i++){
		int block_base = i * blockDim.x;
		int offset = grid_base + block_base + threadIdx.x;
		if(offset < size){
			int addr = address[offset];  

			if(actives[offset]){
				double prob = probs_in[offset];
				int sensor_id = ids_in[offset];
				probs_out[addr] = prob;
				ids_out[addr]   = sensor_id;
				
				double response = sensors_response[sensor_id];
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
			int addr = address[offset];
			if(actives[offset]){
				starts_out[addr] = starts_in[offset];
				ids[addr] = offset;
			}
		}
	}
}

__global__ void forward_denom(double * denoms, int * sensor_starts,
	   	int * sensor_start_ids, double * sensor_probs,
	   	int * voxel_ids, voxel * voxels, int size){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size){
		int start = sensor_starts[id];
		int end = sensor_starts[id+1];

		double denom = 0;

		for(int i=start; i<end; i++){
			int vidx = voxel_ids[i];
			denom += voxels[vidx].E * sensor_probs[i];
		}

		int sidx = sensor_start_ids[id];
		denoms[sidx] = denom;
	}
}

__global__ void mlem_step(voxel * voxels, int * pmt_voxel_starts,
	   	double * pmt_probs, int * pmt_sensor_ids, double * pmt_nums, 
		double * pmt_denoms, int * sipm_voxel_starts, double * sipm_probs,
	   	int * sipm_sensor_ids, double * sipm_nums, double * sipm_denoms,
	   	int size){

	int vidx = blockIdx.x * blockDim.x + threadIdx.x;

	double pmt_eff      = 0;
	double pmt_fwd  = 0;
	double sipm_eff     = 0;
	double sipm_fwd = 0;

	if(vidx < size){
		// If sipms shouldn't be considered, the pointer would be null
		if (sipm_voxel_starts){
			int sipm_start = sipm_voxel_starts[vidx];
			int sipm_end   = sipm_voxel_starts[vidx+1];

			for(int i=sipm_start; i<sipm_end; i++){
				sipm_eff += sipm_probs[i];
				int sidx = sipm_sensor_ids[i];

				// Check for nans
				double value = sipm_nums[i] / sipm_denoms[sidx];
				if(isfinite(value)){
					sipm_fwd += value;
				}
			}
		}

		// If pmts shouldn't be considered, the pointer would be null
		if (pmt_voxel_starts){
			int pmt_start = pmt_voxel_starts[vidx];
			int pmt_end   = pmt_voxel_starts[vidx+1];

			for(int i=pmt_start; i<pmt_end; i++){
				pmt_eff += pmt_probs[i];
				int sidx = pmt_sensor_ids[i];

				// Check for nans
				double value = pmt_nums[i] / pmt_denoms[sidx];
				if(isfinite(value)){
					pmt_fwd += value;
				}
			}
		}

		if (sipm_eff > 0 && sipm_fwd > 0){
			voxels[vidx].E = voxels[vidx].E / (pmt_eff + sipm_eff ) * (pmt_fwd + sipm_fwd);
		}else{
			voxels[vidx].E = 0;
		}
	}
}
