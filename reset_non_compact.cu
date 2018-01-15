#include <stdio.h>

struct voxel{
	float x;
	float y;
	float E;
	bool active;
};

// due to packing this is 12 bytes instead of 9, to be changed
struct sensor{
	int id;
	float charge;
	bool active;
};

struct correction{
	float x;
	float y;
	float factor;
};

//Launch: < (xmax - xmin)/xsize, (ymax - ymin)/ysize >
// All voxels are initialized with "charge" and only those within 
// fiducial volume are set to be active
//TODO: Compact
__global__ void create_voxels(voxel * voxels, float xmin, float xmax,
		float ymin, float ymax, float xsize, float ysize,
		float rmax, float charge){

	voxel * v = voxels + blockIdx.x * gridDim.y + blockIdx.y;
	float x = xmin + xsize * blockIdx.x;
	float y = ymin + ysize * blockIdx.y;

	// Check radial fiducial cut
	v->active = sqrtf(x*x + y*y) < rmax;
	v->x = x;
	v->y = y;
	v->E = charge;

	//printf("[%d, %d]: x=%f, y=%f, E=%f, active=%d\n", blockIdx.x, blockIdx.y, v->x, v->y, v->E, v->active);
}

// Launch <<< nsensors = 1792 >>>
__global__ void initialize_anode(sensor * sensors, 
		float xmin, float xmax, float * xs, 
		float ymin, float ymax, float * ys, 
		float sipm_dist){
	sensor * s = sensors + blockIdx.x;
	int id = blockIdx.x;

	s->active = xs[id] > (xmin - sipm_dist) && xs[id] < (xmax + sipm_dist) &&
		        ys[id] > (ymin - sipm_dist) && ys[id] < (ymax + sipm_dist);
	s->charge = 0;
	s->id = id;

//	printf("[%d]: id=%d, charge=%f, active=%d\n", blockIdx.x, s->id, s->charge, s->active);
}

// Launch < #sensors in slice >
// TODO: Compact
__global__ void create_anode_response(sensor * sensors, 
		int * ids, float * charges){
	int id = ids[blockIdx.x];
	sensor * s = sensors + id;
	s->charge = charges[blockIdx.x];

//	printf("[%d]: id=%d, %d, charge=%f, active=%d\n", blockIdx.x, id, s->id, s->charge, s->active);
}

// Launch < nvoxels >
// active[nvoxels][nsensors]  / probabilities[nvoxels][nsensors]
__global__ void compute_active_sensors(bool * active, float * probabilities,
		sensor * sensors, voxel * voxels, float * xs, float * ys,
		int nsensors, float sensor_dist,
		float step, int nbins, float xmin, float ymin, correction * corrections){
	for(int i=0; i<nsensors; i++){
		int idx = blockIdx.x * nsensors + i;
		int id  = sensors[i].id;

		float xdist = voxels[blockIdx.x].x - xs[id];
		float ydist = voxels[blockIdx.x].y - ys[id];

		bool voxel_sensor = ((abs(xdist) <= sensor_dist) && 
		                     (abs(ydist) <= sensor_dist));
		active[idx] = voxel_sensor;

		// Compute probability
		// In order to avoid accesing wrong parts of the memory 
		// if the sensor is not active for a particular voxel,
		// then we will use index 0.
		// Rounding: plus 0.5 and round down
		int xindex = __float2int_rd((xdist - xmin) / step * voxel_sensor + 0.5f);
		int yindex = __float2int_rd((ydist - ymin) / step * voxel_sensor + 0.5f);
		int prob_idx = xindex * nbins + yindex;
		probabilities[idx] = corrections[prob_idx].factor;
//		printf("[%d]: idx=%d, p=%f, pidx=%d, xindex=%d, %f, nbins=%d, yindex=%d, %f\n", blockIdx.x, idx, probabilities[idx], prob_idx, xindex, xdist, nbins, yindex, ydist);

//		printf("[%d]: id=%d, xdist=%f, ydist=%f, active=%d\n", blockIdx.x, id, xdist, ydist, active[idx], probabilities[idx]);

	}
}

__global__ void mlem_step(voxel * voxels, voxel * voxels_out,
		sensor * anode_response, float * cath_response,
		float * pmt_prob, float * sipm_prob,
		bool * sipm_selection, bool * pmt_selection,
		int nvoxels, int nsipms, int npmts){

	printf("[%d], active: %d, x: %f, y: %f, e: %f\n", blockIdx.x, voxels[blockIdx.x].active, voxels[blockIdx.x].x, voxels[blockIdx.x].y, voxels[blockIdx.x].E);

	float efficiency = 0;

	// Add SiPM part (this can be stored on the first iteration)
	for(int i=0; i<nsipms; i++){
		if(sipm_selection[blockIdx.x * nsipms + i]){
			efficiency += sipm_prob[blockIdx.x * nsipms + i];
		}
	}
	// Add cathode part (just one pmt now)
	for(int i=0; i<npmts; i++){
		if(pmt_selection[blockIdx.x * npmts + i]){
			efficiency += pmt_prob[blockIdx.x * npmts + i];
		}
	}

	// Forward projection anode
	float anode_forward = 0;
	for(int i=0; i<nsipms; i++){
		if(sipm_selection[blockIdx.x * nsipms + i]){
			float num = anode_response[i].charge * sipm_prob[blockIdx.x * nsipms + i];
			float denom = 0;
			for(int j=0; j<nvoxels; j++){
				if(sipm_selection[j * nsipms + i]){
					denom += voxels[j].E * sipm_prob[j * nsipms + i];
				}
			}
			anode_forward += num/denom;
		}
	}

	// Forward projection cathode
	float num = cath_response[0] * pmt_prob[blockIdx.x];
	float denom = 0;
	for(int j=0; j<nvoxels; j++){
		denom += voxels[j].E * pmt_prob[j];
	}
	float cathode_forward = num / denom;
	float result = voxels[blockIdx.x].E / efficiency * (anode_forward + cathode_forward);

	voxel * v = voxels_out + blockIdx.x;
	v->x = voxels[blockIdx.x].x;
	v->y = voxels[blockIdx.x].y;
	if (voxels[blockIdx.x].E <= 0.){
		v->E = 0.;
	}else{
		v->E = result;
	}
}

