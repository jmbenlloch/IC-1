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
// active[nvoxels][nsensors]
__global__ void compute_active_sensors(bool * active, sensor * sensors,
		voxel * voxels, float * xs, float * ys,
		int nsensors, float sipm_dist){
	for(int i=0; i<nsensors; i++){
		int idx = blockIdx.x * nsensors + i;
		int id  = sensors[i].id;

		float xdist = voxels[blockIdx.x].x - xs[id];
		float ydist = voxels[blockIdx.x].y - ys[id];

		active[idx] = ((abs(xdist) <= sipm_dist) && 
				       (abs(ydist) <= sipm_dist));

//		printf("[%d]: id=%d, xdist=%f, ydist=%f, active=%d\n", blockIdx.x, id, xdist, ydist, active[idx]);
	}
}
