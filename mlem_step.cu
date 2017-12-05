#include <stdio.h>

struct voxel {
    int x;
    int y;
    int E;
};

struct sensor {
	int id;
	float charge;
};

//Version 0.1: Only sipms
__global__ void mlem_step(voxel * voxels,
		sensor * anode_response, sensor * cath_response,
		float * pmt_prob, float * sipm_prob,
		bool * selection,
		int nvoxels, int nsipms, int npmts){

	float efficiency = 0;

	// Add SiPM part (this can be stored on the first iteration)
	for(int i=0; i<nsipms; i++){
		if(selection[blockIdx.x * nsipms + i]){
			efficiency += sipm_prob[blockIdx.x * nsipms + i];
		}
	}
	//TODO do the same for cathode

	// Forward projection anode
	float anode_forward = 0;
	for(int i=0; i<nsipms; i++){
		if(selection[blockIdx.x * nsipms + i]){
			float num = anode_response[i].charge * sipm_prob[blockIdx.x * nsipms + i];
			float denom = 0;
			for(int j=0; j<nvoxels; j++){
				if(selection[j * nsipms + i]){
					denom += voxels[j].E * sipm_prob[j * nsipms + i];
				}
			}
			anode_forward += num/denom;
		}
	}

	float result = voxels[blockIdx.x].E / efficiency * anode_forward;
	printf("voxel %d, eff: %f, forward: %f, energy: %f\n", blockIdx.x, efficiency, anode_forward, result);
}

