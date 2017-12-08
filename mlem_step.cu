#include <stdio.h>

struct voxel {
    float x;
    float y;
    float E;
};

struct sensor {
	int id;
	float charge;
};

//Version 0.1: Use sipms and pmts
__global__ void mlem_step(voxel * voxels, voxel * voxels_out,
		sensor * anode_response, float * cath_response,
		float * pmt_prob, float * sipm_prob,
		bool * selection,
		int nvoxels, int nsipms, int npmts){

	//printf("[%d], x: %f, y: %f, e: %f\n", blockIdx.x, voxels[blockIdx.x].x, voxels[blockIdx.x].y, voxels[blockIdx.x].E);

	float efficiency = 0;

	// Add SiPM part (this can be stored on the first iteration)
	//Efficiency for sipms seems to be ok compared with python version (check dtype of sipm_prob...)
	for(int i=0; i<nsipms; i++){
	//	printf("sipm_prob[%d]: %f\n", blockIdx.x * nsipms + i, sipm_prob[blockIdx.x * nsipms + i]);
		//printf("[%d]: offset: %d\n", blockIdx.x, blockIdx.x * nsipms + i);
		if(selection[blockIdx.x * nsipms + i]){
			efficiency += sipm_prob[blockIdx.x * nsipms + i];
			//printf("eff: %f, new val: %f\n", efficiency, sipm_prob[blockIdx.x * nsipms + i]);
		}
	}
	// Add cathode part (just one pmt now)
	efficiency += pmt_prob[blockIdx.x];
	//printf("[%d] eff: %f\n", blockIdx.x, efficiency);


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
			if(isnan(denom)){
			//	printf("[%d], num: %f, den: %f, forward: %f\n", blockIdx.x, num, denom, anode_forward);
			}
		}
	}

	// Forward projection cathode
	float num = cath_response[0] * pmt_prob[blockIdx.x];
	float denom = 0;
	for(int j=0; j<nvoxels; j++){
		denom += voxels[j].E * pmt_prob[j];
	}
	float cathode_forward = num / denom;
//	printf("[%d], num: %f * %f =  %f, den: %f, cath_forward: %f\n", blockIdx.x, cath_response[0], pmt_prob[blockIdx.x], num, denom, cathode_forward);
	

	float result = voxels[blockIdx.x].E / efficiency * (anode_forward + cathode_forward);
	//printf("voxel %d, eff: %f, forward: %f, energy: %f\n", blockIdx.x, efficiency, anode_forward, result);

	voxel * v = voxels_out + blockIdx.x;
	v->x = voxels[blockIdx.x].x;
	v->y = voxels[blockIdx.x].y;
	if (voxels[blockIdx.x].E <= 0.){
		//printf("voxel %d, eff: %f, forward: %f, energy: %f\n", blockIdx.x, efficiency, anode_forward, result);
		v->E = 0.;
	}else{
		v->E = result;
	}


//	voxel * v = voxels_out + blockIdx.x;
//	v->x = voxels[blockIdx.x].x;
//	v->y = voxels[blockIdx.x].y;
//	v->E = voxels[blockIdx.x].E;
}

