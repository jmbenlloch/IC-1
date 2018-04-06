from pytest import fixture

#@fixture(scope="session")
#def serial():
rst_voxels = rst_serial.create_voxels(voxels_data, slices_start, x_size, y_size, rmax)
nslices = voxels_data.xmin.shape[0]
anode_response = rst_serial.create_anode_response(nslices, nsipms, slices)
probs, sensor_ids, voxel_starts, sensor_starts, nprobs, fwd_num = rst_serial.compute_probabilities_cuda(rst_voxels.voxels, rst_voxels.slice_ids, rst_voxels.nv    oxels, nslices, xs, ys, nsipms, sipms_per_voxel, sipm_dist, sipm_params, anode_response)
sensor_probs, voxel_ids, sensor_starts, sensor_starts_c, sensor_starts_c_ids = rst_serial.compute_sensor_probs(probs, nprobs, nslices, nsipms, voxel_starts, s    ensor_starts, sensor_ids, rst_voxels.slice_ids)
sipm_fwd_denom = rst_serial.forward_denoms(nsipms, nslices, rst_voxels.voxels, sensor_probs, voxel_ids, sensor_starts_c, sensor_starts_c_ids)

