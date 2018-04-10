from invisible_cities.evm.nh5           import VoxelsTable
from .  table_io           import make_table

# writers
def voxels_writer(hdf5_file, *, compression='ZLIB4'):
    voxels_table  = make_table(hdf5_file,
                             group       = 'RECO',
                             name        = 'Events',
                             fformat     = VoxelsTable,
                             description = 'Voxels',
                             compression = compression)
    # Mark column to index after populating table
    voxels_table.set_attr('columns_to_index', ['event'])

    def write_voxels(voxels):
        for v in voxels:
            row = voxels_table.row
            row["event"] = v['event']
            row["X"    ] = v['x']
            row["Y"    ] = v['y']
            row["Z"    ] = v['z']
            row["E"    ] = v['E']
            row.append()

    return write_voxels

def load_voxels(DST_file_name):
    """Return the Voxels as PD DataFrames."""

    dst = tables.open_file(DST_file_name,'r')
    dst_size = len(dst.root.TrueVoxels.Voxels)
    all_events = {}

    event = dst.root.TrueVoxels.Voxels[:]['event']
    X     = dst.root.TrueVoxels.Voxels[:]['X']
    Y     = dst.root.TrueVoxels.Voxels[:]['Y']
    Z     = dst.root.TrueVoxels.Voxels[:]['Z']
    E     = dst.root.TrueVoxels.Voxels[:]['E']
    size  = dst.root.TrueVoxels.Voxels[:]['size']

    for i in range(dst_size):
        current_event = all_events.setdefault(event[i],
                                              VoxelCollection([]))
        voxel = Voxel(X[i], Y[i], Z[i], E[i], size[i])
        current_event.voxels.append(voxel)
    return all_events
