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

