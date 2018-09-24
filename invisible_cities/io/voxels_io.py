from invisible_cities.evm.nh5 import VoxelsTable
from invisible_cities.evm.nh5 import EventInfo
from invisible_cities.evm.nh5 import ResetDstTable
from .  table_io              import make_table

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

    events_table  = make_table(hdf5_file,
                             group       = 'Run',
                             name        = 'Events',
                             fformat     = EventInfo,
                             description = 'Event numbers',
                             compression = compression)

    def write_voxels(voxels):
        print("write_voxels: ", len(voxels))
        for v in voxels:
            row = voxels_table.row
            row["event"] = v['event']
            row["X"    ] = v['x']
            row["Y"    ] = v['y']
            row["Z"    ] = v['z']
            row["E"    ] = v['E']
            row.append()

        #If there are voxels, write event number
        if voxels.shape[0]:
            row = events_table.row
            row["evt_number"] = voxels[0]['event']
            row["timestamp"]  = 0
            row.append()

    return write_voxels


def dst_writer(hdf5_file, *, compression='ZLIB4'):
    dst_table  = make_table(hdf5_file,
                            group       = 'ResetDst',
                            name        = 'Events',
                            fformat     = ResetDstTable,
                            description = 'RESET DST',
                            compression = compression)

    def write_dst(df):
        for df_row in df.iterrows():
            row = dst_table.row
            row["evt"  ] = df_row[0]
            row["Xmin" ] = df_row[1][0]
            row["Ymin" ] = df_row[1][1]
            row["Zmin" ] = df_row[1][2]
            row["Xmax" ] = df_row[1][3]
            row["Ymax" ] = df_row[1][4]
            row["Zmax" ] = df_row[1][5]
            row["E"    ] = df_row[1][6]
            row["Ecorr"] = df_row[1][7]
            row.append()

    return write_dst
