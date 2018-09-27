from invisible_cities.io.table_io import make_table

import tables as tb


class ResetLikelihood(tb.IsDescription):
    event      = tb.UInt32Col (pos=0)
    iterations = tb.UInt32Col (pos=1)
    likelihood = tb.Float64Col(pos=2)

class ResetDST(tb.IsDescription):
    event = tb.UInt32Col (pos=0)
    x     = tb.Float64Col(pos=1)
    y     = tb.Float64Col(pos=2)
    z     = tb.Float64Col(pos=3)
    E     = tb.Float64Col(pos=4)


def reset_voxels_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'RECO',
                            name        = table_name,
                            fformat     = ResetDST,
                            description = 'Reset dst',
                            compression = compression)

    def write_voxels(evt, x, y, z, E):
        row = map_table.row
        row["event"] = evt
        row["x"  ]   = x
        row["y"  ]   = y
        row["z"  ]   = z
        row["E"  ]   = E
        row.append()

    return write_voxels


def reset_lhood_writer(hdf5_file, table_name, *, compression='ZLIB4'):
    map_table  = make_table(hdf5_file,
                            group       = 'RECO',
                            name        = table_name,
                            fformat     = ResetLikelihood,
                            description = 'Reset likelihood',
                            compression = compression)

    def write_likelihood(evt, iterations, likelihood):
        row = map_table.row
        row["event"]      = evt
        row["iterations"] = iterations
        row["likelihood"] = likelihood
        row.append()

    return write_likelihood
