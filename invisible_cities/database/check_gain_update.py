import sys
import sqlite3
import pymysql
pymysql.install_as_MySQLdb()
import numpy as np


def connect_sqlite(dbname):
    dbfile        = f'invisible_cities/database/localdb.{dbname}.sqlite3'
    conn_sqlite   = sqlite3.connect(dbfile)
    cursor_sqlite = conn_sqlite.cursor()
    return cursor_sqlite


def connect_mysql(dbname):
    conn_mysql  = pymysql.connect(host="neutrinos1.ific.uv.es",
                                  user='nextreader',passwd='readonly', db=dbname)
    cursor_mysql  = conn_mysql .cursor()
    return cursor_mysql


def check_minrun_maxrun(table, filters):
    sql = f'select distinct(Minrun), maxrun from {table} where {filters} ORDER BY MinRun,MaxRun;'

    cursor_sqlite.execute(sql)
    cursor_mysql .execute(sql)

    data_mysql  = cursor_mysql .fetchall()
    data_sqlite = cursor_sqlite.fetchall()

    # Convert to np float64 arrays to replace None's vy np.nan's
    array_mysql  = np.array(data_mysql , dtype=np.float64)
    array_sqlite = np.array(data_sqlite, dtype=np.float64)

    #return data_mysql, data_sqlite

    np.testing.assert_allclose(array_mysql, array_sqlite)

    try:
        latest_run = data_sqlite[-1][0]
    except IndexError:
        latest_run = None

    return latest_run


if __name__ == '__main__':
    # Get DB name from the list of updated files in the commit
    dbfile = next(filter(lambda f: f.endswith('sqlite3'), sys.argv))
    dbname = dbfile.split('.')[-2]

    # Connect to MySQL server and open SQLite copy
    cursor_sqlite = connect_sqlite(dbname)
    cursor_mysql  = connect_mysql (dbname)

    # Check tables
    tables = ['ChannelGain', 'ChannelMask', 'SipmNoisePDF']
    pmts  = 'SensorID < 100'
    sipms = 'SensorID > 100'

    # Print latest run available in each case
    for table in tables:
        latest_run_sipms = check_minrun_maxrun(table, sipms)
        latest_run_pmts  = check_minrun_maxrun(table, pmts)

        print(f'{table}: latest SiPM run {latest_run_sipms}')
        print(f'{table}: latest PMT  run {latest_run_pmts}')
