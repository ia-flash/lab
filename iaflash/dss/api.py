import requests
import pandas as pd
from io import StringIO

import vertica_python
import psycopg2
from psycopg2.extras import execute_values
#import vertica_db_client
import dataikuapi as dka

def connect(host, connector ):

    if connector == 'vertica' :
        cxn = dict(user = 'dbadmin',
                   password = '',
                   host = host,
                   port = 5433,
                   database = "docker")
        lib = vertica_python

    elif connector == 'postgres' :
        cxn = dict(user = "postgres",
                  password = "postgres",
                  host = host,
                  port = 5432,
                  database = "postgres")
        lib = psycopg2

    print('Connexion parameters : {}'.format(cxn))
    engine = lib.connect(**cxn)
    return engine

def read_dataframe(apiKey, host, keyProject, dataset_name, columns=[], conditions="", limit=-1, sampling=-1, connector='vertica'):

    dataset_name = '%s_%s'%(keyProject, dataset_name)
    engine = connect(host, connector)

    if conditions != '':
        conditions = 'WHERE %s '%conditions
        if float(sampling) > 0:
            conditions += ' AND random() < %s '% sampling
    else:
        if float(sampling) > 0:
            conditions = ' WHERE random() < %s '% sampling

    if len(columns)>0:
        columns =  ["\"%s\""%col for col in columns]
        req = """SELECT {columns} FROM "{table}" {conditions}""".format(columns=','.join(columns),
        table=dataset_name,
        conditions=conditions)
    else:
        req = """SELECT * FROM "{table}" {conditions}""".format(table=dataset_name, conditions=conditions)


    if int(limit) > 0:
         req += " LIMIT %s " %int(limit)

    print(req)
    df = pd.read_sql(req,engine)
    engine.close()
    return  df


def read_dataframe_dss(apiKey,dss_host,keyProject,dataset_name,columns=[],filter=""):
    """
    Load pandas dataframe from dss db
    Args:
        Connexion params
    """
    print('Request REST DSS api')
    req3 = u"""http://{apiKey}:@{dss_host}/public/api/projects/{keyProject}/datasets/{dataset_name}/data?format=tsv-excel-header?formatParams=""".format(
                apiKey=apiKey,
                dss_host=dss_host,
                keyProject=keyProject,
                dataset_name=dataset_name
                )

    if len(columns)>0:
        req3 += '?columns=%s'%(','.join(columns))
    if filter != "":
        req3 += '?filter=%s'%filter

    #req3 += "?sampling=100"
    print (req3)
    r3 = requests.get(req3)
    sio = StringIO(r3.content.decode('utf-8'))
    df = pd.read_csv(sio, sep="\t")

    return df

def write_shema(host, keyProject, dataset_name, schema):
    """
    Write shema from json
    Similar to set_schema from dataikuapi
    """
    client = dka.dssclient.DSSClient(host, keyProject)
    print(client.list_project_keys())

    project = client.get_project(keyProject)
    dataset = project.get_dataset(dataset_name)
    schema = dataset.get_schema()
    print(schema)
    #schema['columns'].append({'name' : 'new_column', 'type' : 'bigint'})
    #dataset.set_schema(schema)

def write_dataframe(host, keyProject, dataset_name, connector, df):
    """
    Write pandas dataframe in vertica or postgres. Dataframe must have named columns
    Args:
        Connexion params
    """
    dataset_name = '%s_%s'%(keyProject,dataset_name)
    req_placeholder = ', '.join(['%s']*len(df.columns))

    engine = connect(host, connector)


    #engine = vertica_db_client.connect("host=**, database=**, port=** user=**")
    data = [tuple(x) for x in df.values]
    try:
        cur = engine.cursor()

        if   connector == 'postgres':
        # same error as pd.to_sql
            req = """INSERT INTO "{dataset_name}" ({col}) VALUES %s""".format(
                                dataset_name=dataset_name,
                                col=req_placeholder % tuple(df.columns)
                                )
            print(req)
            execute_values(cur, req, data)

        elif connector == 'vertica':
            s = StringIO()
            df.to_csv(s,header=False,index=False)
            #cur.copy("COPY %s (img_name,path, traceback) FROM STDIN DELIMITER ',' ENCLOSED BY '\"'"%dataset_name, s.getvalue(), buffer_size=65536)
            #df.to_sql(dataset_name , con=engine)
            req = """COPY "{dataset_name}" ({col}) FROM STDIN DELIMITER ',' ENCLOSED BY '\"'""".format(
                    dataset_name=dataset_name,
                    col=req_placeholder % tuple(df.columns),
            )
            cur.copy(req,
                    s.getvalue(), buffer_size=65536)

        engine.commit()
        engine.close()
        print('injected')

        return ''


    except Exception as e:
        print(e)
        engine.rollback()
        engine.close()
        return e




def test_read_dataframe():


    df = read_dataframe(apiKey,vertica_host,keyProject,dataset_name)
    print(df.head())

    assert df.shape[0]>10000

def test_write_dataframe():
    """
    #from sqlalchemy import create_engine
    keyProject = 'VIT'
    dataset_name = 'VIT_mif'
    engine = vertica_python.connect(**cxn)
    print(df)
    print(engine)
    assert True


    cur = engine.cursor()
    cur.execute("SELECT DI_ReferenceExterne FROM VIT_mif")
    print( cur.fetchone()[0])

    print(pd.read_sql("REFERENTIELMARQUESMODELES_esiv_by_cnit", con=engine))
    engine.close()
    """
    #df.to_sql('%s_%s'%(keyProject,dataset_name),
    #        index=False, if_exists="append", con=engine)
    dataset_name = 'log_retinanet_x101_64x4d_fpn_1x'
    keyProject = 'VIT'
    df = pd.DataFrame(data=[['toto','babar','celeste'],['toto2','babar2','celeste2']],columns=['img_name','path','traceback'])
    write_dataframe(keyProject,dataset_name,df)
    assert True


def test_write_shema():
    write_shema(host, keyProject, dataset_name, {})


if __name__ == "__main__":
    test_write_shema()
