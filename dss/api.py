import requests
import pandas as pd
from io import StringIO

import vertica_python
#import vertica_db_client

def read_dataframe_post(apiKey,dss_host,keyProject,dataset_name):
    """
    Load pandas dataframe from dss db
    Args:
        Connexion params
    """
    print('Request REST DSS api')
    req3 = """http://{apiKey}:@{dss_host}/public/api/projects/{keyProject}/datasets/{dataset_name}/data""".format(
                apiKey=apiKey,
                dss_host=dss_host,
                keyProject=keyProject,
                dataset_name=dataset_name
                )

    headers = {
            'content-type': 'application/json'
            }

    r3 = requests.post(req3, headers=headers, json={
        "format": "tsv-excel-header",
        "filter": "DI_StatutDossier = 4",
        "sampling": {
            "sampling" : 'head',
            "limit" : 10
            },
        "columns": [
            "CG_ModeleVehicule",
            "CG_marque_modele",
            "DI_StatutDossier"
            ],
        })
    #print(r3.text)
    sio = StringIO(r3.content.decode('utf-8'))
    df = pd.read_csv(sio, sep="\t")

    return df

def read_dataframe(apiKey,dss_host,keyProject,dataset_name):
    """
    Load pandas dataframe from dss db
    Args:
        Connexion params
    """
    print('Request REST DSS api')
    req3 = """http://{apiKey}:@{dss_host}/public/api/projects/
    {keyProject}/datasets/{dataset_name}/data?format=tsv-excel-header""".format(
                apiKey=apiKey,
                dss_host=dss_host,
                keyProject=keyProject,
                dataset_name=dataset_name
                )

    r3 = requests.get(req3)
    sio = StringIO(r3.content.decode('utf-8'))
    df = pd.read_csv(sio, sep="\t")

    return df


def write_dataframe(host,keyProject,dataset_name,df):
    """
    Write pandas dataframe in vertica bd
    Args:
        Connexion params
    """
    dataset_name = '%s_%s'%(keyProject,dataset_name)
    req_placeholder = ', '.join(['%s']*len(df.columns))

    cxn = {"user":'dbadmin',
       "password":'',
       "host" :host,
       "port":5433,
       "database":"docker"}

    engine = vertica_python.connect(**cxn)
    s = StringIO()
    df.to_csv(s,header=False,index=False)

    #cur.copy("COPY %s (img_name,path, traceback) FROM STDIN DELIMITER ',' ENCLOSED BY '\"'"%dataset_name, s.getvalue(), buffer_size=65536)
    #df.to_sql(dataset_name , con=engine)
    print('injected')
    #engine = vertica_db_client.connect("host=192.168.4.30 database=docker port=5433 user=dbadmin")
    data = [tuple(x) for x in df.values]
    try:
        cur = engine.cursor()
        # same error as pd.to_sql
        #req = """INSERT INTO {dataset_name} ({col}) VALUES ({val})""".format(
        #                    dataset_name=dataset_name,
        #                    col=req_placeholder % tuple(df.columns),
        #                    val=req_placeholder % tuple(['?' for i in df.columns])
        #                        )
        #print(req)
        #cur.executemany(req, data)
        req = """COPY {dataset_name} ({col}) FROM STDIN DELIMITER ',' ENCLOSED BY '\"'""".format(
                dataset_name=dataset_name,
                col=req_placeholder % tuple(df.columns),
        )

        cur.copy(req,
                s.getvalue(), buffer_size=65536)
        engine.commit()
        engine.close()
        return ''

    except Exception as e:
        print(e)
        engine.rollback()
        engine.close()
        return e



def test_read_dataframe():

    #apiKey = 'g6A3LunBOqufXeNV08rO1WWlj0BUDptz'
    apiKey = 'kANcJMHYaFvIxcMdvtEHpa3HpiZHYh8O'
    dss_host = '192.168.4.30:10000'
    #keyProject = 'REFERENTIELMARQUESMODELES'
    keyProject = 'VIT'
    #dataset_name = 'esiv_by_cnit_clean'
    dataset_name = 'CarteGrise_class'

    df = read_dataframe_post(apiKey,dss_host,keyProject,dataset_name)
    #df = read_dataframe(apiKey,dss_host,keyProject,dataset_name)
    print(df.head())

    ##df.iloc[:10000].to_csv('./bbox_small.csv', index=False)
    #df.to_csv('./label.csv', index=False)
    ##assert df.shape[0]>10000

def test_write_dataframe():
    """
    #from sqlalchemy import create_engine
    keyProject = 'VIT'
    dataset_name = 'VIT_mif'


    #engine = create_engine('vertica+vertica_python://dbadmin:@192.168.4.30:5433/docker')
    #engine = create_engine('vertica+turbodbc://dbadmin:@192.168.4.30:5433/docker')
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

if __name__ == "__main__":
    #test_write_dataframe()

    test_read_dataframe()
