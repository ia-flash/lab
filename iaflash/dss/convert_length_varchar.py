import pandas as pd
import psycopg2



connection = psycopg2.connect(user = "postgres",
                              password = "postgres",
                              host = "192.168.4.25",
                              port = "5432",
                              database = "postgres")

cur = connection.cursor()

req = """select table_name, column_name, data_type, character_maximum_length from INFORMATION_SCHEMA.COLUMNS where table_name ILIKE 'VIT3%' AND data_type = 'character varying' and character_maximum_length =16200  ;"""

print(req)
df = pd.read_sql(req, connection)
print(df)

def format_str_col(args):
    print(args)
    req = """ALTER TABLE "{table_name}" ALTER COLUMN {column_name} TYPE varchar(256);""".format(**args)
    print(req)

    cur.execute(req)


for i, row in df.iterrows():
    print(row['data_type'])
    format_str_col(row)

print('DONE \n')
req = """select table_name, column_name, data_type, character_maximum_length from INFORMATION_SCHEMA.COLUMNS where table_name ILIKE 'VIT3%' AND data_type = 'character varying' ;"""
print(req)
df = pd.read_sql(req, connection)
print(df)

connection.close()
