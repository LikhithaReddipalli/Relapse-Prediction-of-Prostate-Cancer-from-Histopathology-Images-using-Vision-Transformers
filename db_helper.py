import pandas as pd
import sqlite3
from sqlite3 import Error
#import configs.config as config



def create_connection(path):
    '''
    To establish the connection with database
    '''
    try:
        connection = sqlite3.connect(path)
        print(f"Connection to SQLite DB at '{path}' successful")
    except Error as e:
        print(f"The error '{e}' occurred")
        raise e

    return connection


def get_db_table(table_name, db_path, keep_cols=None, index_col=None):
    
    '''
    Function to get the required data file from the database
    '''

    db_con = create_connection(db_path)
    result = pd.read_sql_query(f"SELECT * FROM '{table_name}'", db_con)

    if keep_cols is not None:
        result = result[keep_cols]

    if index_col is not None:
        result = result.set_index(index_col)

    return result




