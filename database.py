from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

USERNAME = '3355'
PASSWORD = input("Enter your password: ")
HOST = '127.0.0.1'
PORT = '3306'
DBNAME = 'post_db'

DB_URL = f'mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}'

class engineconn:
    def __init__(self):
        self.engine = create_engine(DB_URL, pool_recycle=500)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        return conn