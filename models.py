from sqlalchemy import Column, TEXT, INT
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PostData(Base):
    __tablename__ = "post_data"

    id = Column(INT, primary_key=True, autoincrement=True)
    region = Column(TEXT)
    gender = Column(TEXT)
    age = Column(INT)
    title = Column(TEXT)
    message = Column(TEXT)