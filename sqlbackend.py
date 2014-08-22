from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import numpy as np

Base = declarative_base()

class SQLNet(Base):
    __tablename__ = 'net'
    id = Column(Integer, primary_key=True)
    mofname = Column(Text)
    vertices = relationship('SQLVertex', backref='net') 
    edges = relationship('SQLEdge', backref='net')
    cell = relationship('SQLCell', backref='net')

    def __init__(self, mofname):
        self.mofname = mofname

class SQLVertex(Base):
    __tablename__ = 'vertices'
    id = Column(Integer, primary_key=True)
    fx = Column(Float)
    fy = Column(Float)
    fz = Column(Float)
    name = Column(Text)
    mofname = Column(Text, ForeignKey('net.mofname'))

    def __init__(self, pos, name, mofname):
        self.fx
        self.fy
        self.fz
        self.name
        self.mofname

class SQLCell(Base):
    __tablename__ = 'cell'
    id = Column(Integer, primary_key=True)
    a = Column(Float)
    b = Column(Float)
    c = Column(Float)
    alpha = Column(Float)
    beta = Column(Float)
    gamma = Column(Float)
    mofname = Column(Text, ForeignKey('net.mofname'))
    def __init__(self, params, mofname):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.alpha = params[3]
        self.beta = params[4]
        self.gamma = params[5]
        self.mofname = mofname

class SQLEdge(Base):
    __tablename__ = 'edge'
    id = Column(Integer, primary_key=True)
    efx = Column(Float)
    efy = Column(Float)
    efz = Column(Float)
    ofx = Column(Float)
    ofy = Column(Float)
    ofz = Column(Float)
    mofname = Column(Text, ForeignKey('net.mofname'))

    def __init__(self, edge, origin, mofname):
        self.efx = edge[0]
        self.efy = edge[1]
        self.efz = edge[2]
        self.ofx = origin[0]
        self.ofy = origin[1]
        self.ofz = origin[2]
        self.mofname = mofname


class DataStorage(object):

    def __init__(self, db_name):
        self.engine = create_engine('sqlite:///%s.db'%(db_name))
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def store(self, obj):
        self.session.add(obj)

    def flush(self):
        self.session.flush()
        self.session.commit()

