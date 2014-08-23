from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, Float, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
from numpy import cos, sin, array
from plotter import GraphPlot
DEG2RAD = np.pi / 180.

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
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    name = Column(Text)
    mofname = Column(Text, ForeignKey('net.mofname'))

    def __init__(self, pos, name, mofname):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.name = name
        self.mofname = mofname

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

class Net_sql(object):

    def __init__(self, name):
        self.net_name = name
        self.nodes = []
        self.edge_vectors = [] 
        self.cell = None

    def from_database(self, SQLNet):
        for v in SQLNet.vertices:
            name = v.name
            pos = array([v.x, v.y, v.z])
            self.nodes.append((name, pos))

        for e in SQLNet.edges:
            origin = array([e.ofx, e.ofy, e.ofz])
            vector = array([e.efx, e.efy, e.efz])
            self.edge_vectors.append((origin, vector))

        c = SQLNet.cell[0]
        self.params = [c.a, c.b, c.c, c.alpha, c.beta, c.gamma]
        self.__mkcell()

    def __mkcell(self):
        """Update the cell representation to match the parameters."""
        a_mag, b_mag, c_mag = self.params[:3]
        alpha, beta, gamma = [x * DEG2RAD for x in self.params[3:]]
        a_vec = array([a_mag, 0.0, 0.0])
        b_vec = array([b_mag * cos(gamma), b_mag * sin(gamma), 0.0])
        c_x = c_mag * cos(beta)
        c_y = c_mag * (cos(alpha) - cos(gamma) * cos(beta)) / sin(gamma)
        c_vec = array([c_x, c_y, (c_mag**2 - c_x**2 - c_y**2)**0.5])
        self.cell = array([a_vec, b_vec, c_vec])
        self.icell = np.linalg.inv(self.cell.T)

    def show(self):
        gp = GraphPlot()
        gp.plot_cell(cell=self.cell, colour='g')
        for id, (name, node) in enumerate(self.nodes):
            # metal == blue
            if name.startswith('m'):
                colour = 'b'
            # organic == green
            elif name.startswith('o'):
                colour = 'g'
            # functional group == red
            else:
                colour = 'r'
            gp.add_point(point=node, label=name, colour=colour)
     
        for ind, (point,edge) in enumerate(self.edge_vectors):
            # convert to fractional
            plot_point = np.dot(self.icell, point)
            plot_edge = np.dot(self.icell, edge)
            gp.add_edge(plot_edge, origin=plot_point)

        gp.plot()
