#!/usr/bin/env python

import sys
sys.path.append('/home/pboyd/codes_in_development/net_finder')
from sqlbackend import DataStorage, Net_sql, SQLNet

db = DataStorage("test.job")
for net in db.session.query(SQLNet):
    n = Net_sql(net.mofname)
    n.from_database(net)
    n.show()

