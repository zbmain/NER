#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zb

from neo4j import GraphDatabase
from config import NEO4J_CONFIG
import file_manager
import os
import fileinput

driver = GraphDatabase.driver(**NEO4J_CONFIG)


def _load_data(path):
    fault_csv_list = os.listdir(path)
    fault_list = list(map(lambda x: x.split(".")[0], fault_csv_list))

    descriptor_list = []
    for disease_csv in fault_csv_list:
        descriptor = list(map(lambda x: x.strip(), fileinput.FileInput(os.path.join(path, disease_csv))))
        descriptor = list(filter(lambda x: 0 < len(x) < 100, descriptor))
        descriptor_list.append(descriptor)
    return dict(zip(fault_list, descriptor_list))


def write(path):
    fault_descriptor_dict= _load_data(path)
    print(fault_descriptor_dict)

    with driver.session() as session:
        for key, value in fault_descriptor_dict.items():
            cypher = 'MERGE (a:Fault{name:%r}) RETURN a' % key  # 故障
            session.run(cypher)
            print(key, ':', value)
            for v in value:
                cypher = 'MERGE (b:Descriptor{name:%r}) RETURN b' % v  # 描述
                session.run(cypher)
                cypher = 'MATCH (a:Fault{name:%r}) MATCH (b:Descriptor{name:%r})' \
                         'WITH a,b MERGE(a)-[r:dis_to_sym]-(b)' % (key, v)
                session.run(cypher)
        cypher = 'CREATE INDEX ON:Fault(name)'
        session.run(cypher)
        cypher = 'CREATE INDEX ON:Descriptor(name)'
        session.run(cypher)


# MATCH (a)-[r]-(b) WHERE b.name='抛锚' return a
if __name__ == '__main__':
    path = "../structured/reviewed/"
    file_manager.delDirEmptyFile(path)  # python使用删空文件
    write(path)
    print('write_end')
