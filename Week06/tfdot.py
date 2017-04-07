import tensorflow as tf
from graphviz import Digraph
from random import randint
from collections import defaultdict
color_table = {
    "Const": "yellow",
    "MatMul": "#bbffbb",
    "Variable": "#ffbbbb",
    "Assign": "#bbbbff"
    }


def split_name(n):
    ns = n.split('/')
    return "/".join(ns[:-1]), ns[-1]

def common_name_space(n1, n2):
    ns1 = n1.split('/')[:-1]
    ns2 = n2.split('/')[:-1]
    l = min(len(ns1), len(ns2))
    rtn = []
    for i in range(l):
        if ns1[i] != ns2[i]:
            break
        rtn.append(ns1[i])
    return "/".join(rtn)

import html
def tfdot(graph=None, size=(10,30)):
    def get_dot_data(name_space):
        if name_space !='':
            parent, _ = split_name(name_space)
            if name_space not in dot_data_dict[parent]['subgraphs']:
                get_dot_data(parent)['subgraphs'].add(name_space)
        return dot_data_dict[name_space]

    def update_dot(name_space=''):
        name = "cluster_"+name_space if name_space else 'root'
        dot = Digraph(comment="subgraph: "+name_space, name=name, 
                graph_attr={"ratio":"compress",
                "size":"{},{}".format(*size)}
                )
        dot.body.append('label="%s"'%name_space)
        dot_data = dot_data_dict[name_space]
        for s in dot_data['subgraphs']:
            #print(name_space, s)
            dot.subgraph(update_dot(s))
        for node in dot_data['nodes']:
            #print(name_space, "node", node)
            dot.node(**node)
        for edge in dot_data['edges']:
            attr = extra_attr.get(edge, {})
            dot.edge(*edge, **attr)
        return dot


    dot_data_dict = defaultdict(lambda :{"subgraphs":set(), "edges":set(), "nodes": []})
    extra_attr = {}
    if graph is None:
        graph = tf.get_default_graph()
    for op in graph.get_operations():
        if op.type not in color_table:
            new_color = "#%02x%02x%02x"%tuple(randint(0,100)+155 for i in range(3))
            color_table[op.type] = new_color
        color = color_table.get(op.type, "white")
        name_space, name = split_name(op.name)
        outputs_label = "".join("<TR><TD>output:</TD><TD>{} {}</TD></TR>".format(
            html.escape(str(o.shape)), html.escape(o.dtype.name)) for o in op.outputs)
        name_label = "<TD>{}</TD></TR>".format(name)
        op_label = "<TR><TD>{}:</TD>".format(op.node_def.op)
        label = '''< 
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">
        {}{}{}</TABLE>  >'''.format( op_label, name_label, outputs_label)
        dot_data = get_dot_data(name_space)
        dot_data['nodes'].append(dict(name=op.name,  
                        label=label, style="filled", fillcolor=color))
    
    for op in graph.get_operations():
        for i, ip in enumerate(op.inputs):
            name_space = common_name_space(ip.op.name, op.name)
            dot_data = get_dot_data(name_space)
            if op.type == 'Assign' and i ==0:
                dot_data['edges'].add((op.name, ip.op.name))
                extra_attr[(op.name, ip.op.name)]={'color': 'red'}
            else:
                dot_data['edges'].add((ip.op.name, op.name))
    return update_dot()