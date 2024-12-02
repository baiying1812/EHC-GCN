import logging

from .graphs import Graph
from .ntu_feeder import NTU_Feeder, NTU_Location_Feeder


__data_args = {
    'ntu-xsub': {'class': 60, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xview': {'class': 60, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xsub120': {'class': 120, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu-xset120': {'class': 120, 'shape': [3, 6, 300, 25, 2], 'feeder': NTU_Feeder},
    'ntu2d-xsub': {'class': 60, 'shape': [3, 6, 300, 17, 2], 'feeder': NTU_Feeder},
    'ntu2d-xview': {'class': 60, 'shape': [3, 6, 300, 17, 2], 'feeder': NTU_Feeder},
    'our-xsub': {'class': 60, 'shape': [3, 6, 300, 17, 2], 'feeder': NTU_Feeder}
}
########### ntu2d,  ./data  ,               288,       
def create(dataset, root_folder, transform, num_frame, inputs, **kwargs):##输入看yaml文件
    graph = Graph(dataset)###返回邻接矩阵A的计算值
    try:
        data_args = __data_args[dataset] ##dataset：ntu-xsub
        data_args['shape'][0] = len(inputs)##inputs: JVB
        data_args['shape'][2] = num_frame
    except:
        logging.info('')
        logging.error('Error: Do NOT exist this dataset: {}!'.foramt(dataset))
        raise ValueError()
    if transform:
        dataset_path = '{}/transformed/{}'.format(root_folder, dataset)
    else:
        dataset_path = '{}/original/{}'.format(root_folder, dataset)
    kwargs.update({
        'dataset_path': dataset_path,
        'inputs': inputs,##JVB，关节流，帧流，骨架角度三分支
        'num_frame': num_frame,
        'connect_joint': graph.connect_joint,##np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
    })
    feeders = {###多分支处理后的数据
        'train': data_args['feeder']('train', **kwargs),##'train': NTU_Feeder('train', **kwargs)
        'eval' : data_args['feeder']('eval', **kwargs),##NTU_Feeder
    }
    #if 'ntu' in dataset:    
    if 'ntu' in dataset or 'ntu2d' in dataset:
        feeders.update({'location': NTU_Location_Feeder(data_args['shape'])})
    return feeders, data_args['shape'], data_args['class'], graph.A, graph.parts
    ##feeders：多分支处理后的数据，shape[3分支, 6=3*2分支, 300, 25, 2],60个类别，graph.A邻接矩阵，graph.parts人体上肢下肢等各部分关节点
    #####feeders：(data_new_JVB, label, name)
