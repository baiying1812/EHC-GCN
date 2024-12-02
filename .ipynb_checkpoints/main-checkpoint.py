# Processing 进一步看代码processing.start.extract,,,visualizer.start
### import 类/包package。包参与了Python的模块导入机制。Python解释器会查找包含该包的目录结构，并自动执行 __init__.py 文件（如果存在）中的代码

###先批量改modify_configs.py yaml文件，再改main.py里的参数默认值，最后将模型模型参数都写入work_dir/.../config.yaml（initilizer.py）
##kwargs.update(更新关键字参数

###批量生成数据集（X-sub，X-view，X-sub120，X-sub120）文件（标签pkl，数据npy）bash scripts/auto_gen_data.sh，保存在root_folder文件夹下
    ##具体来说generator.py→reader文件夹，生成骨架数据pkl：【样本路径，标签列表，样本长度（多少帧）】，npy：【channel_xyz,帧数，关节点25，选择人数2】

##processing.py→Initializer.py→dataset文件夹：feeders：多分支处理后的数据，shape[3分支, 6=3*2分支, 帧数, 25, 2],60个类别，graph.A邻接矩阵，graph.parts人体上肢下肢等各部分关节点
    #####feeders：(data_new_JVB, label, name)

##processing.py→Initializer.py→model文件夹→_init.py导入模型参数→nets.py模型架构：
import os, yaml, argparse
from time import sleep

from src.generator import Generator##generator.py→reader文件夹，生成骨架数据，保存在root_folder文件夹下
from src.processor import Processor##主进程
from src.visualizer import Visualizer##可视化


def main():
    # Loading parameters
    parser = init_parser()##初始化参数
    args = parser.parse_args()
    args = update_parameters(parser, args)  # cmd > yaml > default 命令行>modify_configs.py>默认，
####已读取所有参数args（.configs/.yaml和main里定义的
    # Waiting to run
    sleep(args.delay_hours * 3600)

    # Processing 进一步看代码processing.start.extract,,,visualizer.start
    if args.generate_data:###若需要生成骨架数据，则不进行processing
        g = Generator(args)
        g.start()

    elif args.extract or args.visualize:###如果不需要生成骨骼数据，是否需要可视化
        if args.extract:###提取特征，用于可视化
            p = Processor(args)
            p.extract()
        if args.visualize:
            v = Visualizer(args)
            v.start()

    else:
        p = Processor(args)  ###主程序
        p.start()

##参数设置
def init_parser():
    parser = argparse.ArgumentParser(description='Method for Skeleton-based Action Recognition')

    # Setting
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--pretrained_model_name', '-pm', type=str, default='', help='pretrained model name')
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar')
    parser.add_argument('--delay_hours', '-dh', type=float, default=0, help='Delay to run')

    # Processing
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualize', '-v', default=False, action='store_true', help='Visualization')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')

    # Visualization
    parser.add_argument('--visualization_class', '-vc', type=int, default=0, help='Class: 1 ~ 60, 0 means true class')
    parser.add_argument('--visualization_sample', '-vs', type=int, default=0, help='Sample: 0 ~ batch_size-1')
    parser.add_argument('--visualization_frames', '-vf', type=int, nargs='+', default=[], help='Frame: 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Args for creating model')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')
    
    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    return parser

###读取参数配置文件config/.yaml
def update_parameters(parser, args):
    if os.path.exists('./configs/{}.yaml'.format(args.config)):
        with open('./configs/{}.yaml'.format(args.config), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)##读取配置文件里的参数
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}.yaml!'.format(args.config))
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))####将工作目录更改为当前目录
    main()
