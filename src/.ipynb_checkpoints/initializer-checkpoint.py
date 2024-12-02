import os, yaml, thop, warnings, logging, pynvml, torch, numpy as np
from copy import deepcopy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch import nn

from . import utils as U
from . import dataset
from . import model
from . import scheduler


class Initializer():
    def __init__(self, args):
        self.args = args
        self.init_save_dir()

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    ###将模型模型参数都写入work_dir/.../config.yaml
    def init_save_dir(self):
        self.save_dir = U.set_logging(self.args)
        with open('{}/config.yaml'.format(self.save_dir), 'w') as f:
            yaml.dump(vars(self.args), f)### 将self.args对象的属性以YAML格式序列化并写入文件 f
        logging.info('Saving folder path: {}'.format(self.save_dir))
####通过设置随机种子和启用CUDA的自动优化和cuDNN，为后续的PyTorch模型训练或推理提供了良好的初始环境。
    def init_environment(self):
        np.random.seed(self.args.seed)##设置随机种子可以确保每次运行程序时产生相同的随机数序列，这对于重复实验和验证结果的一致性非常重要
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            pynvml.nvmlInit()
            for i in self.args.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memused = meminfo.used / 1024 / 1024
                logging.info('GPU-{} used: {}MB'.format(i, memused))
                if memused > 1000:
                    pynvml.nvmlShutdown()
                    logging.info('')
                    logging.error('GPU-{} is occupied!'.format(i))
                    raise ValueError()
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')
####self.train_loader = (data_new_JVB, label, name)加载后
    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]###ntu2d-xsub，ntu2d
        dataset_args = self.args.dataset_args[dataset_name]##yaml文件
        dataset_args['debug'] = self.args.debug
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.dataset, **dataset_args
        )##feeders：多分支处理后的数据(data_new_JVB, label, name)，JVB_shape：[3分支, 3*2, 帧数, 25个关节点, 2个人],60个类别，graph.A邻接矩阵，graph.parts人体上肢下肢等各部分关节点
        self.train_loader = DataLoader(self.feeders['train'],###(batchsize, data_new_JVB, label, name)
            batch_size=self.train_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=True, drop_last=True
        )
        self.eval_loader = DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=False, drop_last=False
        )
        self.location_loader = self.feeders['location'] if dataset_name == 'ntu' else None##在2D帧中的位置（用于匹配红外帧）
        # self.location_loader = self.feeders['location'] if dataset_name in ('ntu', 'ntu2d') else None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,##[3_JVB, 6, 帧数, 25个关节点, 2个人]
            'num_class': self.num_class,##60个类别
            'A': torch.Tensor(self.A),##邻接矩阵
            'parts': self.parts,##人体上肢下肢等各部分关节点
        }
        ##参数：yaml文件，main.py，**kwargs
        self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs)##model_type: EfficientGCN-B0
        ##model return： out, feature
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        with open('{}/model.txt'.format(self.save_dir), 'w') as f:
            print(self.model, file=f)
        flops, params = thop.profile(deepcopy(self.model), inputs=torch.rand([1,1]+self.data_shape), verbose=False)
        logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))
        self.model = torch.nn.DataParallel(
            self.model.to(self.device), device_ids=self.args.gpus, output_device=self.output_device
        )
        pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)
        #pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.pretrained_model_name)
        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
            pretrained_state_dict = checkpoint['model']
            # 假设我们只关注不匹配的层
            layer_names = ['main_stream.block-0_scn.conv.gcn.weight', 'main_stream.block-0_scn.residual.0.weight']
            for layer_name in layer_names:
                pretrained_weight = pretrained_state_dict[layer_name]
                # 获取当前模型中该层的权重尺寸
                current_weight_size = self.model.module.state_dict().get(layer_name).size()
                # 裁剪预训练权重以匹配当前模型的尺寸
                # 假设不匹配的是最后一个维度
                pretrained_weight = pretrained_weight[:,:current_weight_size[1],:,:]
                # 更新预训练权重字典
                pretrained_state_dict[layer_name] = pretrained_weight
                
            # self.model.module.load_state_dict(checkpoint['model'])
            self.model.module.load_state_dict(pretrained_state_dict, strict=False)
            self.cm = checkpoint['best_state']['cm']
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}!'.format(pretrained_model))
            logging.info('Create model randomly.')

    def init_optimizer(self):
        try:
            optimizer = U.import_class('torch.optim.{}'.format(self.args.optimizer))
        except:
            logging.warning('Warning: Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = U.import_class('torch.optim.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))

    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        #self.loss_func = AMSoftmaxLoss().to(self.device)
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))

# 定义 AMSoftmax 损失函数
class AMSoftmaxLoss(nn.Module):
    def __init__(self,
                 in_feats=60,
                 n_classes=60,
                 m=0.35,
                 s=30):
        super(AMSoftmaxLoss, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        #print(x.size())
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        if lb_view.is_cuda: lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss, costh_m_s