import logging, torch, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U###加载模型/保存模型.pth
from .initializer import Initializer


class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        ##(三个输入分支JVB, label, name)，其中JVB[batchsize, 3个分支, 3*2, 帧数, 25个关节点, 2个人]
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Using GPU
            x = x.float().to(self.device)##JVB[batchsize, 3分支, 3*2, 帧数, 25个关节点, 2个人]
            y = y.long().to(self.device)##label[batchsize]

            # Calculating Output
            out, _ = self.model(x)##【batchsize，60个类别*帧数*25个关节点*2个人】

            # Updating Weights
            loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]##
            num_top1 += reco_top1.eq(y).sum().item()

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Showing Train Results
        train_acc = num_top1 / num_sample
        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, time()-start_train_time
        ))
        logging.info('')

    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)##是否显示评估进度条
            ##(三个输入分支JVB, label, name)，其中JVB[batchsize, 3分支, 3*2, 帧数, 25个关节点, 2个人]
            for num, (x, y, _) in enumerate(eval_iter):###  num是当前批量的索引，（x，y）是当前批量中的数据和标签

                # Using GPU
                ## x代表三个输入分支JVB[batchsize,3分支, 3*2, 帧数, 25个关节点, 2个人]
                ## 标签y转换为长整型（long）
                x = x.float().to(self.device)
                y = y.long().to(self.device)##一个batchsize的大小

                # Calculating Output
                out, _ = self.model(x)##out：fc层输出的结果【batchsize，60个数字（60类）】

                # Getting Loss
                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())##将当前批量的损失张量转换为单一的数值，并添加到 eval_loss 列表中

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]###reco_top1: 存储每个样本的Top-1预测类别对应的索引
                num_top1 += reco_top1.eq(y).sum().item()###num_top1 用于累计Top-1准确率的样本数量
                reco_top5 = torch.topk(out,5)[1]
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1####模型预测样本的类别cm[y[i], reco_top1[i]] 的值加1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        return acc_top1, acc_top5, cm

    def start(self):
        start_time = time()
        if self.args.evaluate:##是否验证模型，若否则训练模型
            if self.args.debug:##是否debug？？
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)##file_name = 'work_dir/checkpoint.pth.tar
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])###将checkpoint字典中保存的模型参数.pth加载到当前model属性中
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()      ####验证模型
            logging.info('Finish evaluating!')

        else:##训练模型
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}###设置初始值为0，cm：混淆矩阵
            if self.args.resume:####是否需要从之前的训练状态恢复
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)###已经处理过的批量（batch）的总数，train_loader 是一个迭代器，它提供了训练数据集中的批量
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training训练模型
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:###评估间隔周期，%取余
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        if checkpoint:
            #self.cm = checkpoint['best_state']['cm']
            self.model.module.load_state_dict(checkpoint['model'])
            _, _, self.cm = self.eval()
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        #x, y, names = iter(self.eval_loader).next()####从测试数据加载器中获取下一个批量的数据和标签
        x, y, names = next(iter(self.eval_loader))
        location = self.location_loader.load(names) if self.location_loader else []####如果启用了位置加载器，加载位置信息；否则返回一个空列表。

        # Calculating Output
        self.model.eval()
        out, feature = self.model(x.float().to(self.device))

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.classifier.fc.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                data=data, label=label, name=names, out=out, cm=self.cm,
                feature=feature, weight=weight, location=location
            )
            # data，label，names,out,feature,location:跟测试的批量样本有关系；weight：模型中最后fc层的权重；cm是最终整个测试集的混淆矩阵
        logging.info('Finish extracting!')
        logging.info('')
