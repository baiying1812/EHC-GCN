####关于ntu数据集这块，要看一下论文的【人数，帧数，关节数25，关节坐标xyz】
import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization


class NTU_Reader():
    def __init__(self, args, root_folder, transform, ntu60_path, ntu120_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 25
        self.max_person = 4
        self.select_person_num = 2
        self.dataset = args.dataset
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        # Set paths
        ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(os.path.realpath(__file__)))
        if self.transform:   ###out_path = ./data/ /ntu_xsub
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)

        # Divide train and eval samples
        training_samples = dict()
        training_samples['ntu-xsub'] = [###共40个subjects
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
        ]
        training_samples['ntu-xview'] = [2, 3]##共3个摄像机位置
        training_samples['ntu-xsub120'] = [###共106个受试者subjects
            1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
            38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
            80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
        ]
        training_samples['ntu-xset120'] = set(range(2, 33, 2))###根据受试者距离和高度
        self.training_sample = training_samples[self.dataset]

        # Get ignore samples
        try:
            with open(ntu_ignored, 'r') as f:
                self.ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
        except:
            logging.info('')
            logging.error('Error: Wrong in loading ignored sample file {}'.format(ntu_ignored))
            raise ValueError()

        # Get skeleton file list
        self.file_list = []
        for folder in [ntu60_path, ntu120_path]:
            for filename in os.listdir(folder):
                self.file_list.append((folder, filename))
            if '120' not in self.dataset:  # for NTU 60, only one folder
                break

    def read_file(self, file_path):
        skeleton = np.zeros((self.max_person, self.max_frame, self.max_joint, self.max_channel), dtype=np.float32)
        ####skeleton（最大人数，帧数，关节点数25，通道数）
        with open(file_path, 'r') as fr:
            frame_num = int(fr.readline())###读取文件第一行：帧数
            for frame in range(frame_num):
                person_num = int(fr.readline())###第二行：人数
                for person in range(person_num):
                    person_info = fr.readline().strip().split()###第三行：人的信息：身体标识符，边缘剪裁状态，左手置信度，左手状态...是否被遮挡，身体倾斜度，...
                    joint_num = int(fr.readline())###第四行：关节点数
                    for joint in range(joint_num):
                        joint_info = fr.readline().strip().split()##去除前导和尾随空格，然后使用空格分隔符将其分割成列表[x,y,z，...]
                        skeleton[person,frame,joint,:] = np.array(joint_info[:self.max_channel], dtype=np.float32)
                        ##########skeleton[第几个人，第几帧，第几个关节点，关节点坐标xyz]
        return skeleton[:,:frame_num,:,:], frame_num####帧数，不是最大帧数

    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames删选出非零帧T和，生成索引数组
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
        else:
            s = 0
        return s

    def gendata(self, phase):
        sample_data = []
        sample_label = []
        sample_path = []
        sample_length = []
        iterizer = tqdm(sorted(self.file_list), dynamic_ncols=True) if self.progress_bar else sorted(self.file_list)
        for folder, filename in iterizer:###遍历每一个文件
            if filename in self.ignored_samples:
                continue

            # Get sample information根据文件名分训练集和测试集
            file_path = os.path.join(folder, filename)
            setup_loc = filename.find('S')###S的索引位置，相机设置角度
            camera_loc = filename.find('C')
            subject_loc = filename.find('P')
            action_loc = filename.find('A')
            setup_id = int(filename[(setup_loc+1):(setup_loc+4)])
            camera_id = int(filename[(camera_loc+1):(camera_loc+4)])
            subject_id = int(filename[(subject_loc+1):(subject_loc+4)])
            action_class = int(filename[(action_loc+1):(action_loc+4)])

            # Distinguish train or eval sample
            if self.dataset == 'ntu-xview':
                is_training_sample = (camera_id in self.training_sample)
            elif self.dataset == 'ntu-xsub' or self.dataset == 'ntu-xsub120':
                is_training_sample = (subject_id in self.training_sample)
            elif self.dataset == 'ntu-xset120':
                is_training_sample = (setup_id in self.training_sample)
            else:
                logging.info('')
                logging.error('Error: Do NOT exist this dataset {}'.format(self.dataset))
                raise ValueError()
            if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                continue

            # Read one sample
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num), dtype=np.float32)
            skeleton, frame_num = self.read_file(file_path)##【最多人数4，帧数，关节数25，关节坐标xyz】，最大帧数300

            # Select person by max energy选择人数缩减为2人
            energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(self.max_person)])###计算skeleton[m]中非零元素的标准差
            index = energy.argsort()[::-1][:self.select_person_num]###排序【逆序】选择前2个人
            skeleton = skeleton[index]
            data[:,:frame_num,:,:] = skeleton.transpose(3, 1, 2, 0)###【channel_xyz,帧数，关节点25，选择人数2】

            sample_data.append(data)
            sample_path.append(file_path)
            sample_label.append(action_class - 1)  # to 0-indexed
            sample_length.append(frame_num)

        # Save label  .pkl
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_path, list(sample_label), list(sample_length)), f)###样本路径，标签列表，样本长度（多少帧）

        # Transform data
        sample_data = np.array(sample_data)
        if self.transform:
            sample_data = pre_normalization(sample_data, progress_bar=self.progress_bar)

        # Save data .npy###【channel_xyz,帧数，关节点25，选择人数2】
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_data)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)
