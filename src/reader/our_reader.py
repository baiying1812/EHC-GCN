
import os, pickle, logging, numpy as np
import random
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization


class OUR_Reader():
    def __init__(self, args, root_folder, transform, our_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 17
        self.select_person_num = 2
        self.dataset = args.dataset ##our-xsub
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        # Set paths
        #ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(os.path.realpath(__file__)))
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)
        
        # Divide train and eval samples
        training_samples = dict()
        '''
        training_samples['our-xsub'] = [
        ]
        training_samples['our-xview'] = [2, 3]##共3个摄像机位置
        self.training_sample = training_samples[self.dataset]
        '''
        # Get skeleton file list
        self.file_list = []
        for filename in os.listdir(our_path):
            self.file_list.append((our_path, filename))

    def read_file(self, file_path):
        with open(file_path, 'rb') as f:
            annotations = pickle.loads(f.read())
            keypoints = annotations['keypoint']
            scores = annotations['keypoint_score']
            scores_expanded = scores[:, :, :, np.newaxis]
            skeleton = np.concatenate((keypoints, scores_expanded), axis=-1)
        return skeleton, annotations['total_frames']

    def get_nonzero_std(self, s):  # (T,V,C)
        index = s.sum(-1).sum(-1) != 0  # select valid frames
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

        train_sample = []
        test_sample = []
        # 初始化字典，每个数字对应一个列表
        file_lists = {num: [] for num in self.class_samples}
        iterizer = tqdm(sorted(self.file_list), dynamic_ncols=True) if self.progress_bar else sorted(self.file_list)
        for folder, filename in iterizer:###将不同类别的文件分开
            # 遍历要检查的数字列表
            action_loc = filename.find('A')
            action_class = int(filename[(action_loc + 1):(action_loc + 4)])
            for num in self.class_samples:
                # 检查文件名中是否包含当前数字
                if action_class == int(num):
                    # 如果包含，将该文件添加到对应的列表中
                    file_lists[num].append((filename))
                    break
        for num, file_list in file_lists.items():
            # 使用random.shuffle对文件列表进行随机打乱
            random.shuffle(file_list)
            # 计算分割点，即文件列表长度乘以训练集比例
            split_index = int(len(file_list) * 0.666)
            # 根据分割点将文件列表分为训练集和测试集
            train_sample.extend(file_list[:split_index])  
            # test_sample.extend(file_list[split_index:])  
        
        print(len(train_sample))
        for folder, filename in iterizer:
            file_path = os.path.join(folder, filename)
            action_loc = filename.find('A')
            action_class = int(filename[(action_loc+1):(action_loc+4)])
            
            is_training_sample = (filename in train_sample)
            if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                continue

            # Read one sample
            data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num),
                            dtype=np.float32)
            skeleton, frame_num = self.read_file(file_path) 

            # Select person by max energy
            num_persons = skeleton.shape[0]
            energy = np.array([self.get_nonzero_std(skeleton[m]) for m in range(num_persons)])
            index = energy.argsort()[::-1][:self.select_person_num]
            skeleton = skeleton[index]
            data[:,:frame_num,:,:] = skeleton.transpose(3, 1, 2, 0)

            sample_path.append(file_path)
            sample_label.append(action_class - 1)  # to 0-indexed
            sample_length.append(frame_num)
            sample_data.append(data)

        # Save label  .pkl
        with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
            pickle.dump((sample_path, list(sample_label), list(sample_length)), f)

        # Transform data
        sample_data = np.array(sample_data)
        if self.transform:
            sample_data = pre_normalization(sample_data, progress_bar=self.progress_bar)

        # Save data .npy
        np.save('{}/{}_data.npy'.format(self.out_path, phase), sample_data)

    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)
