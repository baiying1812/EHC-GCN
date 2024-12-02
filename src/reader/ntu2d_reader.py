
import os, pickle, logging, numpy as np
from tqdm import tqdm

from .. import utils as U
from .transformer import pre_normalization


class NTU2d_Reader():
    def __init__(self, args, root_folder, transform, ntu2d_path, **kwargs):
        self.max_channel = 3
        self.max_frame = 300
        self.max_joint = 17
        self.select_person_num = 2
        self.dataset = args.dataset
        self.file_path = ntu2d_path
        self.progress_bar = not args.no_progress_bar
        self.transform = transform

        # Set paths
        # ntu_ignored = '{}/ignore.txt'.format(os.path.dirname(os.path.realpath(__file__)))
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_folder, self.dataset)
        else:
            self.out_path = '{}/original/{}'.format(root_folder, self.dataset)
        U.create_folder(self.out_path)


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
        sample_list = []
        sample_length = []
        # pkl_file_path = '../ntu60_2d_17.pkl'
        with open(self.file_path, 'rb') as file:
            pkl_data = pickle.load(file)
            # 从 pkl 数据中获取 xsub_train 和 xsub_val 列表,xview_train
            xsub_train_list = pkl_data["split"]["xsub_train"]
            xview_train_list = pkl_data["split"]["xview_train"]
            
            # 遍历annotations列表
            for anno in pkl_data['annotations']:
                # Distinguish train or eval sample
                if self.dataset == 'ntu2d-xview':
                    is_training_sample = (anno['frame_dir'] in xview_train_list)
                elif self.dataset == 'ntu2d-xsub' or self.dataset == 'ntu2d-xsub120':
                    is_training_sample = (anno['frame_dir'] in xsub_train_list)
                else:
                    logging.info('')
                    logging.error('Error: Do NOT exist this dataset {}'.format(self.dataset))
                    raise ValueError()
                if (phase == 'train' and not is_training_sample) or (phase == 'eval' and is_training_sample):
                    continue

                # Read one sample
                data = np.zeros((self.max_channel, self.max_frame, self.max_joint, self.select_person_num),
                                dtype=np.float32)
                keypoints = anno['keypoint']
                # keypoints = np.zeros((num_persons, num_frames, num_points, 2), dtype=np.float32)
                scores = anno['keypoint_score']
                scores_expanded = scores[:, :, :, np.newaxis]
                skeleton = np.concatenate((keypoints, scores_expanded), axis=-1)
                frame_num = anno['total_frames']

                # Select person by max energy
                num_persons = skeleton.shape[0]
                energy = np.array(
                    [self.get_nonzero_std(skeleton[m]) for m in range(num_persons)]) 
                index = energy.argsort()[::-1][:self.select_person_num] 
                skeleton = skeleton[index]
                data[:, :frame_num, :, :] = skeleton.transpose(3, 1, 2, 0) 

                sample_list.append(anno['frame_dir'])
                sample_label.append(anno['label'])
                sample_length.append(anno['total_frames'])
                sample_data.append(data)

            # Save label  .pkl
            with open('{}/{}_label.pkl'.format(self.out_path, phase), 'wb') as f:
                pickle.dump((sample_list, list(sample_label), list(sample_length)), f) 
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
