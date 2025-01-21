import logging
import os

import numpy as np
import pandas as pd
from decord import VideoReader
from decord import cpu
from torch.utils.data import Dataset

from cyber.dataset.utils import sync_at_rate


class ArxDataset(Dataset):
    def __init__(self, dataset_path, rate=50, chunk_size=50, chunk_stride=1, step_stride=1):
        """
        Build ARX dataset from parsed ros bags.

        Chunk strides are spacing between chunks,
        |<-chunk_size->|______________________________________
        |<-chunk_stride->|<-chunk_size->|_____________________
        |<-chunk_stride->|<-chunk_stride->|<-chunk_size->|____

        Step strides are spacing between steps, within a chunk,
        !<ss>.<ss>.<ss>!______________________________________
        |<-chunk_stride->!<ss>.<ss>.<ss>!_____________________
        |<-chunk_stride->|<-chunk_stride->!<ss>.<ss>.<ss>!____

        Args:
            dataset_path (str): path to the dataset
            rate (int): rate of sync. in hz (default 50)
            chunk_size (int): size of chunks to use (default 50)
            chunk_stride (int): stride of chunks to use (default 1)
            step_stride (int): stride of steps to use (default 1)


        """
        super().__init__()
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.step_stride = step_stride
        # get episodes
        self.episodes = os.listdir(dataset_path)
        self.topdown_images = []
        self.left_images = []
        self.right_images = []
        self.worm_images = []
        self.left_joints = []
        self.right_joints = []
        self.matched_indices = []
        # build valid indices
        self.chunks = []
        for episode in self.episodes:
            episode_path = os.path.join(dataset_path, episode)
            # load timestamps for videos
            video_path = os.path.join(episode_path, "videos")
            topdown_timestamps = np.array(pd.read_csv(os.path.join(video_path, "topdown.txt"), header=None).iloc[[0], :-1]).flatten()
            left_timestamps = np.array(pd.read_csv(os.path.join(video_path, "left.txt"), header=None).iloc[[0], :-1]).flatten()
            right_timestamps = np.array(pd.read_csv(os.path.join(video_path, "right.txt"), header=None).iloc[[0], :-1]).flatten()
            worm_timestamps = np.array(pd.read_csv(os.path.join(video_path, "worm.txt"), header=None).iloc[[0], :-1]).flatten()
            # decord uses lazy loading, so images shouldn't be clogging up memory
            topdown_images = VideoReader(os.path.join(video_path, "topdown.mp4"), ctx=cpu(0))
            left_images = VideoReader(os.path.join(video_path, "left.mp4"), ctx=cpu(0))
            right_images = VideoReader(os.path.join(video_path, "right.mp4"), ctx=cpu(0))
            worm_images = VideoReader(os.path.join(video_path, "worm.mp4"), ctx=cpu(0))
            # load robot arm joints
            joints_path = os.path.join(episode_path, "joints")
            ljwts = np.array(pd.read_csv(os.path.join(joints_path, "follow_arm_left.csv")).iloc[:, 1:])
            rjwts = np.array(pd.read_csv(os.path.join(joints_path, "follow_arm_right.csv")).iloc[:, 1:])
            # robot arm joints should be small enough to fit in memory
            left_joints = ljwts[:, 1:]
            right_joints = rjwts[:, 1:]
            left_joint_timestamps = ljwts[:, 0].flatten()
            right_joint_timestamps = rjwts[:, 0].flatten()
            # sync timestamps
            try:
                sync_indices, sync_timestamps, latencies, _start_indices = sync_at_rate(
                    [topdown_timestamps, left_timestamps, right_timestamps, worm_timestamps, left_joint_timestamps, right_joint_timestamps], rate=rate
                )
            except ValueError as e:
                logging.error(f"skipping episode {episode} due to error: {e}")
                continue
            # check latency for possible issues
            if np.any(np.abs(latencies) > 1.5 / rate * 1e9):
                logging.warning(f"latencies are too high for episode {episode}, \
                                at times {sync_timestamps[np.where(np.abs(latencies) > 1.5 / rate * 1e9)]}")
            # build chunks
            current_chunk_id = len(self.topdown_images)
            self.topdown_images.append(topdown_images)
            self.left_images.append(left_images)
            self.right_images.append(right_images)
            self.worm_images.append(worm_images)
            self.left_joints.append(left_joints)
            self.right_joints.append(right_joints)
            self.matched_indices.append(sync_indices)
            self.chunks.extend([[current_chunk_id, start_i] for start_i in range(0, len(sync_indices) - chunk_size, chunk_stride)])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        epid = self.chunks[idx][0]
        start_i = self.chunks[idx][1]
        topdown_images = self.topdown_images[epid]
        left_images = self.left_images[epid]
        right_images = self.right_images[epid]
        worm_images = self.worm_images[epid]
        left_joints = self.left_joints[epid]
        right_joints = self.right_joints[epid]

        # get the data
        return {
            "images": {
                "topdown": topdown_images[start_i : start_i + self.chunk_size : self.step_stride],
                "left": left_images[start_i : start_i + self.chunk_size : self.step_stride],
                "right": right_images[start_i : start_i + self.chunk_size : self.step_stride],
                "worm": worm_images[start_i : start_i + self.chunk_size : self.step_stride],
            },
            "joints": {
                "left": left_joints[start_i : start_i + self.chunk_size : self.step_stride],
                "right": right_joints[start_i : start_i + self.chunk_size : self.step_stride],
            },
        }
