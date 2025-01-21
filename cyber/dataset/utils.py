import logging

import numpy as np
import math


# Create a package-level logger
logger = logging.getLogger(__name__)


def match_timestamps(times_a, times_b):
    """
    Match two lists of timestamps by closeness in time.
    inputs:
        times_a: list of ascending timestamps
        times_b: list of ascending timestamps
    outputs:
        matches_a: list of indices into times_b for each element of times_a
        matches_b: list of indices into times_a for each element of times_b
        diffs_a: list of differences between each element of times_a and its match in times_b
        diffs_b: list of differences between each element of times_b and its match in times_a
    """
    i, j = 0, 0
    matches_a = [-1] * len(times_a)
    matches_b = [-1] * len(times_b)
    diffs_a = [float("inf")] * len(times_a)
    diffs_b = [float("inf")] * len(times_b)
    while i < len(times_a) and j < len(times_b):
        curdiff = abs(times_a[i] - times_b[j])
        if curdiff < diffs_a[i]:
            diffs_a[i] = curdiff
            matches_a[i] = j
        if curdiff < diffs_b[j]:
            diffs_b[j] = curdiff
            matches_b[j] = i
        if times_a[i] < times_b[j]:
            i += 1
        else:
            j += 1
    # fill in the rest
    if i < len(times_a):
        matches_a[i:] = [j - 1] * (len(times_b) - i)
        diffs_a[i:] = [abs(times_a[i] - times_b[-1])] * (len(times_a) - i)
    if j < len(times_b):
        matches_b[j:] = [i - 1] * (len(times_a) - j)
        diffs_b[j:] = [abs(times_a[-1] - times_b[j])] * (len(times_b) - j)
    return matches_a, matches_b, diffs_a, diffs_b


def sync_at_rate(data_timestamps, rate=50, custom_sync_timestamps=None):
    """
    Given lists of timestamps of different data modalities, start at the soonest timestamp that all data have arrived,
    sample at a certain rate to create a synced-up list of indices in all modalities

    Args:
    data_timestamps (list): timestamps of different data modalities, in seconds
    rate (int): rate of sync. in hz (default 50)
    custom_sync_timestamps: use custeom sync timestamps instead of sync timestamps sampled at uniform rate (default None)

    Returns:
    sync_indices (np.ndarray): indices of the synced-up timestamps in each modality
    sync_timestamps (np.ndarray): timestamps at which the data is synced up
    latencies (np.ndarray): latencies of each modality at each synced-up timestamp
    start_indices (np.ndarray): starting indices of each modality
    """
    if custom_sync_timestamps is None:
        start_timestamp = np.max([data_timestamps[i][0] for i in range(len(data_timestamps))])
    else:
        start_timestamp = custom_sync_timestamps[0]
    not_overlapping = start_timestamp > np.array([data_timestamps[i][-1] for i in range(len(data_timestamps))])
    if np.any(not_overlapping):
        raise ValueError(f"data modalities are not from the same time period: {np.where(not_overlapping)}")
    if custom_sync_timestamps is None:
        end_timestamp = np.min([data_timestamps[i][-1] for i in range(len(data_timestamps))])
        sync_length = math.floor(end_timestamp - start_timestamp) * rate
        sync_timestamps = np.arange(start_timestamp, end_timestamp, 1 / rate)
    else:
        sync_timestamps = custom_sync_timestamps
        sync_length = len(sync_timestamps)
    current_indices = np.zeros(data_timestamps.shape[0], dtype=int)
    for mod_id in data_timestamps.shape[0]:
        if current_indices[mod_id] == len(data_timestamps[mod_id]):
            continue
        while data_timestamps[mod_id][current_indices[mod_id] + 1] < start_timestamp:
            current_indices[mod_id] += 1
    sync_indices = np.zeros((data_timestamps.shape[0], sync_length), dtype=int)
    latencies = np.zeros((data_timestamps.shape[0], sync_length))
    start_indices = np.array(current_indices)  # save the starting indices
    # perform sync
    for i in range(sync_length):
        for mod_id in data_timestamps.shape[0]:
            if current_indices[mod_id] == len(data_timestamps[mod_id]) - 1:
                sync_indices[mod_id, i] = current_indices[mod_id] - 1
                latencies[mod_id, i] = sync_timestamps[i] - data_timestamps[mod_id][current_indices[mod_id]]
                continue
            while data_timestamps[mod_id][current_indices[mod_id] + 1] < sync_timestamps[i]:
                current_indices[mod_id] += 1
            sync_indices[mod_id, i] = current_indices[mod_id]
            latencies[mod_id, i] = sync_timestamps[i] - data_timestamps[mod_id][current_indices[mod_id]]

    return sync_indices, sync_timestamps, latencies, start_indices


class ConcatMemmap(object):
    def __init__(self, dtype: np.dtype = np.float32):
        """
        create an empty ConcatMemmap object

        Args:
            dtype(np.dtype): the datatype of the memmap

        Returns:
            ConcatMemmap: the empty ConcatMemmap object
        """
        self.dtype = dtype
        self.data: list[np.memmap] = []
        self.indmap: list[int] = []
        self.accumulatedLength: list[int] = [0]

    def append(self, data: np.memmap):
        """
        append a memmap to the ConcatMemmap

        Args:
            data(np.ndarray): the data to append
        """
        assert data.dtype == self.dtype, f"Expected data to have dtype {self.dtype}, got {data.dtype}"
        memmap_ind = len(self.data)
        self.data.append(data)
        self.indmap.extend([memmap_ind] * len(data))
        self.accumulatedLength.append(len(data) + self.accumulatedLength[memmap_ind])

    def __getitem__(self, index):
        """
        get item from the ConcatMemmap.
        This is a nuanced operation because a slice can technically span multiple memmaps.
        Currently, we don't allow that here to preserve the time complexity signature of slicing

        Args:
            index: the index to get, can be int or slice

        Returns:
            np.ndarray: the data at the given index
        """
        if isinstance(index, slice):
            if index.stop > len(self):
                raise IndexError("Index out of bounds")
            super_index = self.indmap[index.start]
            sub_index = slice(index.start - self.accumulatedLength[super_index], index.stop - self.accumulatedLength[super_index], index.step)
            if sub_index.stop > len(self.data[super_index]):
                raise IndexError(f"Slice {index} out of bounds, slicing across memmaps currently not supported")
        else:
            super_index = self.indmap[index]
            sub_index = index - self.accumulatedLength[super_index]
        return self.data[super_index][sub_index]

    def __len__(self):
        """
        get the length of the ConcatMemmap

        Returns:
            int: the length of the ConcatMemmap
        """
        return len(self.indmap)
