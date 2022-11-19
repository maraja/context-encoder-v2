from db import DB

import sys

sys.path.append("../")
from src.dataset.utils import flatten

from cached_property import cached_property


class Dataset:
    def __init__(
        self,
        dataset_type,
        pct_data,
        max_seg_size,
        max_seq_length,
        pipeline=["augment", "shuffle"],
    ):
        self.db = DB(dataset_type)
        self.pct_data = pct_data
        self.max_seg_size = max_seg_size
        self.max_seq_length = max_seq_length
        self.pipeline = pipeline

    @cached_property
    def data(self):
        return self.db.get_random_segments_pct(
            pct_data=self.pct_data, max_segment_size=self.max_seg_size
        )

    @cached_property
    def sentences_and_labels(self):
        return (
            flatten([[x[1] for x in y] for y in self.data]),
            flatten([[x[2] for x in y] for y in self.data]),
        )

