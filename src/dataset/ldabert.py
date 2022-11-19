# from transformers import BertTokenizer, TFBertMainLayer, TFBertForNextSentencePrediction, TFBertPreTrainedModel
# from transformers.modeling_tf_utils import TFPreTrainedModel, get_initializer, keras_serializable, shape_list
import tensorflow as tf
import numpy as np
import pickle
import math
import config
import pathlib
from .dataset_params import DatasetParams

from .nlppreprocessing import (
    preprocess_sent,
    preprocess_word,
)
from .utils import flatten
import sys, os

sys.path.insert(0, config.root_path)
from src.dataset.LDA_BERT.LDA_BERT import LDA_BERT
from db.db import DB, AugmentedDB

from cached_property import cached_property


class LDABERTDataset(DatasetParams):
    def __init__(
        self,
        *,
        dataset_slice="training",
        dataset_type="default",
        pct_data=0.005,
        max_seq_length=256,
        random=False,
        augment_pct=0.0,
        remove_duplicates=False,
        max_segment_length=5,
        lda_topics=10,
        lda_gamma=15
    ):
        self.topics = lda_topics
        self.lda_gamma = lda_gamma
        self.db = DB(dataset_type)
        self.augmented_db = AugmentedDB(dataset_type)
        super().__init__(
            dataset_slice=dataset_slice,
            dataset_type=dataset_type,
            pct_data=pct_data,
            max_seq_length=max_seq_length,
            random=random,
            augment_pct=augment_pct,
            remove_duplicates=remove_duplicates,
            max_segment_length=max_segment_length,
        )

    @cached_property
    def data_segments(self):
        regular_segments = self.db.get_random_segments_pct(
            pct_data=self.pct_data, max_segment_size=self.max_segment_length
        )
        augmented_segments = self.augmented_db.get_random_segments_pct(
            pct_data=self.augment_pct, max_segment_size=self.max_segment_length
        )

        return regular_segments + augmented_segments

    @cached_property
    def data(self):
        return flatten(self.data_segments)

    @cached_property
    def num_samples(self):
        return len(self.data)

    @cached_property
    def sentences(self):
        return [x[1] for x in self.data]

    @cached_property
    def labels(self):
        return [x[2] for x in self.data]

    @cached_property
    def sentence_segments(self):
        return [[y[1] for y in x] for x in self.data_segments]

    @cached_property
    def label_segments(self):
        return [[y[2] for y in x] for x in self.data_segments]

    def _remove_duplicates(self, sentences, labels):
        new_sentences = []
        new_labels = []
        prev_sent = ""
        for t, l in zip(sentences, labels):
            if t == prev_sent:
                continue
            else:
                new_sentences.append(t)
                new_labels.append(l)
            prev_sent = t

        return new_sentences, new_labels

    def _preprocess(self, sentences, labels):
        """
        Preprocess the data
        """

        print("Preprocessing raw texts ...")
        new_sentences = []  # sentence level preprocessed
        token_lists = []  # word level preprocessed
        # idx_in = []  # index of sample selected
        new_labels = []
        #     samp = list(range(100))
        print(len(sentences), len(labels))
        for i, sent in enumerate(sentences):
            sentence = preprocess_sent(sent)
            token_list = preprocess_word(sentence)
            if token_list:
                # idx_in.append(idx)
                new_sentences.append(sentence)
                token_lists.append(token_list)
                new_labels.append(labels[i])
            print(
                "{} %".format(str(np.round((i + 1) / len(sentences) * 100, 2))),
                end="\r",
            )
        print("Preprocessing raw texts. Done!")
        return new_sentences, token_lists, new_labels

    def format_sentences_tri_input(self, sentences):
        left_input = tf.convert_to_tensor([sentences[-1], *sentences[:-1]])

        mid_input = tf.convert_to_tensor(sentences)

        right_input = tf.convert_to_tensor([*sentences[1:], sentences[0]])

        return left_input, mid_input, right_input

    def create_vectors(self, filepath):
        print("root path", config.root_path)
        self.lda_bert = LDA_BERT(self.sentences, self.topics, self.token_lists)
        vectors = self.lda_bert.vectorize(method="LDA_BERT")
        absolute_filepath = os.path.join(
            config.root_path,
            "data",
            "generated_vectors",
            filepath.split("/")[-2],
            filepath.split("/")[-1],
        )
        pickle.dump(
            [vectors, self.labels, self.sentences], open(absolute_filepath, "wb")
        )
        return vectors, self.labels, self.sentences

    def get_vectors(self, filepath):
        absolute_filepath = os.path.join(
            config.root_path,
            "data",
            "generated_vectors",
            filepath.split("/")[-2],
            filepath.split("/")[-1],
        )
        try:
            vectors = pickle.load(open(absolute_filepath, "rb", buffering=0))
            return (
                vectors[0] if len(vectors[0]) else [],
                vectors[1] if len(vectors[1]) else [],
                vectors[2] if len(vectors[2]) else [],
            )
        except Exception as e:
            print("something went wrong", e)
            return [], [], []

    def process(self, preprocess=False):
        sentence_segments, label_segments = self.sentence_segments, self.label_segments
        sentences, labels = self.sentences, self.labels

        # convert to a vertical stacked array.
        labels = np.expand_dims(np.array(labels), axis=1)

        if self.remove_duplicates:
            sentences, labels = self._remove_duplicates(sentences, labels)

        if preprocess:
            sentences, token_lists, labels = self._preprocess(sentences, labels)
            self.sentences, self.token_lists, self.labels = (
                sentences,
                token_lists,
                labels,
            )
            return sentences, token_lists, labels
        else:
            self.sentences, self.token_lists, self.labels = sentences, [], labels
            return sentences, [], labels

