# create a distance matrix between threads
from datetime import datetime

import os
from nltk import TreebankWordTokenizer
import unicodedata
import multiprocessing
import pickle
from multiprocessing import Manager
import math

from RedditThread import RedditThread
from SemanticSimilarityHelper import SemanticSimilarityHelper
from vocabulary import Vocabulary
from embeddings import WordEmbeddings
from AnnotatedRedditComment import AnnotatedRedditComment


def chunks(l: list, n: int):
    """
    Split a list into evenly sized chunks
    :param l: list
    :param n: chunks
    :return: list of lists
    """
    return [l[i:i + n] for i in range(0, len(l), n)]


def compute_parallel(input_file_name: str, output_file_prefix: str):
    # run parallel as jobs; command line parameter determines which part of the batch is processed
    # 605 x 6 jobs; each 32 cores
    if int(os.sys.argv[1]) not in range(0, 4):
        raise Exception("You must specify a number between 0-5 as a command line parameter")

    # which split of the data are we using
    current_split_number = int(os.sys.argv[1])

    manager = Manager()

    with open(input_file_name, "rb") as f:
        positive_instances_all, negative_instances = pickle.load(f)

    # sort again to make sure the chunks are all the same
    # positive_instances_all = sorted(positive_instances_all)
    # negative_instances = sorted(negative_instances)
    print(len(positive_instances_all))

    positive_instances_by_ids = dict()
    for _ in positive_instances_all:
        assert isinstance(_, RedditThread)
        positive_instances_by_ids[_.get_last_comment_name()] = _

    assert len(positive_instances_all) == len(positive_instances_by_ids)

    # split positive instances into chunks (330 each) for 4 jobs
    # positive_instances_chunks = chunks(positive_instances_all, 330)
    positive_instances_chunks = chunks(positive_instances_all, 425)
    print(type(positive_instances_chunks))
    print(len(positive_instances_chunks))

    # make sure all chunks sum up to the number of instances
    assert len(positive_instances_all) == sum([len(_) for _ in positive_instances_chunks])
    # and that they are all unique
    _ = set()
    for c in positive_instances_chunks:
        for cm in c:
            _.add(cm)
    assert len(_) == len(positive_instances_all)

    # make sure there are no intersections between chunks
    ids_of_chunks = set()
    for i, chunk in enumerate(positive_instances_chunks):
        for _ in chunk:
            assert isinstance(_, RedditThread)
            if _.get_last_comment_name() in ids_of_chunks:
                raise Exception("%s from chunk %d already in other chunk!" % (_.get_last_comment_name(), i))
            ids_of_chunks.add(_.get_last_comment_name())

    assert len(positive_instances_all) == len(ids_of_chunks)

    # and assign the current positive instances
    positive_instances = positive_instances_chunks[current_split_number]

    print("Length of current split positive instances", len(positive_instances))

    # project all instances to average word embeddings and lengths
    similarity_helper = SemanticSimilarityHelper()

    positive_instances_emb_vectors = dict()
    negative_instances_emb_vectors = dict()
    positive_instances_lengths = dict()
    negative_instances_lengths = dict()

    for instance in positive_instances:
        assert isinstance(instance, RedditThread)
        # we ignore the last comment here (the actual AH)
        positive_instances_emb_vectors[
            instance.get_last_comment_name()] = similarity_helper.average_embeddings_vector_thread(instance, True)
        positive_instances_lengths[instance.get_last_comment_name()] = sum([len(c.body) for c in instance.comments[:-1]])

    for instance in negative_instances:
        assert isinstance(instance, RedditThread)
        negative_instances_emb_vectors[
            instance.get_last_comment_name()] = similarity_helper.average_embeddings_vector_thread(instance)
        negative_instances_lengths[instance.get_last_comment_name()] = sum([len(c.body) for c in instance.comments])

    print("Pre-processing done, all average embeddings computed")
    print("Positive instances", len(positive_instances_emb_vectors))
    print("Negative instances", len(negative_instances_emb_vectors))

    def do_job(job_id, _positive_instance_id, _negative_instances_keys, _result_dict):
        # print("Started job", job_id)

        for negative_instance_id in _negative_instances_keys:
            distance = SemanticSimilarityHelper.distance_vec(
                positive_instances_emb_vectors[_positive_instance_id],
                negative_instances_emb_vectors[negative_instance_id],
                positive_instances_lengths[_positive_instance_id],
                negative_instances_lengths[negative_instance_id])

            _result_dict[negative_instance_id] = distance

    # we have positive_instances and negative_instances
    positive_to_negative_distances = dict()

    print("Need to compute %d distances" % len(positive_instances_emb_vectors))

    for counter, positive_instance_id in enumerate(positive_instances_emb_vectors):
        print("Counter: %d" % counter)

        start = datetime.now()

        temp_parallel_dict = manager.dict()

        job_number = 8

        # pool.map()
        total = len(negative_instances_emb_vectors.keys())
        chunk_size = int(math.ceil(total / job_number))
        current_slice = chunks(list(negative_instances_emb_vectors.keys()), chunk_size)
        # print(slice)
        jobs = []

        for i, negative_instances_keys in enumerate(current_slice):
            j = multiprocessing.Process(target=do_job,
                                        args=(
                                            i, positive_instance_id, negative_instances_keys,
                                            temp_parallel_dict))
            jobs.append(j)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        new_dict = dict(temp_parallel_dict)
        positive_to_negative_distances[positive_instance_id] = new_dict

        delta = datetime.now() - start

        # print(d)
        print("[%s] Computed all distances for %s (size: %d)" % (delta, positive_instance_id, len(new_dict)))

        print("Current size of the resulting dict: %d" % len(positive_to_negative_distances))
    # for key in positive_to_negative_distances:
    #     print(key)
    #     print(len(positive_to_negative_distances[key]))

    output_file_name = "%s_%d.pkl" % (output_file_prefix, current_split_number)

    with open(output_file_name, "wb") as f:
        pickle.dump(positive_to_negative_distances, f)
        f.close()


if __name__ == "__main__":
    # this first one was for ah/non-ah sampling
    # compute_parallel("ah-positive-negative-instances-all.pkl", "distance_dict")

    # now for threads 3
    # compute_parallel("threads-with-ah-threads-with-delta-context3.pkl",
    #                  "threads-with-ah-threads-with-delta-context3-distances")
    # now for threads 2
    compute_parallel("threads-with-ah-threads-with-delta-context2.pkl",
                     "threads-with-ah-threads-with-delta-context2-distances")
