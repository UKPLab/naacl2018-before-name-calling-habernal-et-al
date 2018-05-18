import copy
import os
import pickle

from AnnotatedRedditComment import AnnotatedRedditComment
from RedditComment import RedditComment


class RedditThread:
    def __init__(self):
        self.comments = []

    def get_last_comment_name(self) -> str:
        """
        Returns name (id) of the last comment. This is unique for each thread.
        :return: str
        """
        if not self.comments:
            raise Exception("The thread contains no comments")

        return self.comments[-1].name

    def has_deleted_comments(self) -> bool:
        """
        Returns true if any of the comments in the thread had been deleted (no author and '[deleted]' body)
        of is empty
        :return: bool
        """
        for comment in self.comments:
            assert isinstance(comment, RedditComment)
            # if comment.body == '[deleted]' or comment.body == '[removed]' or len(comment.body.strip()) == 0:
            if comment.is_deleted():
                # print("Thread has deleted comments:", ' -> '.join([x.__str__() for x in self.comments]))
                # print("Deleted comment")
                return True
            if comment.is_empty():
                # print("Empty comment")
                return True
        return False

    @staticmethod
    def load_comments_from_file(file_name: str) -> dict:
        """
        Reads all comments from the given file (which contains a single JSON object per line)
        and returns a map of RedditComment instances where the key is the name
        :param file_name:
        """
        result = dict()

        with open(file_name, 'r') as input_file:
            for line in input_file.readlines():
                reddit_comment = RedditComment.create_from_json(line)
                result[reddit_comment.name] = reddit_comment

        # print("Loaded %s comments from %s" % (len(result), file_name))

        return result

    @staticmethod
    def extract_all_paths(all_paths: list, stack: list, adjacency_matrix: dict, current_node: str) -> None:
        # add to the stack
        stack.append(current_node)
        # print("Current stack:", stack)

        # not leaf
        if current_node in adjacency_matrix and len(adjacency_matrix[current_node]) > 0:
            # print("And current node is in adjacency_matrix wich children:", adjacency_matrix[current_node].keys())
            current_node_dict = adjacency_matrix[current_node]
            assert isinstance(current_node_dict, dict)
            for child in current_node_dict.keys():
                RedditThread.extract_all_paths(all_paths, copy.deepcopy(stack), adjacency_matrix, child)
        # leaf
        else:
            all_paths.append(stack)
            # print("Leaf reached (%s), adding new path (length %s): %s" % (current_node, len(stack), stack))

    @staticmethod
    def reconstruct_threads_from_submission(reddit_comments: dict) -> list:
        """
        Given a map of RedditComment instances from a single submission, it creates
        all paths and return a list of RedditThread instances
        :param reddit_comments: list of RedditComment
        """
        assert len(reddit_comments) > 0

        root = None

        # create map of comment_id and their label (str, int)
        comment_id_violation_map = dict()

        # create map of comment_id and delta approval (str, bool)
        comment_id_delta_map = dict()

        # map of deleted comment_name and its author name reconstructed from its children
        deleted_comment_name_author_name_map = dict()

        # create an adjacency matrix
        adjacency_matrix = dict()
        for comment in reddit_comments.values():
            assert isinstance(comment, RedditComment)

            # is this the root?
            if comment.parent_id == '':
                assert root is None
                root = comment.name

            # find the parent and add parent->child entry
            parent_id = comment.parent_id
            child_id = comment.name

            if len(parent_id) > 0:
                if parent_id not in adjacency_matrix:
                    adjacency_matrix[parent_id] = dict()

                # check for "false" children - comments that only label the parent with
                # delta or rule violation
                rule_violation = comment.get_rule_violation()

                # reconstruct the author name of the deleted comment
                if rule_violation:
                    rule_violation_author_name = comment.get_rule_violation_author_name()
                    if rule_violation_author_name:
                        deleted_comment_name_author_name_map[parent_id] = rule_violation_author_name

                delta = comment.get_delta_awarded_bot()

                # only if parent didn't violate rules, add this edge to the matrix
                if rule_violation == 0 and not delta:
                    adjacency_matrix[parent_id][child_id] = comment
                elif not delta:
                    # there is not yet a label for the parent comment, add one
                    if parent_id not in comment_id_violation_map:
                        comment_id_violation_map[parent_id] = rule_violation
                        # print("Child %s labels parent %s with label %s" % (child_id, parent_id, rule_violation))
                    else:
                        # there's already a label, so we have to combine this one with the existing ones
                        new_label = RedditComment.merge_two_labels(comment_id_violation_map[parent_id], rule_violation)
                        # print("Parent post (%s) has been already labeled with label %s but there's yet another label (%s) from child (%s)" %
                        #                 (parent_id, comment_id_violation_map[parent_id], rule_violation, child_id))
                        # print("New label is ", new_label)
                        comment_id_violation_map[parent_id] = new_label

                else:
                    # we need the grandparent id because parent is the OP who gave delta
                    grandparent_id = reddit_comments[parent_id].parent_id
                    comment_id_delta_map[grandparent_id] = True

        # print(comment_id_delta_map)
        # print(comment_id_violation_map)
        # print(deleted_comment_name_author_name_map)

        # we must have a root
        assert root

        # print(adjacency_matrix, root)

        stack = []
        all_paths = []

        RedditThread.extract_all_paths(all_paths, copy.deepcopy(stack), adjacency_matrix, root)

        result = []

        # now convert the paths to RedditThread instances
        for path in all_paths:
            thread = RedditThread()
            for comment_name in path:
                # create a new annotated comment
                # print(comment_name)
                annotated_reddit_comment = AnnotatedRedditComment(reddit_comments[comment_name])
                annotated_reddit_comment.violated_rule = comment_id_violation_map.get(comment_name, 0)
                annotated_reddit_comment.delta = comment_id_delta_map.get(comment_name, False)
                # reconstruct author of rule-violating comments
                if annotated_reddit_comment.violated_rule > 0:
                    annotated_reddit_comment.author_name = deleted_comment_name_author_name_map.get(comment_name, '')
                thread.comments.append(annotated_reddit_comment)

            result.append(thread)

        return result

    @staticmethod
    def discard_corrupted_threads(threads: list) -> list:
        """
        Creates a new list that does not contain 'corrupted' threads (such as those with deleted comments)
        :param threads: list of RedditThread instances
        """
        assert isinstance(threads[0], RedditThread)

        return [thread for thread in threads if not thread.has_deleted_comments()]

    def has_some_ad_hominem(self) -> bool:
        """
        Returns true if any of the comments is ad hominem, false otherwise
        :return: boolean value
        """
        return any([comment.is_ad_hominem() for comment in self.comments])

    def has_some_delta(self) -> bool:
        """
        Returns true if any of the comments is a delta-comment
        :return: boolean value
        """
        return any([comment.delta for comment in self.comments])

    def get_positions_of_ad_hominem_comments(self) -> list:
        """
        Returns a list of indices pointing to comments that are ad-hominem
        :return: a list of integers (might be empty if no ad-hominem is present
        """

        result = []

        for i, comment in enumerate(self.comments):
            if comment.is_ad_hominem():
                result.append(i)

        return result

    def get_positions_of_delta_comments(self) -> list:
        """
        Returns a list of indices pointing to comments that are delta
        :return: a list of integers (might be empty if no delta is present)
        """
        result = []

        for i, comment in enumerate(self.comments):
            if comment.delta:
                result.append(i)

        return result

    @staticmethod
    def collect_all_comments(threads: list) -> set:
        """
        Collects all unique comments from the given list of threads and returns them
        as a set
        :param threads: list of RedditThread
        :return: set of AnnotatedRedditComment
        """
        name_comments_dict = dict()

        for thread in threads:
            assert isinstance(thread, RedditThread)
            for comment in thread.comments:
                name_comments_dict[comment.name] = comment

        return set(name_comments_dict.values())

    def __str__(self):
        return ' -> '.join([x.__str__() for x in self.comments])

    @staticmethod
    def load_or_unpickle_ad_hominem_threads() -> list:
        """
        Returns a list of RedditThread instances, each thread contains at least one
        ad hominem argument; uses pickle to cache the list to a file
        :return: a list of threads
        """
        result = []
        pickle_file = 'ad-hominem-threads-list-dump.pkl'
        if os.path.isfile(pickle_file):
            with open(pickle_file, "rb") as f:
                result = pickle.load(f)
                # make sure we're loading the correct data
                assert isinstance(result[0], RedditThread)
        else:
            main_dir = '/home/user-ukp/data2/cmv-full-2017-09-22/'
            files = [f for f in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir, f))]

            for f in files:
                comments = RedditThread.load_comments_from_file(os.path.join(main_dir, f))
                clean_threads = RedditThread.discard_corrupted_threads(
                    RedditThread.reconstruct_threads_from_submission(comments))

                for thread in clean_threads:
                    assert isinstance(thread, RedditThread)
                    if thread.has_some_ad_hominem():
                        result.append(thread)
            with open(pickle_file, 'wb') as f:
                pickle.dump(result, f)

        # check that all comments have an author
        # for thread in result:
        #     for comment in thread.comments:
        #         if len(comment.author_name) < 1:
        #             raise Exception("Author name must not be empty")

        return result

    @staticmethod
    def group_threads_by_original_post(threads: list) -> dict:
        """
        Groups all threads by their first comment (original post), thus re-creating
        the submissions
        :param threads: list of RedditThread instances
        :return: dict(op = AnnotatedRedditComment; list[RedditThread])
        """
        result = dict()

        for thread in threads:
            op = thread.comments[0]

            if op not in result:
                result[op] = []
            result[op].append(thread)

        return result

    @staticmethod
    def export_all_deleted_comments_to_json():
        main_dir = '/home/user-ukp/data2/cmv-full-2017-09-22/'
        files = [f for f in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir, f))]

        comments_to_export = []

        for f in files:
            comments = RedditThread.load_comments_from_file(os.path.join(main_dir, f))
            clean_threads = RedditThread.discard_corrupted_threads(
                RedditThread.reconstruct_threads_from_submission(comments))

            for comment in RedditThread.collect_all_comments(clean_threads):
                assert isinstance(comment, AnnotatedRedditComment)
                label = comment.violated_rule
                # update counter
                if label != 0:
                    comments_to_export.append(comment)

        with open('/tmp/deleted-comments.json', 'w') as f:
            for comment in comments_to_export:
                assert isinstance(comment, AnnotatedRedditComment)
                f.write(comment.to_json_string() + "\n")

    @staticmethod
    def filter_threads_with_all_authors(threads: list) -> list:
        """
        Creates a new list of RedditThread in which each comment has an author
        :param threads: list of RedditThread
        :return: a new list of RedditThread
        """
        result = []
        for thread in threads:
            assert isinstance(thread, RedditThread)
            if all([c.author_name for c in thread.comments]):
                result.append(thread)
        return result


def __main__():
    # comments = RedditThread.load_comments_from_file('/home/user-ukp/data2/cmv-full-2017-09-22/16ralh.json')
    # comments = RedditThread.load_comments_from_file('/home/user-ukp/data2/cmv-full-2017-09-22/6jecb2.json')
    # comments = RedditThread.load_comments_from_file('/home/user-ukp/data2/cmv-full-2017-09-22/3n7h2a.json')
    # comments = RedditThread.load_comments_from_file('/home/user-ukp/data2/cmv-full-2017-09-22/71k3km.json')
    comments = RedditThread.load_comments_from_file('/home/user-ukp/data2/cmv-full-2017-09-22/6xo972.json')

    # multiple labels
    # comments = RedditThread.load_comments_from_file('/home/user-ukp/data2/cmv-full-2017-09-22/6zmhxk.json')
    # comments = RedditThread.load_comments_from_file('/home/user-ukp/data2/cmv-full-2017-09-22/5b42cj.json')

    threads = RedditThread.reconstruct_threads_from_submission(comments)
    clean_threads = RedditThread.discard_corrupted_threads(threads)
    print("%s threads (%s discarded)" % (len(clean_threads), (len(threads) - len(clean_threads))))

    for thread in clean_threads:
        print(thread)
        print(thread.has_some_ad_hominem())

    hominem_threads = RedditThread.load_or_unpickle_ad_hominem_threads()

    print(len(RedditThread.collect_all_comments(hominem_threads)))

    print(len(RedditThread.group_threads_by_original_post(hominem_threads)))

    # make sure we know all authors even of the ad-hominem

    # RedditThread.export_all_deleted_comments_to_json()

    # print(len(RedditThread.load_or_unpickle_all_threads()))

    # if len(sys.argv) < 2:
    #     raise Exception('Expected output directory argument')
    # extractor = RedditCMVExtractor(sys.argv[1])
    # extractor.run_full_scraping()
    # extractor.extract_single_submission('6xo972')
    # extractor.extract_single_submission('16ralh')

    # extract all submission ids from changemyview
    # extractor.fetch_or_load_cmv_submission_ids()


if __name__ == "__main__":
    __main__()
