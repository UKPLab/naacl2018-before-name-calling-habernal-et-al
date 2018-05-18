import os
import praw
import json
import sys
from pathlib import Path

import time
from praw.models.reddit.comment import Comment
from praw.models.reddit.redditor import Redditor
from praw.models.reddit.submission import Submission

import jsonpickle

from RedditComment import RedditComment


class RedditCMVExtractor:
    def __init__(self, output_dir: str):
        # Warning: you need to provide your credentials
        self.reddit = praw.Reddit(client_id='loremipsum',
                                  client_secret='loremipsum',
                                  password='loremipsum',
                                  user_agent='loremipsum',
                                  username='loremipsum')
        print(self.reddit.user.me())

        # storing all ids
        self.all_cmv_submissions_ids = []

        # output dir
        self.output_dir = output_dir

    def download_submission(self, submission_id: str) -> Submission:
        submission = self.reddit.submission(id=submission_id)
        # replace all 'more comments' by their comment content
        submission.comments.replace_more(limit=None)

        assert isinstance(submission, Submission)

        return submission

    def save_comments_to_json(self, submission_id: str, comments: list) -> None:
        comments_file = self.output_dir + "/" + submission_id + ".json"

        # convert to JSON string objects
        comments_json = [comment.to_json_string() for comment in comments]

        with open(comments_file, 'w') as outfile:
            for comment_json in comments_json:
                outfile.write("%s\n" % comment_json)

        print(submission_id, 'successfully saved')
        print('Comments:', len(comments))

    # DEPRECATED; TOO SLOW
    def ___extract_comment_thread(self, authors: dict, comments: list, current_comment: Comment) -> None:
        # there is a single Comment for processing
        assert isinstance(current_comment, Comment)

        print('DEBUG All comments:', len(comments))

        reddit_comment = RedditComment.create_from_reddit_comment(current_comment)
        # print(reddit_comment.to_json_string())

        # extract author, if known
        if isinstance(current_comment.author, Redditor):
            reddit_author = RedditAuthor.create_from_reddit(current_comment.author)
            authors[reddit_author.id] = reddit_author

        # add to the results
        comments.append(reddit_comment)

        for reply in current_comment.replies:
            self.___extract_comment_thread(authors, comments, reply)

    def extract_comment_thread_json(self, comments: list, current_comment: dict) -> None:
        # comment is a dict
        assert isinstance(current_comment, dict)

        reddit_comment = RedditComment.create_from_reddit_comment_json(current_comment)

        # add to the results
        comments.append(reddit_comment)

        # iterate recursively over replies
        for reply in current_comment['_replies']['_comments']:
            self.extract_comment_thread_json(comments, reply)

    def extract_single_submission(self, submission_id: str) -> None:
        print('Fetching %s ... ' % submission_id, end='', flush=True)
        submission = self.download_submission(submission_id)

        # 'freeze' the whole submission thread to JSON using jsonpickle
        # this is just a string
        submission_json_frozen_str = jsonpickle.encode(submission)
        # encode back to a standard JSON
        submission_json_dict = json.loads(submission_json_frozen_str)
        # which is a standard dictionary, not a Reddit object
        assert isinstance(submission_json_dict, dict)

        # list of all comments for the current submission
        comments = []

        # convert json to RedditComment and add to all comments
        first_post = RedditComment.create_from_reddit_submission_json(submission_json_dict)
        comments.append(first_post)

        top_level_comment_dict = submission_json_dict['_comments']['_comments']

        for top_level_comment in top_level_comment_dict:
            self.extract_comment_thread_json(comments, top_level_comment)

        # and save to the output files
        self.save_comments_to_json(submission_id, comments)

    def extract_all_cmv_submissions_ids(self) -> list:
        """
        Extracts ids of all submissions in the CMV sub-reddit
        and stores them into self.all_cmv_submission_ids
        """
        subreddit = self.reddit.subreddit('changemyview')

        result = []

        for submission in subreddit.submissions():
            result.append(submission.id)

        return result

    def fetch_or_load_cmv_submission_ids(self) -> None:
        """
        All submissions IDs are either fetched from reddit or loaded from
        a cache file (cmv_submission_ids.json)
        """
        # try to load first
        json_file_name = 'cmv_submission_ids.json'

        if os.path.isfile(json_file_name):
            with (open(json_file_name, 'r')) as infile:
                self.all_cmv_submissions_ids = json.load(infile)
        else:
            # extract all submission ids from changemyview
            self.all_cmv_submissions_ids = self.extract_all_cmv_submissions_ids()
            # and store them to a JSON file
            with open(json_file_name, 'w') as outfile:
                json.dump(self.all_cmv_submissions_ids, outfile, indent=4)

        print("Submissions: ", len(self.all_cmv_submissions_ids))

    def run_full_scraping(self) -> None:
        """
        Fetches all Reddit submissions
        """
        self.fetch_or_load_cmv_submission_ids()

        sleep_timeout_s = 2

        for submission_id in self.all_cmv_submissions_ids:
            if not Path(self.output_dir + "/" + submission_id + ".json").is_file():
                self.extract_single_submission(submission_id)

                # give some timeout
                print("Sleep for %.1f seconds... " % sleep_timeout_s, end='', flush=True)
                time.sleep(sleep_timeout_s)
                print("done.", flush=True)
            else:
                print("%s already fetched, skipping..." % submission_id)


def __main__():
    if len(sys.argv) < 2:
        raise Exception('Expected output directory argument')

    extractor = RedditCMVExtractor(sys.argv[1])
    extractor.run_full_scraping()
    # extractor.extract_single_submission('6xo972')
    # extractor.extract_single_submission('16ralh')

    # extract all submission ids from changemyview
    # extractor.fetch_or_load_cmv_submission_ids()


if __name__ == "__main__":
    # since we're using recursion for comment tree traversal, we need more stack for extra large threads
    sys.setrecursionlimit(10000)
    __main__()
