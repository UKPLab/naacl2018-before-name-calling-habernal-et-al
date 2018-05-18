import json
import re


class RedditComment:
    from praw.models.reddit.comment import Comment
    from praw.models.reddit.submission import Submission

    def __init__(self):
        self.id = ''
        self.parent_id = ''
        self.controversiality = 0
        self.gilded = 0
        self.edited = False
        self.created_utc = 0.0
        self.downs = 0
        self.link_id = ''
        self.name = ''
        self.archived = False
        self.created = 0.0
        self.body_html = ''
        self.ups = 0
        self.body = ''
        self.user_reports = []
        self.mod_reports = []
        self.author_name = ''
        self.title = ''

    @staticmethod
    def create_from_reddit_comment(comment: Comment):
        """
        Create a new instance from the Reddit Comment object
        :param comment: Comment object
        :return: new instance of RedditComment
        """
        result = RedditComment()

        result.id = str(comment.id)
        result.parent_id = str(comment.parent_id)
        result.controversiality = int(comment.controversiality)
        result.gilded = int(comment.gilded)
        result.edited = bool(comment.edited)
        result.created_utc = float(comment.created_utc)
        result.downs = int(comment.downs)
        result.link_id = str(comment.link_id)
        result.name = str(comment.name)
        result.archived = bool(comment.archived)
        result.created = float(comment.created)
        result.body_html = str(comment.body_html)
        result.ups = int(comment.ups)
        result.body = str(comment.body)
        result.user_reports = list(comment.user_reports)
        result.mod_reports = list(comment.mod_reports)
        result.author_name = str(comment.author.id) if comment.author is not None else ''

        return result

    @staticmethod
    def create_from_reddit_comment_json(comment: dict):
        """
        Create a new instance from the Reddit Comment object
        :param comment: Comment object
        :return: new instance of RedditComment
        """
        result = RedditComment()

        result.id = str(comment['id'])
        result.parent_id = str(comment['parent_id'])
        result.controversiality = int(comment['controversiality'])
        result.gilded = int(comment['gilded'])
        result.edited = bool(comment['edited'])
        result.created_utc = float(comment['created_utc'])
        result.downs = int(comment['downs'])
        result.link_id = str(comment['link_id'])
        result.name = str(comment['name'])
        result.archived = bool(comment['archived'])
        result.created = float(comment['created'])
        result.body_html = str(comment['body_html'])
        result.ups = int(comment['ups'])
        result.body = str(comment['body'])
        result.user_reports = list(comment['user_reports'])
        result.mod_reports = list(comment['mod_reports'])

        if isinstance(comment['author'], dict):
            result.author_name = str(comment['author']['name'])

        return result

    @staticmethod
    def create_from_reddit_submission(submission: Submission):
        """
        Create a new instance from the Reddit Comment object
        :param submission: Comment object
        :return: new instance of RedditComment
        """
        result = RedditComment()

        result.id = str(submission.id)
        result.parent_id = ''  # no parent
        result.controversiality = 0  # no controversiality
        result.gilded = int(submission.gilded)
        result.edited = bool(submission.edited)
        result.created_utc = float(submission.created_utc)
        result.downs = int(submission.downs)
        result.link_id = ''  # none
        result.name = str(submission.name)
        result.archived = bool(submission.archived)
        result.created = float(submission.created)
        result.body_html = str(submission.selftext_html)
        result.ups = int(submission.ups)
        result.body = str(submission.selftext)
        result.user_reports = list(submission.user_reports)
        result.mod_reports = list(submission.mod_reports)
        result.author_name = str(submission.author.name) if submission.author is not None else ''
        result.title = str(submission.title)

        return result

    @staticmethod
    def create_from_reddit_submission_json(submission: dict):
        result = RedditComment()

        result.id = str(submission['id'])
        result.parent_id = ''  # no parent
        result.controversiality = 0  # no controversiality
        result.gilded = int(submission['gilded'])
        result.edited = bool(submission['edited'])
        result.created_utc = float(submission['created_utc'])
        result.created = float(submission['created'])
        result.downs = int(submission['downs'])
        result.link_id = ''  # none
        result.name = str(submission['name'])
        result.archived = bool(submission['archived'])
        result.body_html = str(submission['selftext_html'])
        result.ups = int(submission['ups'])
        result.body = str(submission['selftext'])
        result.user_reports = list(submission['user_reports'])
        result.mod_reports = list(submission['mod_reports'])

        if 'author' in submission and submission['author'] is not None and 'name' in submission['author']:
            result.author_name = str(submission['author']['name'])

        result.title = str(submission['title'])

        return result

    def to_json_string(self) -> str:
        """
        Conversion to JSON
        :return: JSON string
        """
        return json.dumps(self.__dict__, sort_keys=True)

    @staticmethod
    def create_from_json(json_str: str):
        """
        Re-creates an instance from a JSON string
        :param json_str: JSON string
        :return: new instance of RedditComment
        """
        result = RedditComment()

        loaded_dict = json.loads(json_str)
        assert isinstance(loaded_dict, dict)

        result.id = str(loaded_dict.get('id', ''))
        result.parent_id = str(loaded_dict.get('parent_id', ''))
        result.title = str(loaded_dict.get('title', ''))
        result.controversiality = int(loaded_dict.get('controversiality', 0))
        result.gilded = int(loaded_dict.get('gilded', 0))
        result.edited = bool(loaded_dict.get('edited', False))
        result.created_utc = float(loaded_dict.get('created_utc', 0.0))
        result.downs = int(loaded_dict.get('downs', 0))
        result.link_id = str(loaded_dict.get('link_id', ''))
        result.name = str(loaded_dict.get('name', ''))
        result.archived = bool(loaded_dict.get('archived', False))
        result.created = float(loaded_dict.get('created', 0.0))
        result.body_html = str(loaded_dict.get('body_html', ''))
        result.ups = int(loaded_dict.get('ups', 0))
        result.body = str(loaded_dict.get('body', ''))
        result.user_reports = list(loaded_dict.get('user_reports', []))
        result.mod_reports = list(loaded_dict.get('mod_reports', []))
        result.author_name = str(loaded_dict.get('author_name', ''))

        return result

    def is_deleted(self) -> bool:
        """
        Returns true if this comment was deleted from Reddit; false otherwise
        :return: boolean value
        """
        return self.author_name == '' and self.body == '[deleted]'

    def is_empty(self) -> bool:
        return len(self.body.strip()) == 0

    def get_rule_violation(self) -> int:
        """
        Returns an integer label if this comments is a meta-post describing the previous post
        as being removed for violating rules. Multiple labels do occur, so they are packed into
        a single (e.g., 1, 2 -> 12; 5, 2 -> 25)
        :return: integer or 0 if no rule violation is mentioned
        """
        if "your comment has been removed" in self.body:
            pattern = re.compile("> Comment Rule (\d+)")
            # convert to an integer (e.g., 1, 2 -> 12; 5, 2 -> 25)
            # str_value = ''.join(sorted(pattern.findall(self.body)))
            # result = int(str_value) if str_value != '' else 0

            result = RedditComment.create_rule_violation_label_from_str_set(set(pattern.findall(self.body)))
            return result

        return 0

    def get_rule_violation_author_name(self) -> str:
        """
        Finds the author of the deleted comment as mentioned in the comment; if no name
        is available, returns an empty string
        :return: author name or empty string
        """
        author_pattern = re.compile('(\S+), your comment has been removed:')
        found = author_pattern.findall(self.body)
        if len(found) == 1:
            return found[0]
        else:
            return ''

    @staticmethod
    def create_rule_violation_label_from_str_set(str_set: set) -> int:
        """
        Given a list of string integers ('violation rule' numbers), sorts them
        and return as a single integer; for example
        1, 2 -> 12
        or
        5, 2 -> 25
        :param str_set: set of strings
        :return: integer or 0 if empty list
        """
        str_value = ''.join(sorted(list(str_set)))
        return int(str_value) if str_value != '' else 0

    @staticmethod
    def merge_two_labels(label1: int, label2: int) -> int:
        # convert int number to string and set of characters and merge (union)
        union_set = set(list(str(label1))).union(set(list(str(label2))))
        return RedditComment.create_rule_violation_label_from_str_set(union_set)

    def get_delta_awarded_bot(self) -> bool:
        return self.body.startswith('Confirmed: 1 delta awarded to')

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not(self == other)

pass
