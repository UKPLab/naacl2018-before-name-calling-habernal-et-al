from functools import total_ordering

from RedditComment import RedditComment


@total_ordering
class AnnotatedRedditComment(RedditComment):
    def _is_valid_operand(self, other):
        return hasattr(other, "name")

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.name == other.name

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.name < other.name

    def __hash__(self):
        """
        One funny note about Python :)

        By default, the __hash__() values of str, bytes and datetime objects are “salted” with an unpredictable
        random value. Although they remain constant within an individual Python process, they are not predictable
        between repeated invocations of Python.

        This is intended to provide protection against a denial-of-service caused by carefully-chosen inputs that
        exploit the worst case performance of a dict insertion, O(n^2) complexity.
        See http://www.ocert.org/advisories/ocert-2011-003.html for details.

        Changing hash values affects the iteration order of dicts, sets and other mappings. Python has never made
        guarantees about this ordering (and it typically varies between 32-bit and 64-bit builds).
        :return: hash
        """
        return self.name.__hash__()

    def __init__(self, reddit_comment: RedditComment):
        super().__init__()

        self.id = reddit_comment.id
        self.parent_id = reddit_comment.parent_id
        self.controversiality = reddit_comment.controversiality
        self.gilded = reddit_comment.gilded
        self.edited = reddit_comment.edited
        self.created_utc = reddit_comment.created_utc
        self.downs = reddit_comment.downs
        self.link_id = reddit_comment.link_id
        self.name = reddit_comment.name
        self.archived = reddit_comment.archived
        self.created = reddit_comment.created
        self.body_html = reddit_comment.body_html
        self.ups = reddit_comment.ups
        self.body = reddit_comment.body
        self.user_reports = reddit_comment.user_reports
        self.mod_reports = reddit_comment.mod_reports
        self.author_name = reddit_comment.author_name
        self.title = reddit_comment.title

        self.delta = False
        self.violated_rule = 0

    def is_ad_hominem(self) -> bool:
        """
        Returns true if and only if the comments is mod-labeled as ad hominem (multiple labels, such
        as low effort and ad hominem would return false)
        :return: boolean value
        """
        return self.violated_rule == 2

    @staticmethod
    def truncate_op_body(op_body: str) -> str:
        index = op_body.find('_____')

        # or it's a quoted text
        if index < 0:
            # the "- 3" is because of some weird behavior of str.find() if the text starts with "> *"
            index = op_body.find('Hello, users of CMV! This is a footnote from your moderators') - 3

        # in other form
        if index < 0:
            index = op_body.find('This is a footnote from the CMV moderators.') - 3

        if index:
            return op_body[0:index].strip()
        else:
            return op_body.strip()

    def __str__(self):
        # return "%s (%s): %d" % (self.name, self.author_name, self.violated_rule)
        return "%d%s[%s]" % (self.violated_rule, '∆' if self.delta else ' ', self.name)


pass
