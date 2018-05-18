import json
import random

import preprocessing
from tcframework import LabeledTokenizedDocumentReader, LabeledTokenizedDocument


class JSONPerLineDocumentReader(LabeledTokenizedDocumentReader):
    def read_instances(self, input_path: str, instances_limit: int = -1) -> list:
        result = []

        with open(input_path) as f:
            for line in f:
                result.append(self.line_to_instance(line))

        random.shuffle(result)

        if instances_limit > 0:
            return result[:instances_limit]
        else:
            return result

    def line_to_instance(self, line: str) -> LabeledTokenizedDocument:
        m = json.loads(line)

        result = LabeledTokenizedDocument()
        result.id = m["name"]
        result.label = 'AH' if m["violated_rule"] == 2 else 'None'
        text_without_quotations = JSONPerLineDocumentReader.replace_quoted_text_with_special_token(m["body"])

        result.tokens = preprocessing.tokenize(text_without_quotations)

        return result

    @staticmethod
    def replace_quoted_text_with_special_token(text: str) -> str:
        """
        Replaces all quoted paragraphs with a single special token
        :param text: comment text
        :return: text with replacements
        """
        paragraphs = text.splitlines(keepends=False)
        replaced = []
        for paragraph in paragraphs:
            if paragraph.startswith('>'):
                replaced.append('__quoted_text__')
            else:
                replaced.append(paragraph)

        return '\n'.join(replaced)


class DeltaVersusAHMinus1Reader(JSONPerLineDocumentReader):

    def line_to_instance(self, line: str) -> LabeledTokenizedDocument:
        m = json.loads(line)

        result = LabeledTokenizedDocument()
        result.id = m["name"]
        result.label = 'Delta' if m['delta'] else 'AH-1'
        result.tokens = preprocessing.tokenize(m["body"])

        return result


class AHVersusDeltaThreadReader(LabeledTokenizedDocumentReader):
    def read_instances(self, input_path: str, instances_limit: int = -1) -> list:
        import glob
        json_files = glob.glob(input_path + "/*.json")

        result = []

        for json_file in json_files:
            result.append(self.file_to_instance(json_file))

        random.shuffle(result)

        if instances_limit > 0:
            return result[:instances_limit]
        else:
            return result

    def file_to_instance(self, file_name: str) -> LabeledTokenizedDocument:
        relative_name = file_name.split('/')[-1]
        label = relative_name.split('_', 2)[1]
        file_id = relative_name.split('_', 2)[2]

        assert label in ['ah', 'delta']

        result = LabeledTokenizedDocument()
        result.label = label
        result.id = relative_name

        # read all lines first
        lines = []
        with open(file_name) as f:
            for line in f:
                lines.append(line)

        # remove last comment if this is AH
        if 'ah' == label:
            lines = lines[:-1]

        # here we can adjust the total size of the context; now there are 3 last comments
        # lines = lines[-2:]  # would leave only last two ones
        # print(len(lines))

        for line in lines:
            m = json.loads(line)
            result.tokens.append('___' + m["name"] + '___start__')
            result.tokens.extend(
                preprocessing.tokenize(JSONPerLineDocumentReader.replace_quoted_text_with_special_token(m["body"])))

        # print(result.id)
        # print(result.label)
        # print(result.tokens)

        return result


if __name__ == "__main__":
    AHVersusDeltaThreadReader("data/sampled-threads-ah-delta-context3")
