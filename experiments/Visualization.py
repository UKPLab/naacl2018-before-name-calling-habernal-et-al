import json

from typing import IO

from Colomaps import Colormaps
from vocabulary import Vocabulary


class Visualization:
    def __init__(self):

        self.latex_head = '''\\documentclass[10pt,a4paper]{article}
\\usepackage[left=1.00cm, right=1.00cm, top=1.00cm, bottom=2.00cm]{geometry}
\\usepackage{color}
\\begin{document}
        '''
        self.latex_footer = '''
        \\end{document}'''

    def print_docs_latex(self, docs: list, section: str, f: IO) -> None:
        f.write("\\section{%s}\n\n" % section)
        for doc in docs:
            f.write(Visualization._doc_to_latex(doc, vocabulary))

    def print_fold_latex(self, input_file: str, output_html: str, vocabulary: Vocabulary) -> None:
        # open a single json file
        with open(input_file) as f:
            loaded = json.load(f)
            f.close()
        assert isinstance(loaded, list)

        # filter first true positives (AH as AH)
        true_positives = [_ for _ in loaded if _['gold'] == 'ah' and _['predicted'] == 'ah']
        # and false negatives (AH as delta)
        false_negatives = [_ for _ in loaded if _['gold'] == 'ah' and _['predicted'] == 'delta']
        # and false positives
        false_positives = [_ for _ in loaded if _['gold'] == 'delta' and _['predicted'] == 'ah']

        with open(output_html, "w") as f:
            f.write(self.latex_head)

            self.print_docs_latex(true_positives, 'True positives (AH as AH) ' + str(len(true_positives)), f)
            self.print_docs_latex(false_positives,
                                  'False positives (Type I error) (delta as AH) ' + str(len(false_positives)), f)
            self.print_docs_latex(false_negatives,
                                  'False negatives (Type II error) (AH as delta) ' + str(len(false_negatives)), f)

            f.write(self.latex_footer)
            f.close()

    @staticmethod
    def _doc_to_latex(doc: dict, vocabulary: Vocabulary) -> str:
        global token_counter
        tokens = doc['words']
        weights = doc['weights']
        doc_id = doc['id'].replace('_', '\\_')
        gold_label = doc['gold']
        predicted_label = doc['predicted']

        string_buffer = []

        for i in range(len(tokens)):
            token = tokens[i]
            token_counter += 1

            linebreak = '___start__' in token

            if not token in vocabulary.word_to_index_mapping:
                if not linebreak:
                    # if this is OOV, replace with OOV
                    token = 'OOV'
                else:
                    token = '(OOV_comment_begin)'

            assert isinstance(token, str)
            token = token.replace('_', '\\_').replace('&', '\\&').replace('$', '\\$').replace('#', '\\#').replace('^',
                                                                                                                  ' ').replace(
                '%', '\\%')
            print(i, token, weights[i])

            # black & white heatmap
            w = min(float(1 - weights[i] + 0.1), 1.0)
            colormap = (w, w, w)

            # blue
            colormap = Colormaps.map_to_rgb_blue(float(weights[i]))

            if weights[i] > 0.4:
                token = "\\textcolor{white}{%s}" % token

            if linebreak:
                string_buffer.append('\n\n\\noindent')

            # box only if the weights are different from white
            if w < 1.0:
                string_buffer.append(
                    "\\colorbox[rgb]{%.2f,%.2f,%.2f}{\\strut %s}" % (colormap[0], colormap[1], colormap[2], token))
            else:
                string_buffer.append(token)

            # and a space
            string_buffer.append(' ')

            # as colorbox doesn't support line breaking, add extra space every 5 words
            # if len(string_buffer) % 3 == 0:
            #     string_buffer.append(' ')

            # string_buffer.append(token)

            # if linebreak:
            #     string_buffer.append('\n\n')
        print(string_buffer)

        print("-----------")

        # prediction type
        prediction_result = " (%s--%s)" % (gold_label, predicted_label)

        return "\n\\subsubsection*{%s %s}\n\n%s\n\n" % (doc_id, prediction_result, ''.join(string_buffer))

    @staticmethod
    def _doc_to_html(doc: dict) -> str:
        tokens = doc['words']
        weights = doc['weights']
        doc_id = doc['id']

        string_buffer = []

        for i in range(len(tokens)):
            print(i, tokens[i], weights[i])
            if '___start__' in tokens[i]:
                string_buffer.append(' <br />')
            string_buffer.append('<span data-weight="' + str(100 * float(weights[i])) + '">' + tokens[i] + '</span>')
            if '___start__' in tokens[i]:
                string_buffer.append(' <br />')
        # print(string_buffer)

        print("-----------")
        return '<div class="answer"><h1>ID: ' + str(doc_id) + "</h1>" + "".join(string_buffer) + "</div>"


if __name__ == "__main__":
    token_counter = 0
    vocabulary = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
    Visualization().print_fold_latex("visualization-context3/fold1.json",
                                     "/tmp/temp1.tex", vocabulary)
    Visualization().print_fold_latex("visualization-context3/fold2.json",
                                     "/tmp/temp2.tex", vocabulary)
    print(token_counter)
