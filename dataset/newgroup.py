from glob import glob
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
from .ssl_dataset import BaseSSLDataset


TRAIN_PATH = "dataset/file/20news-bydate-train/*"
TEST_PATH = "dataset/file/20news-bydate-test/*"


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition("\n\n")

    return after


_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:" r"|^In article|^Quoted from|^\||^>)"
)


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
    return "\n".join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.
    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split("\n")
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip("-") == "":
            break

    if line_num > 0:
        return "\n".join(lines[:line_num])
    else:
        return text


class NewGroupDataset(BaseSSLDataset):
    def __init__(self) -> None:
        super().__init__()
        print(os.getcwd())

        train_files = glob(TRAIN_PATH)
        test_files = glob(TEST_PATH)
        labelEncoder = LabelEncoder()

        file_paths = train_files + test_files
        self.name = "NewsGroup"

        self.X = []
        self.y = []

        for folder in tqdm(file_paths):
            key = folder.split("\\")[-1]
            for file in os.listdir(folder):
                with open(
                    os.path.join(folder, file), encoding="latin-1"
                ) as opened_file:
                    self.y.append(key)
                    text = opened_file.read()
                    self.X.append(text)

        self.X = [self.preprocess(i) for i in self.X]

        self.y = labelEncoder.fit_transform(self.y)

        self.tfidf_transform()

        # np.save('new_group_X.pkl', self.X)

        # # with open('new_group_X.pkl', 'wb') as f:
        # #     pkl.dump(self.X, f)

        # with open('new_group_y.pkl', 'wb') as f:
        #     pkl.dump(self.y, f)

    def preprocess(self, text):
        text = strip_newsgroup_footer(text)
        text = strip_newsgroup_header(text)
        text = strip_newsgroup_quoting(text)
        text = re.sub("\d+", " ", text)

        return text

    def tfidf_transform(self):
        self.vectorizer = TfidfVectorizer(max_features=6000)
        self.vectorizer.fit(self.X)
        self.X = self.vectorizer.transform(self.X)
        self.X = self.X.toarray()
        self.X = np.array(self.X)


if __name__ == "__main__":
    dataset = NewGroupDataset()

    print(dataset.get_data(40))
