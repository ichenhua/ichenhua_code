import pandas as pd
import numpy as np


import numpy as np

class TfidfCounter():
    def __init__(self) -> None:
        self.termset = set()  #('a','b','c')
        self.tfs = {}  #{'A':{'a':0.1, 'b':0.2}, 'B':{'b':0.3, 'c':0.2}}
        self.idfs = {}  #{'a':0.1, 'b':0.2, 'c':0.5}
        self.tfidfs = {}  #{'A':{'a':0.1, 'b':0.2, 'c':0.1}, 'B':{'a':0.3, 'b':0.3, 'c':0.2}}

    def add(self, docid, term_list):
        self.termset = self.termset.union(term_list)
        self.compute_tf(docid, term_list)

    def compute_tf(self, docid, term_list):
        count = len(term_list)
        tf = {}
        for term in term_list:
            tf[term] = term_list.count(term) / count
        self.tfs[docid] = tf

    def compute(self):
        self.compute_idf()
        self.compute_tfidf()

    def compute_idf(self):
        total = len(self.tfs)
        for term in self.termset:
            count = 0
            for tfs in self.tfs.values():
                if term in tfs.keys():
                    count += 1
            self.idfs[term] = np.log10((total + 1) / (count))

    def compute_tfidf(self):
        for docid, tfs in self.tfs.items():
            tfidf = {}
            for term, tf in tfs.items():
                tfidf[term] = tf * self.idfs[term]
            self.tfidfs[docid] = dict(sorted(tfidf.items(), key=lambda x:x[1], reverse=True))

    def get_tfidf(self):
        return self.tfidfs

    def get_termset(self):
        return self.termset


docA = 'the cat sat on my bed'
docB = 'the dog sat on my knees'

bowA = docA.split()
bowB = docB.split()

counter = TfidfCounter()
counter.add('a', bowA)
counter.add('b', bowB)

counter.compute()

# print(counter.tfs)
# print(counter.idfs)

print(pd.DataFrame(counter.tfidfs))
