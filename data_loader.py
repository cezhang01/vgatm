import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from stanfordcorenlp import StanfordCoreNLP
import os
import collections
from tqdm import tqdm
import string


class Data():

    def __init__(self, args):

        print('Preparing data...')
        self.parse_args(args)
        self.load_data()
        self.split_data()
        self.generate_word_word_semantic_links()
        self.generate_word_word_pmi_and_syntactic_links()
        self.sample_neighbors()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.minibatch_size = args.minibatch_size
        self.training_ratio = args.training_ratio
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.word_word_graph_window_size = args.word_word_graph_window_size
        self.word_word_graph_num_neighbors = args.word_word_graph_num_neighbors
        self.author_prediction = args.author_prediction
        self.word_embedding_model = 'glove'
        self.word_embedding_dimension = 300

    def load_data(self):

        print('Loading data...')
        self.documents = self.load_files('./data/' + self.dataset_name + '/contents.txt')
        if not self.author_prediction:
            self.doc_contents_bow = np.loadtxt('./data/' + self.dataset_name + '/contents_bow.txt')
        self.num_documents = len(self.documents)
        self.citations, self.num_citations = self.symmatrize_links(self.load_files('./data/' + self.dataset_name + '/citations.txt'))
        self.pairwise_citations = self.generate_pairwise_links(self.citations)
        self.authors = self.load_files('./data/' + self.dataset_name + '/authors.txt')
        self.num_authors = len(np.unique([author_id for doc_id in self.authors for author_id in self.authors[doc_id]]))
        if self.author_prediction:
            self.hide_authors(self.authors)
        self.coauthors = self.generate_coauthors(self.authors)
        self.venues_available = os.path.isfile('./data/' + self.dataset_name + '/venues.txt')
        if self.venues_available:
            self.venues = np.reshape(np.loadtxt('./data/' + self.dataset_name + '/venues.txt', dtype=int), [-1, 1])
            self.num_venues = int(np.amax(self.venues) + 1)
            self.covenues = np.reshape(np.arange(self.num_venues, dtype=int), [-1, 1])
        self.labels_available = os.path.isfile('./data/' + self.dataset_name + '/labels.txt')
        if self.labels_available:
            self.labels = np.loadtxt('./data/' + self.dataset_name + '/labels.txt')
            self.num_labels = len(np.unique(self.labels))
        self.word_embeddings = np.loadtxt('./data/' + self.dataset_name + '/word_embeddings_' + self.word_embedding_model + '_' + str(self.word_embedding_dimension) + 'd.txt', dtype=float)
        self.voc = np.genfromtxt('./data/' + self.dataset_name + '/voc.txt', dtype=str)
        self.word2id = {}
        for word_id, word in enumerate(self.voc):
            self.word2id[word] = word_id
        self.num_words = len(self.voc)

        self.doc_contents_word_embed = []
        for doc_id in range(len(self.documents)):
            self.doc_contents_word_embed.append(np.mean(self.word_embeddings[self.documents[doc_id]], axis=0))
        self.doc_contents_word_embed = np.array(self.doc_contents_word_embed)

    def load_files(self, file_path):

        files = collections.defaultdict(list)
        with open(file_path) as file:
            for row_id, row in enumerate(file):
                row = row.split()
                row = [int(i) for i in row]
                files[row_id] = row

        return files

    def symmatrize_links(self, links):

        symmetric_links_set = collections.defaultdict(set)
        for id1 in links:
            for id2 in links[id1]:
                symmetric_links_set[id1].add(id2)
                symmetric_links_set[id2].add(id1)
                # symmetric_links_set[id1].add(id1)
                # symmetric_links_set[id2].add(id2)
        symmetric_links, num_links = collections.defaultdict(list), 0
        for id in symmetric_links_set:
            symmetric_links[id] = list(symmetric_links_set[id])
            num_links += len(symmetric_links_set[id])
        num_links -= len(links)

        return symmetric_links, num_links

    def hide_authors(self, authors):

        self.test_doc_author_links = []
        doc_author_links = self.generate_pairwise_links(authors)
        for author_id in range(self.num_authors):
            docs_with_same_author = doc_author_links[doc_author_links[:, 1] == author_id]
            if len(docs_with_same_author) < 3:
                continue
            idx = np.random.choice(len(docs_with_same_author))
            doc_id = docs_with_same_author[idx, 0]
            if len(self.authors[doc_id]) == 1:
                if len(self.citations[doc_id]) > 1:
                    neighbor_doc_id = np.array(self.citations[doc_id])[self.citations[doc_id] != doc_id][-1]
                    self.authors[doc_id].append(self.authors[neighbor_doc_id][0])
                else:
                    self.authors[doc_id].append(self.authors[0][0])
            self.test_doc_author_links.append([doc_id, author_id])
            self.authors[doc_id].remove(author_id)
        self.test_doc_author_links = np.array(self.test_doc_author_links)
        print('%.4f doc-author links are hidden!' % (len(self.test_doc_author_links) / len(doc_author_links)))

    def generate_coauthors(self, authors):

        coauthors_set = collections.defaultdict(set)
        for doc_id in authors:
            for author_id1 in authors[doc_id]:
                for author_id2 in authors[doc_id]:
                    coauthors_set[author_id1].add(author_id2)
                    coauthors_set[author_id1].add(author_id1)
                    coauthors_set[author_id2].add(author_id2)
        coauthors = collections.defaultdict(list)
        for author_id in coauthors_set:
            coauthors[author_id] = list(coauthors_set[author_id])

        return coauthors

    def generate_pairwise_links(self, links):

        pairwise_links = []
        # for id1 in range(len(links)):
        for id1 in links:
            for id2 in links[id1]:
                pairwise_links.append([id1, id2])
                # pairwise_links.append([id2, id1])
                # pairwise_links.append([id1, id1])
                # pairwise_links.append([id2, id2])
        pairwise_links = np.unique(pairwise_links, axis=0)

        return pairwise_links

    def split_data(self):

        print('Splitting data...')
        split_idx = int(self.num_documents * self.training_ratio)
        self.training_documents, self.test_documents = collections.defaultdict(list), collections.defaultdict(list)
        for doc_id in range(self.num_documents):
            if doc_id < split_idx:
                self.training_documents[doc_id] = self.documents[doc_id]
            else:
                self.test_documents[doc_id] = self.documents[doc_id]
        self.training_citations, self.test_citations, self.inference_citations = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for doc_id1 in self.citations:
            if doc_id1 < split_idx:
                training_citations_one_doc = []
                for doc_id2 in self.citations[doc_id1]:
                    if doc_id2 < split_idx:
                        training_citations_one_doc.append(doc_id2)
                self.training_citations[doc_id1] = training_citations_one_doc
            else:
                test_citations_one_doc, inference_citations_one_doc = [], []
                for doc_id2 in self.citations[doc_id1]:
                    if doc_id2 >= split_idx and doc_id1 != doc_id2:
                        test_citations_one_doc.append(doc_id2)
                    elif doc_id2 < split_idx or doc_id1 == doc_id2:
                        inference_citations_one_doc.append(doc_id2)
                if len(test_citations_one_doc) > 0:
                    self.test_citations[doc_id1] = test_citations_one_doc
                if len(inference_citations_one_doc) > 0:
                    self.inference_citations[doc_id1] = inference_citations_one_doc
        self.training_pairwise_citations = self.generate_pairwise_links(self.training_citations)
        self.test_pairwise_citations = self.generate_pairwise_links(self.test_citations)
        np.random.shuffle(self.training_pairwise_citations)
        self.num_training_citations, self.num_test_citations = len(self.training_pairwise_citations), len(self.test_pairwise_citations)
        if self.venues_available:
            self.training_venues, self.test_venues = self.venues[:split_idx], self.venues[split_idx:]
        if self.labels_available:
            self.training_labels, self.test_labels = self.labels[:split_idx], self.labels[split_idx:]
        self.num_training_documents, self.num_test_documents = len(self.training_documents), len(self.test_documents)

    def generate_word_word_semantic_links(self):

        cos_sim = cosine_similarity(self.word_embeddings, self.word_embeddings)
        self.word_word_semantic_links = collections.defaultdict(list)
        for word_id, row in enumerate(cos_sim):
            self.word_word_semantic_links[word_id] = np.argsort(row)[-self.word_word_graph_num_neighbors:]
        self.word_word_semantic_links, _ = self.symmatrize_links(self.word_word_semantic_links)

    def generate_word_word_pmi_and_syntactic_links(self):

        windows = []
        for doc_id in self.training_documents:
            doc_length = len(self.training_documents[doc_id])
            if doc_length <= self.word_word_graph_window_size:
                windows.append(self.training_documents[doc_id])
            else:
                for start_idx in range(doc_length - self.word_word_graph_window_size + 1):
                    window = self.training_documents[doc_id][start_idx:start_idx + self.word_word_graph_window_size]
                    windows.append(window)

        print('Generating word-word pmi links...')
        word_pair_counts = self.generate_word_word_pmi_links(windows)
        print('Generating word-word syntactic links...')
        self.generate_word_word_syntactic_links(windows, word_pair_counts)

    def generate_word_word_pmi_links(self, windows):

        word_counts = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_counts:
                    word_counts[window[i]] += 1
                else:
                    word_counts[window[i]] = 1
                appeared.add(window[i])

        word_pair_counts = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_j = window[j]
                    if word_i == word_j:
                        continue
                    word_pair_str = str(word_i) + ',' + str(word_j)
                    word_pair_str_reversed = str(word_j) + ',' + str(word_i)
                    if word_pair_str in word_pair_counts:
                        word_pair_counts[word_pair_str] += 1
                    elif word_pair_str_reversed in word_pair_counts:
                        word_pair_counts[word_pair_str_reversed] += 1
                    else:
                        word_pair_counts[word_pair_str] = 1

        pmi_sim = np.zeros([self.num_words, self.num_words])
        num_windows = len(windows)
        for word_pair_str in word_pair_counts:
            tmp = word_pair_str.split(',')
            word_i = int(tmp[0])
            word_j = int(tmp[1])
            word_pair_count = word_pair_counts[word_pair_str]
            word_count_i = word_counts[word_i]
            word_count_j = word_counts[word_j]
            pmi = np.log((1.0 * word_pair_count / num_windows) / (1.0 * word_count_i * word_count_j / (num_windows * num_windows)))
            if pmi <= 0:
                continue
            pmi_sim[word_i, word_j] = pmi
            pmi_sim[word_j, word_i] = pmi

        self.word_word_pmi_links = collections.defaultdict(list)
        for word_id, row in enumerate(pmi_sim):
            self.word_word_pmi_links[word_id] = np.argsort(row)[-self.word_word_graph_num_neighbors:]
        self.word_word_pmi_links, _ = self.symmatrize_links(self.word_word_pmi_links)

        return word_pair_counts

    def generate_word_word_syntactic_links(self, windows, word_pair_counts):

        syntactic_links_available = os.path.isfile('./data/' + self.dataset_name + '/word_word_syntactic_links.txt')
        if syntactic_links_available:
            self.word_word_syntactic_links = self.load_files('./data/' + self.dataset_name + '/word_word_syntactic_links.txt')
            return

        nlp = StanfordCoreNLP(r'./data/stanford-corenlp-full-2016-10-31', lang='en')
        rela_pair_count_str = {}
        for window in tqdm(windows):
            words = self.voc[window]
            words_str = ''
            for word in words:
                words_str += word
                words_str += ' '
            words_str = words_str.strip()
            words = nlp.word_tokenize(words_str)
            # window = window.replace(string.punctuation, ' ')
            res = nlp.dependency_parse(words_str)
            for tuple in res:
                if tuple[0] == 'ROOT':
                    continue
                pair = [words[tuple[1] - 1], words[tuple[2] - 1]]
                # rela.append(str(window[tuple[1] - 1]) + ', ' + str(window[tuple[2] - 1]))
                if pair[0] == pair[1]:
                    continue
                # if pair[0] in string.punctuation or pair[1] in string.punctuation:
                #     continue
                if pair[0] not in self.word2id or pair[1] not in self.word2id:
                    continue
                #word_pair_str = pair[0] + ',' + pair[1]
                word_pair_str = str(self.word2id[pair[0]]) + ',' + str(self.word2id[pair[1]])
                word_pair_str_reversed = str(self.word2id[pair[1]]) + ',' + str(self.word2id[pair[0]])
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                elif word_pair_str_reversed in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str_reversed] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
                # # two orders
                # word_pair_str = str(self.word2id[pair[1]]) + ',' + str(self.word2id[pair[0]])
                # if word_pair_str in rela_pair_count_str:
                #     rela_pair_count_str[word_pair_str] += 1
                # else:
                #     rela_pair_count_str[word_pair_str] = 1

        syntactic_sim = np.zeros([self.num_words, self.num_words])
        for word_pair_str in rela_pair_count_str:
            pair = word_pair_str.split(',')
            word_i = int(pair[0])
            word_j = int(pair[1])
            word_pair_count = rela_pair_count_str[word_pair_str]
            if word_pair_str not in word_pair_counts:
                continue
            syntactic_sim[word_i, word_j] = (1.0 * word_pair_count) / (1.0 * word_pair_counts[word_pair_str])
            syntactic_sim[word_j, word_i] = (1.0 * word_pair_count) / (1.0 * word_pair_counts[word_pair_str])

        self.word_word_syntactic_links = collections.defaultdict(list)
        for word_id, row in enumerate(syntactic_sim):
            self.word_word_syntactic_links[word_id] = np.argsort(row)[-self.word_word_graph_num_neighbors:]
        self.word_word_syntactic_links, _ = self.symmatrize_links(self.word_word_syntactic_links)
        with open('./data/' + self.dataset_name + '/word_word_syntactic_links.txt', 'w') as file:
            for word_id in range(self.num_words):
                words = self.word_word_syntactic_links[word_id]
                for w in words:
                    file.write(str(w))
                    file.write(' ')
                file.write('\n')
        with open('./data/' + self.dataset_name + '/word_word_syntactic_counts.txt', 'w') as file:
            for word_pair_str in rela_pair_count_str:
                pair = word_pair_str.split(',')
                word_i = int(pair[0])
                word_j = int(pair[1])
                word_pair_count = rela_pair_count_str[word_pair_str]
                file.write(str(word_i) + ',' + str(word_j))
                file.write('\t')
                file.write(str(word_pair_count))
                file.write('\n')

    def sample_neighbors(self):

        self.words_semantic_neighbors = self.sample_neighbors_func(self.word_word_semantic_links)
        self.words_pmi_neighbors = self.sample_neighbors_func(self.word_word_pmi_links)
        self.words_syntactic_neighbors = self.sample_neighbors_func(self.word_word_syntactic_links)
        self.documents_neighbors = self.sample_neighbors_func(self.documents)
        training_citations_neighbors = self.sample_neighbors_func(self.training_citations)
        inference_citations_neighbors = self.sample_neighbors_func(self.inference_citations)
        self.citations_neighbors = np.concatenate([training_citations_neighbors, inference_citations_neighbors], axis=0)
        self.authors_neighbors = self.sample_neighbors_func(self.authors)
        self.coauthors_neighbors = self.sample_neighbors_func(self.coauthors)
        if self.venues_available:
            self.venues_neighbors = np.tile(self.venues, [1, self.num_sampled_neighbors])
            self.covenues_neighbors = np.tile(self.covenues, [1, self.num_sampled_neighbors])

    def sample_neighbors_func(self, neighbors):

        sampled_neighbors = []
        keys = np.sort([key for key in neighbors.keys()])
        #for id in range(len(neighbors)):
        for id in keys:
            replace = len(neighbors[id]) < self.num_sampled_neighbors
            neighbor_idx = np.random.choice(len(neighbors[id]), size=self.num_sampled_neighbors, replace=replace)
            sampled_neighbors.append([neighbors[id][idx] for idx in neighbor_idx])
        sampled_neighbors = np.array(sampled_neighbors)

        return sampled_neighbors