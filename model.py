import tensorflow as tf
import numpy as np
import time
import collections
from tqdm import tqdm
from evaluation import *
from sklearn.metrics import accuracy_score
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Model():

    def __init__(self, args, data):

        self.data = data
        self.parse_args(args)
        self.show_config()
        self.generate_placeholders()
        self.generate_variables()

    def parse_args(self, args):

        self.dataset_name = args.dataset_name
        self.training_ratio = args.training_ratio
        self.num_documents = self.data.num_documents
        self.num_training_documents = self.data.num_training_documents
        self.num_test_documents = self.data.num_test_documents
        self.num_authors = self.data.num_authors
        if self.data.venues_available:
            self.num_venues = self.data.num_venues
        self.num_citations = self.data.num_citations
        self.num_training_citations = self.data.num_training_citations
        self.num_words = self.data.num_words
        if self.data.labels_available:
            self.num_labels = self.data.num_labels
        self.word_embedding_model = self.data.word_embedding_model
        self.word_embedding_dimension = self.data.word_embedding_dimension
        self.word_word_graph_window_size = args.word_word_graph_window_size
        self.word_word_graph_num_neighbors = args.word_word_graph_num_neighbors
        self.num_sampled_neighbors = args.num_sampled_neighbors
        self.num_negative_samples = args.num_negative_samples
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.minibatch_size = args.minibatch_size
        self.num_topics = args.num_topics
        self.supervision = args.supervision
        self.reg_sup = args.reg_sup
        self.num_convolutional_layers = args.num_convolutional_layers
        self.divergence = args.divergence
        self.prior = args.prior
        self.reg_divergence = args.reg_divergence
        self.reg_l2 = args.reg_l2
        self.author_prediction = args.author_prediction
        self.dropout_keep_prob = args.dropout_keep_prob

    def show_config(self):

        print('******************************************************')
        print('tf version:', tf.__version__)
        print('dataset name:', self.dataset_name)
        print('training ratio:', self.training_ratio)
        print('#total documents:', self.num_documents)
        print('#training documents:', self.num_training_documents)
        print('#authors:', self.num_authors)
        if self.data.venues_available:
            print('#venues:', self.num_venues)
        print('#citations:', self.num_citations)
        print('#training citations:', self.num_training_citations)
        print('#words:', self.num_words)
        if self.data.labels_available:
            print('#labels:', self.num_labels)
        print('word embedding model:', self.word_embedding_model)
        print('dimension of word embeddings:', self.word_embedding_dimension)
        print('word-word graph window size:', self.word_word_graph_window_size)
        print('word-word graph num of neighbors:', self.word_word_graph_num_neighbors)
        print('#sampled neighbors:', self.num_sampled_neighbors)
        print('#negative samples:', self.num_negative_samples)
        print('#epochs:', self.num_epochs)
        print('learning rate:', self.learning_rate)
        print('minibatch size:', self.minibatch_size)
        print('#topics:', self.num_topics)
        print('supervision:', self.supervision)
        print('#convolutional layers:', self.num_convolutional_layers)
        print('divergence:', self.divergence)
        print('prior:', self.prior)
        print('author prediction:', self.author_prediction)
        print('******************************************************')

    def generate_placeholders(self):

        self.placeholders = collections.defaultdict(list)
        self.placeholders['doc_ids_i'] = tf.placeholder(dtype=tf.int32, shape=[self.minibatch_size])
        self.placeholders['doc_ids_j'] = tf.placeholder(dtype=tf.int32, shape=[self.minibatch_size])
        self.placeholders['self_loop_mask'] = tf.placeholder(dtype=tf.bool, shape=[self.minibatch_size])
        self.placeholders['pretrained_word_embeddings'] = tf.placeholder(tf.float32, [self.num_words, self.word_embedding_dimension])
        self.placeholders['dropout_keep_prob'] = tf.placeholder(tf.float32)
        self.placeholders['author_ids_inference'] = tf.placeholder(dtype=tf.int32, shape=[self.minibatch_size])
        self.placeholders['word_ids_inference'] = tf.placeholder(dtype=tf.int32, shape=[self.minibatch_size])
        if self.data.labels_available:
            self.placeholders['labels_i'] = tf.placeholder(dtype=tf.int32, shape=[self.minibatch_size])
            self.placeholders['labels_j'] = tf.placeholder(dtype=tf.int32, shape=[self.minibatch_size])

    def generate_variables(self):

        self.variables = collections.defaultdict(list)
        self.convolution_dim = collections.defaultdict(list)
        self.convolution_dim['doc'] = [self.word_embedding_dimension] + [self.num_topics] * self.num_convolutional_layers
        self.convolution_dim['word_semantic'] = [self.word_embedding_dimension] + [self.num_topics] * self.num_convolutional_layers
        self.convolution_dim['word_pmi'] = [self.num_words] + [self.num_topics] * self.num_convolutional_layers
        self.convolution_dim['word_syntactic'] = [self.num_words] + [self.num_topics] * self.num_convolutional_layers
        self.convolution_dim['word'] = [self.num_words] + [self.num_topics] * self.num_convolutional_layers
        self.convolution_dim['author'] = [self.num_authors] + [self.num_topics] * self.num_convolutional_layers
        if self.data.venues_available:
            self.convolution_dim['venue'] = [self.num_venues] + [self.num_topics] * self.num_convolutional_layers

    def message_passing(self, self_embeds, neighbor_embeds, neighbor_embeds_types, layer_id, act=tf.nn.tanh, link_weights=None):

        with tf.variable_scope('graph_convolutional_layer', reuse=tf.AUTO_REUSE):
            for neighbor_embeds_type in neighbor_embeds_types:
                type = neighbor_embeds_type.split('_mean')[0]
                type = type.split('_logvar')[0]
                self.variables[neighbor_embeds_type + '_convolution_w_' + str(layer_id)] = tf.get_variable(
                    name=neighbor_embeds_type + '_convolution_w_' + str(layer_id),
                    shape=[self.convolution_dim[type][layer_id], self.convolution_dim[type][layer_id + 1]],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                self.variables[neighbor_embeds_type + '_convolution_b_' + str(layer_id)] = tf.get_variable(
                    name=neighbor_embeds_type + '_convolution_b_' + str(layer_id),
                    shape=[self.convolution_dim[type][layer_id + 1]],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

        # self_embeds = tf.nn.dropout(self_embeds, keep_prob=self.placeholders['dropout_keep_prob'])
        self_embeds = tf.matmul(self_embeds, self.variables[neighbor_embeds_types[0] + '_convolution_w_' + str(layer_id)]) #+ self.variables[neighbor_embeds_types[0] + '_convolution_b_' + str(layer_id)]
        self_embeds_agg1, neighbor_embeds_agg = [], []
        # intra-graph aggregation when len(neighbor_embeds) == 1
        for idx, embeds in enumerate(neighbor_embeds):
            embeds_dropout = tf.nn.dropout(embeds, keep_prob=self.placeholders['dropout_keep_prob'])
            embeds_proj = tf.matmul(embeds_dropout, self.variables[neighbor_embeds_types[idx] + '_convolution_w_' + str(layer_id)]) #+ self.variables[neighbor_embeds_types[idx] + '_convolution_b_' + str(layer_id)]
            attention_values = self.evaluate_attention(self_embeds, embeds_proj, neighbor_embeds_types[0], neighbor_embeds_types[idx], layer_id, link_weights)
            neighbor_embeds_agg_tmp, self_embeds_agg = self.neighbor_aggregation(self_embeds, embeds_proj, attention_values, act)
            self_embeds_agg1.append(self_embeds_agg)
            if idx == 0:
                continue
            neighbor_embeds_agg.append(neighbor_embeds_agg_tmp)
        # cross-graph aggregation
        if len(neighbor_embeds) > 1:
            neighbor_embeds_agg = tf.squeeze(tf.concat(neighbor_embeds_agg, axis=1))
            neighbor_embeds_agg = tf.reshape(neighbor_embeds_agg, [-1, tf.shape(self_embeds)[-1]])
            attention_values = self.evaluate_attention(self_embeds_agg1[0], neighbor_embeds_agg, 'cross', 'cross', layer_id)
            _, self_embeds_agg = self.neighbor_aggregation(self_embeds_agg1[0], neighbor_embeds_agg, attention_values, act='linear')

        return self_embeds_agg

    def evaluate_attention(self, self_embeds, neighbor_embeds, self_embeds_type, neighbor_embeds_type, layer_id, link_weights=None):

        with tf.variable_scope('att', reuse=tf.AUTO_REUSE):
            self.variables[self_embeds_type + '_' + neighbor_embeds_type + '_att_b_' + str(layer_id)] = tf.get_variable(
                name=self_embeds_type + '_' + neighbor_embeds_type + '_att_b_' + str(layer_id),
                shape=[2 * self.convolution_dim['doc'][layer_id + 1], 1],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        num_neighbors = tf.cast(tf.divide(tf.shape(neighbor_embeds)[0], tf.shape(self_embeds)[0]), tf.int32)
        embeds_repeat = tf.reshape(tf.tile(tf.expand_dims(self_embeds, axis=1), [1, num_neighbors, 1]), tf.shape(neighbor_embeds))
        attention_values = tf.matmul(tf.concat([embeds_repeat, neighbor_embeds], axis=1), self.variables[self_embeds_type + '_' + neighbor_embeds_type + '_att_b_' + str(layer_id)])
        attention_values = tf.reshape(attention_values, [-1, num_neighbors])
        if link_weights == None:
            attention_values = tf.nn.softmax(tf.nn.leaky_relu(attention_values))
        else:
            link_weights = tf.reshape(link_weights, [-1, num_neighbors])
            attention_values = tf.nn.softmax(tf.nn.leaky_relu(tf.multiply(attention_values, link_weights)))

        return attention_values

    def neighbor_aggregation(self, self_embeds, neighbor_embeds, attention_values, act=tf.nn.tanh):

        attention_values = tf.expand_dims(attention_values, axis=1)
        neighbor_embeds = tf.reshape(neighbor_embeds, [tf.shape(attention_values)[0], tf.shape(attention_values)[-1], tf.shape(neighbor_embeds)[-1]])
        neighbor_embeds_agg = tf.squeeze(tf.matmul(attention_values, neighbor_embeds))
        if act == 'linear':
            eta = 0.9
            self_embeds_agg = eta * self_embeds + (1 - eta) * neighbor_embeds_agg
        else:
            eta = 0.5
            self_embeds_agg = act(eta * self_embeds + (1 - eta) * neighbor_embeds_agg)
            neighbor_embeds_agg = act(neighbor_embeds_agg)
            # embeds_agg = neighbor_embeds_agg
        return neighbor_embeds_agg, self_embeds_agg

    def get_neighbors(self, doc_ids):

        neighbors = collections.defaultdict(list)
        neighbors['doc_ids'] = [doc_ids]
        neighbors['author_ids'] = [tf.reshape(tf.gather(self.data.authors_neighbors, doc_ids), [-1])]
        neighbors['word_ids_pmi'] = [tf.reshape(tf.gather(self.data.documents_neighbors, doc_ids), [-1])]
        neighbors['word_ids_syntactic'] = [tf.reshape(tf.gather(self.data.documents_neighbors, doc_ids), [-1])]
        if self.data.venues_available:
            neighbors['venue_ids'] = [tf.reshape(tf.gather(self.data.venues_neighbors, doc_ids), [-1])]
        for layer_id in range(self.num_convolutional_layers):
            neighbor_doc_ids = tf.reshape(tf.gather(self.data.citations_neighbors, neighbors['doc_ids'][layer_id]), [-1])
            neighbors['doc_ids'].append(neighbor_doc_ids)
            neighbor_author_ids = tf.reshape(tf.gather(self.data.coauthors_neighbors, neighbors['author_ids'][layer_id]), [-1])
            neighbors['author_ids'].append(neighbor_author_ids)
            neighbor_word_ids_pmi = tf.reshape(tf.gather(self.data.words_pmi_neighbors, neighbors['word_ids_pmi'][layer_id]), [-1])
            neighbors['word_ids_pmi'].append(neighbor_word_ids_pmi)
            neighbor_word_ids_syntactic = tf.reshape(tf.gather(self.data.words_syntactic_neighbors, neighbors['word_ids_syntactic'][layer_id]), [-1])
            neighbors['word_ids_syntactic'].append(neighbor_word_ids_syntactic)
            if self.data.venues_available:
                neighbor_venue_ids = tf.reshape(tf.gather(self.data.covenues_neighbors, neighbors['venue_ids'][layer_id]), [-1])
                neighbors['venue_ids'].append(neighbor_venue_ids)

        return neighbors

    def evaluate_word_embeds_semantic(self):

        with tf.variable_scope('topic', reuse=tf.AUTO_REUSE):
            self.variables['topic_embeddings'] = tf.get_variable(
                name='topic_embeddings',
                shape=[self.num_topics, self.word_embedding_dimension],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        word_embeddings_norm = tf.nn.l2_normalize(self.placeholders['pretrained_word_embeddings'], axis=-1)
        topic_embeddings_norm = tf.nn.l2_normalize(self.variables['topic_embeddings'], axis=-1)
        self.word_embeds_semantic = 1 - tf.matmul(word_embeddings_norm, tf.transpose(topic_embeddings_norm))  # [num_words, num_topics]

    def variational_regularization(self, z_mean, z_logvar):

        if self.divergence == 'kl' and (self.prior == 'normal' or self.prior == 'gaussian'):
            eps = tf.random_normal(tf.shape(z_logvar), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
            z = z_mean + tf.exp(z_logvar / 2) * eps
            kl = 1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
            kl = - 0.5 * tf.reduce_sum(kl)
            divergence = kl
        elif self.divergence == 'kl' and self.prior == 'dirichlet':
            dirchlet_alpha = 1 * np.ones(self.num_topics).astype(np.float32)
            mean_prior = tf.constant(np.log(dirchlet_alpha) - np.mean(np.log(dirchlet_alpha)), dtype=tf.float32)
            mean_prior = tf.tile(tf.expand_dims(mean_prior, axis=0), [tf.shape(z_mean)[0], 1])
            var_prior = tf.constant((1.0 / dirchlet_alpha) * (1 - 2.0 / self.num_topics) + (1.0 / (self.num_topics * self.num_topics)) * np.sum(1.0 / dirchlet_alpha), dtype=tf.float32)
            var_prior = tf.tile(tf.expand_dims(var_prior, axis=0), [tf.shape(z_mean)[0], 1])
            eps = tf.random_normal(tf.shape(z_logvar), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
            z = z_mean + tf.exp(z_logvar / 2) * eps
            kl = 1 * (tf.reduce_sum(tf.div(tf.exp(z_logvar), var_prior), axis=-1) +
                      tf.reduce_sum(tf.multiply(tf.div(mean_prior - z_mean, var_prior), (mean_prior - z_mean)), axis=-1) - self.num_topics +
                      tf.reduce_sum(tf.log(var_prior), axis=-1) - tf.reduce_sum(z_logvar, axis=-1))
            kl = 0.5 * tf.reduce_sum(kl)
            divergence = kl
        elif self.divergence == 'wasserstein' and (self.prior == 'gaussian' or self.prior == 'normal'):
            eps = tf.random_normal(tf.shape(z_logvar), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
            z = z_mean + tf.exp(z_logvar / 2) * eps
            wasserstein = tf.reduce_sum(tf.square(z_mean) + tf.square(tf.exp(z_logvar / 2) - 1))
            divergence = wasserstein

        return z, divergence

    def convolutional_layer(self, layer_id, hop, doc_embeds, word_embeds_semantic, word_embeds_pmi, word_embeds_syntactic, author_embeds, venue_embeds=None, mean_or_logvar='', act=tf.nn.tanh, dropout_keep_prob=1):

        if mean_or_logvar == '_mean' or mean_or_logvar == '_logvar':
            if self.num_convolutional_layers == 1:
                word_embeds = tf.nn.dropout(word_embeds_pmi[0], keep_prob=dropout_keep_prob)
            else:
                word_embeds = (word_embeds_semantic[0] + word_embeds_pmi[0] + word_embeds_syntactic[0]) / 3.0
            doc_neighbor_embeds = [tf.nn.dropout(doc_embeds[hop + 1], keep_prob=dropout_keep_prob), tf.nn.dropout(word_embeds, keep_prob=dropout_keep_prob), tf.nn.dropout(author_embeds[0], keep_prob=dropout_keep_prob)]
            doc_neighbor_embeds_types = ['doc' + mean_or_logvar, 'word' + mean_or_logvar, 'author' + mean_or_logvar]
            if self.data.venues_available:
                doc_neighbor_embeds += [tf.nn.dropout(venue_embeds[0], keep_prob=dropout_keep_prob)]
                doc_neighbor_embeds_types += ['venue' + mean_or_logvar]
        else:
            doc_neighbor_embeds = [doc_embeds[hop + 1]]
            doc_neighbor_embeds_types = ['doc']

        doc_embeds_agg = self.message_passing(self_embeds=tf.nn.dropout(doc_embeds[hop], keep_prob=dropout_keep_prob),
                                              neighbor_embeds=doc_neighbor_embeds,
                                              neighbor_embeds_types=doc_neighbor_embeds_types, layer_id=layer_id, act=act)
        word_embeds_pmi_agg = self.message_passing(self_embeds=tf.nn.dropout(word_embeds_pmi[hop], keep_prob=dropout_keep_prob),
                                                   neighbor_embeds=[word_embeds_pmi[hop + 1]],
                                                   neighbor_embeds_types=['word' + mean_or_logvar], layer_id=layer_id, act=act)
        word_embeds_syntactic_agg = self.message_passing(self_embeds=tf.nn.dropout(word_embeds_syntactic[hop], keep_prob=dropout_keep_prob),
                                                         neighbor_embeds=[word_embeds_syntactic[hop + 1]],
                                                         neighbor_embeds_types=['word' + mean_or_logvar], layer_id=layer_id, act=act)
        author_embeds_agg = self.message_passing(self_embeds=tf.nn.dropout(author_embeds[hop], keep_prob=dropout_keep_prob),
                                                 neighbor_embeds=[author_embeds[hop + 1]],
                                                 neighbor_embeds_types=['author' + mean_or_logvar], layer_id=layer_id, act=act)
        if self.data.venues_available:
            venue_embeds_agg = self.message_passing(self_embeds=tf.nn.dropout(venue_embeds[hop], keep_prob=dropout_keep_prob),
                                                    neighbor_embeds=[venue_embeds[hop + 1]],
                                                    neighbor_embeds_types=['venue' + mean_or_logvar], layer_id=layer_id, act=act)
            return doc_embeds_agg, word_embeds_pmi_agg, word_embeds_syntactic_agg, author_embeds_agg, venue_embeds_agg

        return doc_embeds_agg, word_embeds_pmi_agg, word_embeds_syntactic_agg, author_embeds_agg

    def graph_convolution(self, doc_ids):

        neighbors = self.get_neighbors(doc_ids)

        doc_embeds = [tf.nn.embedding_lookup(self.doc_features, indices) for indices in neighbors['doc_ids']]
        # word_embeds_semantic = [tf.gather(self.placeholders['pretrained_word_embeddings'], indices) for indices in words_semantic]
        word_embeds_semantic = [tf.gather(self.word_embeds_semantic, neighbors['word_ids_pmi'][0])]
        word_embeds_pmi = [tf.one_hot(indices, self.num_words) for indices in neighbors['word_ids_pmi']]
        word_embeds_syntactic = [tf.one_hot(indices, self.num_words) for indices in neighbors['word_ids_syntactic']]
        #author_embeds = [tf.nn.embedding_lookup(self.author_features, indices) for indices in neighbors['author_ids']]
        author_embeds = [tf.one_hot(indices, self.num_authors) for indices in neighbors['author_ids']]
        if self.data.venues_available:
            venue_embeds = [tf.one_hot(indices, self.num_venues) for indices in neighbors['venue_ids']]

        z, z_mean, divergence = collections.defaultdict(list), collections.defaultdict(list), 0
        for layer_id in range(self.num_convolutional_layers):
            next_doc_embeds, next_word_embeds_semantic, next_word_embeds_pmi, next_word_embeds_syntactic, next_word_embeds, next_author_embeds, next_venue_embeds = [], [], [], [], [], [], []
            for hop in range(self.num_convolutional_layers - layer_id):
                if layer_id == self.num_convolutional_layers - 1:
                    act = lambda x: x
                    dropout_keep_prob = self.placeholders['dropout_keep_prob']
                    if self.data.venues_available:
                        doc_embeds_agg_mean, word_embeds_pmi_agg_mean, word_embeds_syntactic_agg_mean, author_embeds_agg_mean, venue_embeds_agg_mean = \
                            self.convolutional_layer(layer_id, hop, doc_embeds, word_embeds_semantic, word_embeds_pmi, word_embeds_syntactic, author_embeds, venue_embeds, mean_or_logvar='_mean', act=act, dropout_keep_prob=dropout_keep_prob)
                        doc_embeds_agg_logvar, word_embeds_pmi_agg_logvar, word_embeds_syntactic_agg_logvar, author_embeds_agg_logvar, venue_embeds_agg_logvar = \
                            self.convolutional_layer(layer_id, hop, doc_embeds, word_embeds_semantic, word_embeds_pmi, word_embeds_syntactic, author_embeds, venue_embeds, mean_or_logvar='_logvar', act=act, dropout_keep_prob=dropout_keep_prob)
                        venue_embeds_agg, venue_divergence = self.variational_regularization(venue_embeds_agg_mean, venue_embeds_agg_logvar)
                        #next_venue_embeds.append(venue_embeds_agg)
                    else:
                        doc_embeds_agg_mean, word_embeds_pmi_agg_mean, word_embeds_syntactic_agg_mean, author_embeds_agg_mean = \
                            self.convolutional_layer(layer_id, hop, doc_embeds, word_embeds_semantic, word_embeds_pmi, word_embeds_syntactic, author_embeds, mean_or_logvar='_mean', act=act, dropout_keep_prob=dropout_keep_prob)
                        doc_embeds_agg_logvar, word_embeds_pmi_agg_logvar, word_embeds_syntactic_agg_logvar, author_embeds_agg_logvar = \
                            self.convolutional_layer(layer_id, hop, doc_embeds, word_embeds_semantic, word_embeds_pmi, word_embeds_syntactic, author_embeds, mean_or_logvar='_logvar', act=act, dropout_keep_prob=dropout_keep_prob)
                    doc_embeds_agg, doc_divergence = self.variational_regularization(doc_embeds_agg_mean, doc_embeds_agg_logvar)
                    #next_doc_embeds.append(doc_embeds_agg)
                    z['doc_embeds'] = tf.nn.dropout(doc_embeds_agg, keep_prob=self.placeholders['dropout_keep_prob'])
                    z_mean['doc_embeds'] = doc_embeds_agg_mean
                    word_embeds_pmi_agg, word_pmi_divergence = self.variational_regularization(word_embeds_pmi_agg_mean, word_embeds_pmi_agg_logvar)
                    #next_word_embeds_pmi.append(word_embeds_pmi_agg)
                    z['word_embeds_pmi'] = tf.nn.dropout(word_embeds_pmi_agg, keep_prob=self.placeholders['dropout_keep_prob'])
                    z_mean['word_embeds_pmi'] = word_embeds_pmi_agg_mean
                    word_embeds_syntactic_agg, word_syntactic_divergence = self.variational_regularization(word_embeds_syntactic_agg_mean, word_embeds_syntactic_agg_logvar)
                    #next_word_embeds_syntactic.append(word_embeds_syntactic_agg)
                    z['word_embeds_syntactic'] = tf.nn.dropout(word_embeds_syntactic_agg, keep_prob=self.placeholders['dropout_keep_prob'])
                    z_mean['word_embeds_syntactic'] = word_embeds_syntactic_agg_mean
                    z['word_embeds'] = (word_embeds_semantic[0] + word_embeds_pmi_agg + word_embeds_syntactic_agg) / 3.0
                    z_mean['word_embeds'] = (word_embeds_semantic[0] + word_embeds_pmi_agg_mean + word_embeds_syntactic_agg_mean) / 3.0
                    author_embeds_agg, author_divergence = self.variational_regularization(author_embeds_agg_mean, author_embeds_agg_logvar)
                    #next_author_embeds.append(author_embeds_agg)
                    z['author_embeds'] = tf.nn.dropout(author_embeds_agg, keep_prob=self.placeholders['dropout_keep_prob'])
                    z_mean['author_embeds'] = author_embeds_agg_mean
                    divergence += (doc_divergence + word_pmi_divergence + word_syntactic_divergence + author_divergence)
                    if self.data.venues_available:
                        z['venue_embeds'] = tf.nn.dropout(venue_embeds_agg, keep_prob=self.placeholders['dropout_keep_prob'])
                        z_mean['venue_embeds'] = venue_embeds_agg_mean
                        divergence += venue_divergence
                else:
                    if self.data.venues_available:
                        doc_embeds_agg, word_embeds_pmi_agg, word_embeds_syntactic_agg, author_embeds_agg, venue_embeds_agg = \
                            self.convolutional_layer(layer_id, hop, doc_embeds, word_embeds_semantic, word_embeds_pmi, word_embeds_syntactic, author_embeds, venue_embeds)
                        next_venue_embeds.append(venue_embeds_agg)
                    else:
                        doc_embeds_agg, word_embeds_pmi_agg, word_embeds_syntactic_agg, author_embeds_agg = \
                            self.convolutional_layer(layer_id, hop, doc_embeds, word_embeds_semantic, word_embeds_pmi, word_embeds_syntactic, author_embeds)
                    next_doc_embeds.append(doc_embeds_agg)
                    next_word_embeds_pmi.append(word_embeds_pmi_agg)
                    next_word_embeds_syntactic.append(word_embeds_syntactic_agg)
                    next_author_embeds.append(author_embeds_agg)
            doc_embeds = next_doc_embeds
            # word_embeds_semantic = next_word_embeds_semantic
            word_embeds_pmi = next_word_embeds_pmi
            word_embeds_syntactic = next_word_embeds_syntactic
            # word_embeds = next_word_embeds
            author_embeds = next_author_embeds
            if self.data.venues_available:
                venue_embeds = next_venue_embeds

        return z, z_mean, divergence

    def loss_function(self, embeds_i, embeds_j, neg_embeds, true_weights=None):

        embeds_i = tf.reshape(tf.tile(tf.expand_dims(embeds_i, axis=1), [1, tf.div(tf.shape(embeds_j)[0], tf.shape(embeds_i)[0]), 1]), tf.shape(embeds_j))
        pos_logits = tf.reduce_sum(tf.multiply(embeds_i, embeds_j), axis=1)
        neg_indices = tf.random_uniform(shape=[self.num_negative_samples * tf.shape(embeds_i)[0]], maxval=tf.shape(neg_embeds)[0], dtype=tf.int32)
        sampled_neg_embeds = tf.gather(neg_embeds, neg_indices)
        # sampled_neg_embeds = neg_embeds
        embeds_i_repeat = tf.reshape(tf.tile(tf.expand_dims(embeds_i, axis=1), [1, self.num_negative_samples, 1]), tf.shape(sampled_neg_embeds))
        neg_logits = tf.reduce_sum(tf.multiply(embeds_i_repeat, sampled_neg_embeds), axis=1)
        if true_weights == None:
            pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_logits), logits=pos_logits)
        else:
            pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_weights, logits=pos_logits)
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_logits), logits=neg_logits)
        loss = tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)

        return loss

    def classification_loss(self, embeds, labels):

        with tf.variable_scope('clf', reuse=tf.AUTO_REUSE):
            self.variables['weights'] = tf.get_variable(
                name='weights',
                shape=[self.convolution_dim['doc'][-1], self.num_labels],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            self.variables['bias'] = tf.get_variable(
                name='bias',
                shape=[self.num_labels],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        logits = tf.matmul(embeds, self.variables['weights']) + self.variables['bias']
        y_pred = tf.argmax(logits, axis=1)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(labels, self.num_labels)))

        return loss, y_pred

    def l2_loss(self):

        types = ['doc', 'word', 'author', 'doc_mean', 'doc_logvar', 'word_mean', 'word_logvar', 'author_mean', 'author_logvar']
        if self.data.venues_available:
            types += ['venue', 'venue_mean', 'venue_logvar']
        l2_loss = 0
        for type in types:
            for layer_id in range(self.num_convolutional_layers):
                l2_loss += tf.nn.l2_loss(self.variables[type + '_convolution_' + str(layer_id)])

        return l2_loss

    def construct_model(self):

        self.doc_features = tf.Variable(tf.constant(self.data.doc_contents_word_embed, dtype=tf.float32), trainable=False)
        self.evaluate_word_embeds_semantic()

        z_i, z_mean_i, divergence_i = self.graph_convolution(self.placeholders['doc_ids_i'])
        z_j, z_mean_j, divergence_j = self.graph_convolution(self.placeholders['doc_ids_j'])
        doc_ids_neg = tf.random_uniform(shape=[self.minibatch_size], maxval=self.num_training_documents, dtype=tf.int32)
        z_neg, z_mean_neg, divergence_neg = self.graph_convolution(doc_ids_neg)
        self.doc_embeds_mean_i = z_mean_i['doc_embeds']

        doc_embeds_i_mask = tf.boolean_mask(z_i['doc_embeds'], self.placeholders['self_loop_mask'])
        doc_embeds_j_mask = tf.boolean_mask(z_j['doc_embeds'], self.placeholders['self_loop_mask'])
        # doc_doc_loss = self.loss_function(doc_embeds_i_mask, doc_embeds_j_mask, tf.concat([z_i['doc_embeds'], z_j['doc_embeds']], axis=0))
        # doc_word_loss_i = self.loss_function(z_i['doc_embeds'], z_i['word_embeds'], tf.concat([z_i['word_embeds'], z_j['word_embeds']], axis=0))
        # doc_word_loss_j = self.loss_function(z_j['doc_embeds'], z_j['word_embeds'], tf.concat([z_i['word_embeds'], z_j['word_embeds']], axis=0))
        # doc_author_loss_i = self.loss_function(z_i['doc_embeds'], z_i['author_embeds'], tf.concat([z_i['author_embeds'], z_j['author_embeds']], axis=0))
        # doc_author_loss_j = self.loss_function(z_j['doc_embeds'], z_j['author_embeds'], tf.concat([z_i['author_embeds'], z_j['author_embeds']], axis=0))
        doc_doc_loss = self.loss_function(doc_embeds_i_mask, doc_embeds_j_mask, z_neg['doc_embeds'])
        doc_word_loss_i = self.loss_function(z_i['doc_embeds'], z_i['word_embeds'], z_neg['word_embeds'])
        doc_word_loss_j = self.loss_function(z_j['doc_embeds'], z_j['word_embeds'], z_neg['word_embeds'])
        doc_author_loss_i = self.loss_function(z_i['doc_embeds'], z_i['author_embeds'], z_neg['author_embeds'])
        doc_author_loss_j = self.loss_function(z_j['doc_embeds'], z_j['author_embeds'], z_neg['author_embeds'])
        loss_reconstruction = doc_doc_loss + 1 * (doc_word_loss_i + doc_word_loss_j) + 1 * (doc_author_loss_i + doc_author_loss_j)
        loss = loss_reconstruction + self.reg_divergence * (divergence_i + divergence_j) + self.reg_l2 * self.l2_loss()
        if self.data.venues_available:
            doc_venue_loss_i = self.loss_function(z_i['doc_embeds'], z_i['venue_embeds'], z_neg['venue_embeds'])
            doc_venue_loss_j = self.loss_function(z_j['doc_embeds'], z_j['venue_embeds'], z_neg['venue_embeds'])
            loss += (doc_venue_loss_i + doc_venue_loss_j)
        if self.supervision:
            doc_clf_loss_i, self.y_pred = self.classification_loss(z_i['doc_embeds'], self.placeholders['labels_i'])
            doc_clf_loss_j, _ = self.classification_loss(z_j['doc_embeds'], self.placeholders['labels_j'])
            loss += self.reg_sup * (doc_clf_loss_i + doc_clf_loss_j)

        return loss, loss_reconstruction

    def prepare_feed_dict(self, feed_dict, minibatch_idx, pairwise_citations):

        sampled_citations = pairwise_citations[minibatch_idx * self.minibatch_size:(minibatch_idx + 1) * self.minibatch_size]
        if len(sampled_citations) < self.minibatch_size:
            replace_indices = np.random.choice(len(pairwise_citations),
                                               self.minibatch_size - len(sampled_citations),
                                               replace=len(pairwise_citations) < self.minibatch_size)
            sampled_citations = np.concatenate([sampled_citations, pairwise_citations[replace_indices]], axis=0)

        feed_dict[self.placeholders['self_loop_mask']] = [citation[0] != citation[1] for citation in sampled_citations]
        feed_dict[self.placeholders['doc_ids_i']] = sampled_citations[:, 0]
        feed_dict[self.placeholders['doc_ids_j']] = sampled_citations[:, 1]
        feed_dict[self.placeholders['dropout_keep_prob']] = self.dropout_keep_prob
        feed_dict[self.placeholders['pretrained_word_embeddings']] = self.data.word_embeddings
        if self.supervision:
            feed_dict[self.placeholders['labels_i']] = self.data.labels[sampled_citations[:, 0]]
            feed_dict[self.placeholders['labels_j']] = self.data.labels[sampled_citations[:, 1]]

        return feed_dict

    def infer_author_embeds(self, sess):

        # get neighbors
        neighbors = collections.defaultdict(list)
        neighbors['author_ids'] = [self.placeholders['author_ids_inference']]
        for layer_id in range(self.num_convolutional_layers):
            neighbor_author_ids = tf.reshape(tf.gather(self.data.coauthors_neighbors, neighbors['author_ids'][layer_id]), [-1])
            neighbors['author_ids'].append(neighbor_author_ids)

        # graph convolution
        author_embeds = [tf.one_hot(indices, self.num_authors) for indices in neighbors['author_ids']]
        z, z_mean = collections.defaultdict(list), collections.defaultdict(list)
        for layer_id in range(self.num_convolutional_layers):
            next_author_embeds = []
            for hop in range(self.num_convolutional_layers - layer_id):
                if layer_id == self.num_convolutional_layers - 1:
                    act = lambda x: x
                    #act = tf.nn.tanh
                    author_embeds_agg_mean = self.message_passing(self_embeds=author_embeds[hop],
                                                                  neighbor_embeds=[author_embeds[hop + 1]],
                                                                  neighbor_embeds_types=['author_mean'], layer_id=layer_id, act=act)
                    author_embeds_agg_logvar = self.message_passing(self_embeds=author_embeds[hop],
                                                                    neighbor_embeds=[author_embeds[hop + 1]],
                                                                    neighbor_embeds_types=['author_logvar'], layer_id=layer_id, act=act)
                    author_embeds_agg, _ = self.variational_regularization(author_embeds_agg_mean, author_embeds_agg_logvar)
                    z['author_embeds'] = author_embeds_agg
                    z_mean['author_embeds'] = author_embeds_agg_mean
                else:
                    author_embeds_agg = self.message_passing(self_embeds=author_embeds[hop],
                                                             neighbor_embeds=[author_embeds[hop + 1]],
                                                             neighbor_embeds_types=['author'], layer_id=layer_id)
                    next_author_embeds.append(author_embeds_agg)
            author_embeds = next_author_embeds

        # infer author_embeds
        author_topic_dist = []
        num_minibatch_eval = int(np.ceil(self.num_authors / self.minibatch_size))
        for minibatch_idx in range(num_minibatch_eval):
            feed_dict = {}
            sampled_author_ids = np.arange(self.num_authors)[minibatch_idx * self.minibatch_size:(minibatch_idx + 1) * self.minibatch_size]
            if len(sampled_author_ids) < self.minibatch_size:
                replace_indices = np.random.choice(self.num_authors,
                                                   self.minibatch_size - len(sampled_author_ids),
                                                   replace=self.num_authors < self.minibatch_size)
                sampled_author_ids = np.concatenate([sampled_author_ids, replace_indices], axis=0)
            feed_dict[self.placeholders['author_ids_inference']] = sampled_author_ids
            feed_dict[self.placeholders['dropout_keep_prob']] = 1

            author_topic_dist.extend(sess.run(z_mean['author_embeds'], feed_dict=feed_dict))
        author_topic_dist = np.array(author_topic_dist)[:self.num_authors]

        return author_topic_dist

    def infer_word_embeds(self, sess):

        # get neighbors
        neighbors = collections.defaultdict(list)
        neighbors['word_ids_pmi'] = [self.placeholders['word_ids_inference']]
        neighbors['word_ids_syntactic'] = [self.placeholders['word_ids_inference']]
        word_embeds_semantic = tf.gather(self.word_embeds_semantic, self.placeholders['word_ids_inference'])
        for layer_id in range(self.num_convolutional_layers):
            neighbor_word_ids_pmi = tf.reshape(tf.gather(self.data.words_pmi_neighbors, neighbors['word_ids_pmi'][layer_id]), [-1])
            neighbors['word_ids_pmi'].append(neighbor_word_ids_pmi)
            neighbor_word_ids_syntactic = tf.reshape(tf.gather(self.data.words_syntactic_neighbors, neighbors['word_ids_syntactic'][layer_id]), [-1])
            neighbors['word_ids_syntactic'].append(neighbor_word_ids_syntactic)

        # graph convolution
        word_embeds_pmi = [tf.one_hot(indices, self.num_words) for indices in neighbors['word_ids_pmi']]
        word_embeds_syntactic = [tf.one_hot(indices, self.num_words) for indices in neighbors['word_ids_syntactic']]
        z, z_mean = collections.defaultdict(list), collections.defaultdict(list)
        for layer_id in range(self.num_convolutional_layers):
            next_word_embeds_pmi, next_word_embeds_syntactic = [], []
            for hop in range(self.num_convolutional_layers - layer_id):
                if layer_id == self.num_convolutional_layers - 1:
                    act = lambda x: x
                    #act = tf.nn.tanh
                    word_embeds_pmi_agg_mean = self.message_passing(self_embeds=word_embeds_pmi[hop],
                                                                    neighbor_embeds=[word_embeds_pmi[hop + 1]],
                                                                    neighbor_embeds_types=['word_mean'],
                                                                    layer_id=layer_id, act=act)
                    word_embeds_syntactic_agg_mean = self.message_passing(self_embeds=word_embeds_syntactic[hop],
                                                                          neighbor_embeds=[word_embeds_syntactic[hop + 1]],
                                                                          neighbor_embeds_types=['word_mean'],
                                                                          layer_id=layer_id, act=act)
                    z_mean['word_embeds'] = (word_embeds_pmi_agg_mean + word_embeds_syntactic_agg_mean + word_embeds_semantic) / 3.0
                else:
                    word_embeds_pmi_agg = self.message_passing(self_embeds=word_embeds_pmi[hop],
                                                               neighbor_embeds=[word_embeds_pmi[hop + 1]],
                                                               neighbor_embeds_types=['word'], layer_id=layer_id)
                    word_embeds_syntactic_agg = self.message_passing(self_embeds=word_embeds_syntactic[hop],
                                                                     neighbor_embeds=[word_embeds_syntactic[hop + 1]],
                                                                     neighbor_embeds_types=['word'], layer_id=layer_id)
                    next_word_embeds_pmi.append(word_embeds_pmi_agg)
                    next_word_embeds_syntactic.append(word_embeds_syntactic_agg)
            word_embeds_pmi = next_word_embeds_pmi
            word_embeds_syntactic = next_word_embeds_syntactic

        # infer word_embeds
        word_embeds = []
        num_minibatch_eval = int(np.ceil(self.num_words / self.minibatch_size))
        for minibatch_idx in range(num_minibatch_eval):
            feed_dict = {}
            sampled_word_ids = np.arange(self.num_words)[minibatch_idx * self.minibatch_size:(minibatch_idx + 1) * self.minibatch_size]
            if len(sampled_word_ids) < self.minibatch_size:
                replace_indices = np.random.choice(self.num_words,
                                                   self.minibatch_size - len(sampled_word_ids),
                                                   replace=self.num_words < self.minibatch_size)
                sampled_word_ids = np.concatenate([sampled_word_ids, replace_indices], axis=0)
            feed_dict[self.placeholders['word_ids_inference']] = sampled_word_ids
            feed_dict[self.placeholders['dropout_keep_prob']] = 1
            feed_dict[self.placeholders['pretrained_word_embeddings']] = self.data.word_embeddings

            word_embeds.extend(sess.run(z_mean['word_embeds'], feed_dict=feed_dict))
        word_embeds = np.array(word_embeds)[:self.num_words]

        return word_embeds

    def train(self):

        loss, loss_reconstruction = self.construct_model()
        optimizer_loss = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        optimizer_loss_reconstruction = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_reconstruction)
        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init)
            num_minibatch = int(np.ceil(self.num_training_citations / self.minibatch_size))
            t = time.time()
            one_epoch_loss = 0
            for epoch_idx in range(1, self.num_epochs + 1):
                np.random.shuffle(self.data.training_pairwise_citations)
                self.data.sample_neighbors()
                for minibatch_idx in tqdm(range(num_minibatch)):
                    feed_dict = {}
                    feed_dict = self.prepare_feed_dict(feed_dict, minibatch_idx, self.data.training_pairwise_citations)
                    num_sub_epochs = 0
                    for sub_epoch_idx in range(num_sub_epochs):
                        _, one_epoch_loss = sess.run([optimizer_loss_reconstruction, loss_reconstruction], feed_dict=feed_dict)
                    _, one_epoch_loss = sess.run([optimizer_loss, loss], feed_dict=feed_dict)
                if epoch_idx % 1 == 0 or epoch_idx == 1:
                    print('******************************************************')
                    print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_idx, self.num_epochs), '\tLoss: %f' % one_epoch_loss)

                    # infer training doc embeds
                    doc_topic_dist_training = []
                    num_minibatch_eval = int(np.ceil(self.num_training_documents / self.minibatch_size))
                    pairwise_citations_eval = np.array([[doc_id, doc_id] for doc_id in range(self.num_training_documents)])
                    for minibatch_idx in range(num_minibatch_eval):
                        feed_dict = {}
                        feed_dict = self.prepare_feed_dict(feed_dict, minibatch_idx, pairwise_citations_eval)
                        feed_dict.update({self.placeholders['dropout_keep_prob']: 1})
                        doc_topic_dist_training.extend(sess.run(self.doc_embeds_mean_i, feed_dict=feed_dict))

                    # infer test doc embeds
                    doc_topic_dist_test, pred_labels = [], []
                    num_minibatch_eval = int(np.ceil(self.num_test_documents / self.minibatch_size))
                    pairwise_citations_eval = np.array([[doc_id + self.num_training_documents, doc_id + self.num_training_documents] for doc_id in range(self.num_test_documents)])
                    for minibatch_idx in range(num_minibatch_eval):
                        feed_dict = {}
                        feed_dict = self.prepare_feed_dict(feed_dict, minibatch_idx, pairwise_citations_eval)
                        feed_dict.update({self.placeholders['dropout_keep_prob']: 1})
                        if self.supervision:
                            doc_topic_dist_test_tmp, y_pred_tmp = sess.run([self.doc_embeds_mean_i, self.y_pred], feed_dict=feed_dict)
                            doc_topic_dist_test.extend(doc_topic_dist_test_tmp)
                            pred_labels.extend(y_pred_tmp)
                        else:
                            doc_topic_dist_test.extend(sess.run(self.doc_embeds_mean_i, feed_dict=feed_dict))
                    doc_topic_dist_training = np.array(doc_topic_dist_training)[:self.num_training_documents]
                    doc_topic_dist_test = np.array(doc_topic_dist_test)[:self.num_test_documents]
                    pred_labels = np.array(pred_labels)[:self.num_test_documents]

                    if not self.author_prediction and self.data.labels_available:
                        if self.supervision:
                            print('Test accuracy: %.4f' % accuracy_score(self.data.test_labels, pred_labels))
                            # print('Micro F1: %.4f' % (f1_score(self.data.test_labels, pred_labels, average='micro')))
                            # print('Macro F1: %.4f' % (f1_score(self.data.test_labels, pred_labels, average='macro')))
                        else:
                            classification_knn(doc_topic_dist_training, doc_topic_dist_test, self.data.training_labels, self.data.test_labels)
                    if self.author_prediction:
                        author_embeds = self.infer_author_embeds(sess)
                        doc_topic_dist = np.concatenate([doc_topic_dist_training, doc_topic_dist_test], axis=0)
                        authorship_link_prediction_map(doc_topic_dist, author_embeds, self.data.test_doc_author_links)
                    else:
                        citation_link_prediction_map(doc_topic_dist_test, self.data.test_pairwise_citations, self.num_training_documents)
            output_top_keywords(np.transpose(word_embeds_semantic), self.data.voc)
            doc_topic_dist = np.concatenate([doc_topic_dist_training, doc_topic_dist_test], axis=0)
            word_embeds = self.infer_word_embeds(sess)
            author_embeds = self.infer_author_embeds(sess)
            np.savetxt('./results/' + self.dataset_name + '_doc_topic_dist_training.txt', doc_topic_dist_training, fmt='%f', delimiter=' ')
            np.savetxt('./results/' + self.dataset_name + '_doc_topic_dist_test.txt', doc_topic_dist_test, fmt='%f', delimiter=' ')
            np.savetxt('./results/' + self.dataset_name + '_word_topic_dist.txt', word_embeds, fmt='%f', delimiter=' ')
            np.savetxt('./results/' + self.dataset_name + '_author_topic_dist.txt', author_embeds, fmt='%f', delimiter=' ')