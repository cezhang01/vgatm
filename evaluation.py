from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


def classification_knn(X_train, X_test, Y_train, Y_test):

    for k in [5, 10, 15, 20]:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, Y_train)
        prediction_label = classifier.predict(X_test)
        # print('Micro F1 %d: %.4f' % (k, f1_score(Y_test, prediction_label, average='micro')))
        # print('Macro F1 %d: %.4f' % (k, f1_score(Y_test, prediction_label, average='macro')))
        print('Test accuracy %d: %.4f' % (k, accuracy_score(Y_test, prediction_label)))


def citation_link_prediction_map(test_embeds, test_links, num_training_docs):

    test_links = np.copy(test_links)
    test_links -= num_training_docs
    auc, count = 0, 0
    for row_idx, row in enumerate(test_embeds):
        y_true = np.zeros(len(test_embeds))
        citations = test_links[test_links[:, 0] == row_idx]
        if len(citations) == 0:
            continue
        y_true[citations[:, 1]] = 1
        y_true = np.delete(y_true, row_idx)
        distance = np.sum(np.square(np.delete(test_embeds, row_idx, axis=0) - row), axis=1)
        y_score = - distance
        #y_score = np.matmul(np.delete(test_embeds, row_idx, axis=0), np.expand_dims(row, axis=1))
        auc += roc_auc_score(y_true, y_score)
        count += 1
    auc /= count
    print('Doc-doc link prediction AUC: %.4f' % auc)


def authorship_link_prediction_map(test_embeds, candidate_embeds, test_links):

    test_links = np.copy(test_links)
    # test_links[:, 0] -= num_training_docs
    auc, count = 0, 0
    for row_idx, row in enumerate(test_embeds):
        y_true = np.zeros(len(candidate_embeds))
        authors = test_links[test_links[:, 0] == row_idx]
        if len(authors) == 0:
            continue
        y_true[authors[:, 1]] = 1
        distance = np.sum(np.square(candidate_embeds - row), axis=1)
        #y_score = - distance
        y_score = np.matmul(candidate_embeds, np.expand_dims(row, axis=1))
        auc += roc_auc_score(y_true, y_score)
        count += 1
    auc /= count
    print('Authorship AUC: %.4f' % auc)


num_top_words = 10


def output_top_keywords(weights, voc):

    index = np.flip(np.argsort(weights)[:, -num_top_words:], axis=1)
    words = voc[index]
    print(words)