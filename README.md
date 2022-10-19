# VGATM
This is the tensorflow implementation of KDD-2022 paper "[Variational Graph Author Topic Modeling](/paper/KDD22-VGATM.pdf)" by [Delvin Ce Zhang](http://delvincezhang.com/) and [Hady W. Lauw](http://www.hadylauw.com/home).

VGATM is a Grapn Neural Network model that extracts interpretable topics for documents with authors and venues. Topics of documents then fulfill document classification, citation prediction, etc.

![](/paper/model_architecture.JPG)

## Implementation Environment
- Python == 3.6
- tensorflow == 1.9.0
- numpy == 1.17.4
- sklearn == 0.23.2
- scipy == 1.5.2

## Run
- `python main.py -div kl -p gaussian  # VGATM-G (unsupervised)`
- `python main.py -div kl -p dirichlet  # VGATM-D (unsupervised)`
- `python main.py -div wasserstein -p gaussian  # VGATM-W (unsupervised)`

- `python main.py -div kl -p gaussian -sup 1  # VGATM-G (supervised)`
- `python main.py -div kl -p dirichlet -sup 1  # VGATM-D (supervised)`
- `python main.py -div wasserstein -p gaussian -sup 1  # VGATM-W (supervised)`

### Parameter Setting
- -ne: number of training epochs, default=15
- -lr: learning rate, default=0.01
- -ms: minibatch size, default=128
-	-dn: dataset name, ml or pl, default=ml
-	-nt: number of topics, default=64
-	-sup: label supervision, default=0 (no supervision)
-	-tr: training ratio of documents, default=0.8
-	-nn: number of sampled neighbours for convolution, default=5
-	-ws: word-word graph sliding window size, default=5
-	-wn: word-word graph number of neighbours for each word, default=5
-	-nl: number of convolutional steps L, default=2
-	-div: variational divergence metric R, kl or wasserstein, default=wasserstein
-	-p: predefined prior distribution p(.), gaussian or dirichlet, default=gaussian
-	-reg_div: hyperparameter \lambda_reg controlling variational divergence term, default=0.01
-	-reg_l2: hyperparameter for l2 regularization, default=1e-3
-	-kp: dropout keep probability, default=0.9
-	-ap: author prediction, default=0 (no author prediction)
-	-rs: random seed
-	-gpu: gpu

## Output
Results will be saved to `./results` file.
- `doc_topic_dist_training.txt` contains topic proportions of training documents.
- `doc_topic_dist_test.txt` contains topic proportions of test documents.
- `word_topic_dist.txt` contains topic proportions of words.
- `author_topic_dist.txt` contains topic proportions of authors.

## Reference
If you use our paper, including code and data, please cite

```
@inproceedings{vgatm,
  title={Variational Graph Author Topic Modeling},
  author={Zhang, Delvin Ce and Lauw, Hady W},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2429--2438},
  year={2022}
}
```
