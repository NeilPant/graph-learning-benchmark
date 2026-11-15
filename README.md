# Graph learning with controlled baselines

How much do real citation edges help a graph neural network classify papers? This project compares three versions of the same two-layer graph convolutional network on Cora. It grew out of a 2025 Ego-Facebook notebook where the target classes were defined by node degree, which was also a model input.

## Cora results

| Model input | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Real citation edges and paper features | **79.90% ± 0.84 pp** | **79.04% ± 0.97 pp** |
| Paper features and self-loops | 53.10% ± 3.98 pp | 51.39% ± 3.36 pp |
| Paper features and shuffled node assignments | 25.74% ± 2.63 pp | 22.19% ± 2.99 pp |

Values are means ± sample standard deviations over seeds 42–46 on the same public Planetoid split. Citation edges improved mean accuracy by **26.80 percentage points** over self-loops in this setup. Shuffling reassigns node IDs in the edge list while keeping the paper features and labels in place. It preserves the unlabeled graph topology and degree distribution, but changes which papers occupy the graph's nodes. Its lower score is consistent with message passing mixing features from unrelated papers.

### Evaluation protocol

- Cora contains 2,708 papers, 1,433 binary word features, and seven topic classes. The public split uses 140 training, 500 validation, and 1,000 test nodes.
- Each condition uses the same parameter count, optimizer, dropout, and seed for initialization. All receive self-loops. The self-loop condition is equivalent to a feature-only two-layer perceptron.
- Training uses only labels in the training split. Validation cross-entropy selects the checkpoint, with up to 200 epochs and a patience of 30. Accuracy and macro-F1 are calculated on the test split after checkpoint selection.
- This is a transductive experiment. Training can use features and graph connections for test nodes without using their labels. The reported variation reflects random seeds on a single fixed split. The comparison establishes behavior for these settings and does not estimate performance on unseen graphs or compare independently tuned models.

[Per-seed scores, checkpoint epochs, package versions, and input checksums](results/cora.json)

## What the original notebook showed

The Ego-Facebook notebook called its classes active, passive, and inactive. Those names corresponded to degree above 10, degree from 2 through 10, and degree at most 1. On the published graph, predicting the majority class gives 76.23% accuracy. Applying the same degree rule used to define the classes gives 100% by construction. These classes describe graph structure. The dataset and experiment supply no observed activity labels.

[Aggregate audit results](results/ego-facebook-audit.json)

## Reproduce the results

The reference run used Python 3.14.7 on Apple Silicon, on CPU. The Cora loader fetches the published dataset on the first run and caches it locally.

```sh
git clone https://github.com/NeilPant/graph-learning-benchmark.git
cd graph-learning-benchmark
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m graphbench.cora --output results/cora-reproduced.json
```

To rerun the degree audit, download and decompress the combined edge list from [Stanford SNAP](https://snap.stanford.edu/data/ego-Facebook.html), then run the following command with the path to the file.

```sh
python -m graphbench.ego_audit data/facebook_combined.txt
```

## Project history and references

Neil Pant's original notebook explored engineered network features and graph convolution on Ego-Facebook. This follow-up was prepared with Codex assistance and documents the experiment, controls, and verification commands. The [MIT license](LICENSE) covers the project code. Obtain datasets from their publishers.

- McAuley and Leskovec, *Learning to Discover Social Circles in Ego Networks*, 2012. [Paper and data](https://snap.stanford.edu/data/ego-Facebook.html)
- Yang, Cohen, and Salakhutdinov, *Revisiting Semi-Supervised Learning with Graph Embeddings*, 2016. [Paper](https://proceedings.mlr.press/v48/yanga16.html)
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, 2017. [Paper](https://openreview.net/forum?id=SJU4ayYgl)
