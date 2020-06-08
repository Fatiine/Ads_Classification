import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation


nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")


def text_to_words(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        token
        for token in tokens
        if token not in russian_stopwords
        and token != " "
        and token.strip() not in punctuation
    ]
    return tokens


def compute_words(text):
    tokens = text_to_words(text)
    # text = " ".join(tokens)
    return len(tokens)


def len_commun_tokens(text1, text2):
    tokens1 = text_to_words(text1)
    tokens2 = text_to_words(text2)
    commun_words = [w for w in set(tokens1) if w in set(tokens2)]

    return len(commun_words)

def len_commun_different_tokens(text1, text2):
    tokens1 = text_to_words(text1)
    tokens2 = text_to_words(text2)
    commun_words = [w for w in set(tokens1) if w in set(tokens2)]
    all_tokens = set(tokens1).union(set(tokens2))
    return len(commun_words), len(all_tokens) - len(commun_words)


def len_different_tokens(text1, text2):
    tokens1 = text_to_words(text1)
    tokens2 = text_to_words(text2)
    all_tokens = set(tokens1).union(set(tokens2))
    return len(all_tokens) - len_commun_tokens(text1, text2)


def plot_confusion(
    cm, target_names, title="Confusion matrix", cmap=None, normalize=True
):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="blue" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="blue" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
    )
    plt.show()
