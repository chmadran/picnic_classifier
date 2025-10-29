from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


#Based on the documentation and a few tries, let's pre process our data a bit more with the following arguments (after removing the headers/quotes etc from the original text as well) : 
# * stop_words : Removes common words like the, and, is, to
# * lowercase : Converts all text to lowercase e.g NASA == nasa
# * max_df : ignores terms that appear in x% of docs
# * min_df : Ignores terms that appear in fewer than x docs
# * [OPTIONAL] max_features : Keeps only the top-x words by frequency, actually we loose a lot of information here so not very relevant

def load_vectorizer(vectorization_method):
    if vectorization_method == "TF-IDF":
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            max_df=0.8,
            min_df=2,
            # max_features=10000,
        )
    else:
        raise ValueError(f"Unknown model '{vectorization_method}'.")
    return vectorizer


def load_model(model_name):
    if model_name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000) 
    else:
        raise ValueError(f"Unknown model '{model_name}'.")

def predict_text(texts, model, vectorizer, target_names, top_k=2):
    if isinstance(texts, str):
        texts = [texts]

    text_vec = vectorizer.transform(texts)
    probs = model.predict_proba(text_vec)

    for i, t in enumerate(texts):
        print(f"\nText: {t}\n")

        sorted_idx = np.argsort(probs[i])[::-1]

        for rank in range(top_k):
            cls_idx = sorted_idx[rank]
            print(f"Top {rank+1}: {target_names[cls_idx]} ({probs[i][cls_idx]*100:.2f}%)")

        print("-" * 60)

def show_top_k_words(vectors, label, doc_number, vectorizer, k=5):
    feature_names = vectorizer.get_feature_names_out()
    doc_vector = vectors[doc_number].toarray().flatten()
    top_indices = doc_vector.argsort()[-k:][::-1]

    print(f"\nTop {k} words for document {doc_number} of class {label}:\n")

    for idx in top_indices:
        print(f"{feature_names[idx]}: {doc_vector[idx]:.3f}")
