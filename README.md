# Picnic Classifier  
**Take-home test — 3rd stage interview (30/10/2025)**  

## Dataset  

For this task, I chose the **20 Newsgroups dataset**, a benchmark for text classification. It contains a balanced number of documents and topics, making it a good compromise between dataset size and complexity. The dataset comprises around **18,846** newsgroup posts or variable-length spread across **20 categories**. 

**Source:** [Scikit-learn 20 Newsgroups Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)


## Preprocessing  

1. **Text Cleaning:** minimal preprocessing (with the scikit-learn built in dataset getter).  
2. **Vectorization:** used **TF-IDF (Term Frequency–Inverse Document Frequency)** 
3. **Train/Test Split:** predefined by the dataset (60/40) 

## Model  

The model used is a **Logistic Regression classifier** which outputs **probabilities** for each class making it easy to interpret how confident the model is on its preds (e.g *“80% chance this is about space”*) and to see what each word’s represents for a specific class.

## Results  

Accuracy, Precision, Recall and F1 Score all sit at about `0.88`. The model performs consistently across categories, though it can struggle with closely related topics (like `comp.sys.ibm.pc.hardware` vs `comp.sys.mac.hardware`).

## Thinking Points  

This project is simple and efficient but it has clear limitations like no contextual understanding (like *“bank robber”* vs *“river bank”*) as the embeddings and the model do not take into account word order and semantics.

Improvement points could be : 
- **Vectorization:** : Experiment `TF-IDF` with different parameters, or straight dense embeddings (Word2Vec etc).  
- **Model comparison:** : try other linear models (Naive Bayes, SVM etc).  
- **Deep Learning:** : fine-tune (or even try with just a pretrained and a classification head?) a transformer (e.g DistilBERT to keep it light) for context-aware classification.  
- **Performance optimization:** : benchmark training/inference time and memory footprint.  

But first, ask the right questions to the stakeholders. Is accuracy or speed or maybe even size the priority ? What's the actual dataset like ? 

## How to run 

* Create a venv : `conda create picnic_env`   
* Activate the env : `conda activate picnic_env`   
* Install dependencies : `pip install -r requirements.txt`   
* Open and run the notebook : `picnic_classifier.ipynb`   
