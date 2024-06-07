import evaluate
from bert_score import score as bert_score
import json 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the Universal Sentence Encoder model
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(model_url)

def embed_sentences(sentences):
    embeddings = embed(sentences)
    return embeddings

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def main():
    
    # Load JSON data from file
    with open('../../Experiments_results/results_x86codellama_description.json', 'r') as f:
        data = json.load(f)
        
    all_predictions = data.get('pred', [])
    all_references = data.get('gt', [])
    
    predictions = []
    references = []


    for i in range(len(all_references)):
        if all_references[i] and all_predictions[i]:  # Check if the value in gt is not empty
            predictions.append(all_predictions[i])
            references.append(all_references[i])
        
    print(len(references))
    # Load evaluation metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    bert_metric = evaluate.load("bertscore")

    # Compute BERTScore
    bert_score_precision, bert_score_recall, bert_score_f1 = bert_score(predictions, references, lang="en", verbose=False, batch_size=2)

    # Compute BLEU
    bleu_score = bleu_metric.compute(predictions=predictions, references=references,smooth=True)['bleu']
    
    # Compute Rouge-L
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)['rougeL']
    
    # Get embeddings for the predictions and references
    pred_embeddings = embed_sentences(predictions)
    ref_embeddings = embed_sentences(references)

    # Compute cosine similarity
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        similarity = cosine_similarity(pred_emb, ref_emb)
        similarities.append(similarity)
    
    # Print the similarity scores
    for i, similarity in enumerate(similarities):
        print(f"Similarity between prediction {i+1} and reference {i+1}: {similarity:.4f}")

    # Calculate and print the overall similarity score
    overall_similarity = np.mean(similarities)
    print(f'Bleu score: {bleu_score}\n' +
            f'Rouge: {rouge_score}\n' +
            f'Bert_score_precision: {bert_score_precision.mean().item()}\n' +
            f'Bert_score_recall: {bert_score_recall.mean().item()}\n' +
            f'Bert_score_f1: {bert_score_f1.mean().item()}\n' +
            f"Average similarity score: {overall_similarity:.4f}")
    print('end final')
    
if __name__ == "__main__":
    main()