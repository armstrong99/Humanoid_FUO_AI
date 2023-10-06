# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from greetingData import greetings
from greetingData import responses

# Set a threshold for cosine similarity
cosine_similarity_threshold = 0.5 # You can adjust this threshold as needed


def resolveAIResponse(text):
    # Convert labels to numerical values
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(greetings)

    # Preprocess and vectorize the user's input
    input_text = text  # Use the provided input text
    input_vector = vectorizer.transform([input_text])

    # Calculate cosine similarity between the input and all greetings
    cosine_similarities = cosine_similarity(input_vector, X)

    # Find the index of the most similar greeting
    most_similar_idx = cosine_similarities.argmax()

    # Get the corresponding response and its similarity score
    max_similarity_score = cosine_similarities[0, most_similar_idx]
    response = responses[most_similar_idx]

    # Check if the similarity score is below the threshold
    if max_similarity_score < cosine_similarity_threshold:
        default_response = "Sorry, I can not really answer that."
        return default_response
    else:
        return response


# Example usage
user_input = "How's is the family"
response = resolveAIResponse(user_input)
print("Response:", response)
