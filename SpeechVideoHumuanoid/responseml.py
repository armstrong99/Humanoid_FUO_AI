# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from greetingData import greetings
from greetingData import responses
from boxQuestion import similar_questions

# Set a threshold for cosine similarity
cosine_similarity_threshold = 0.4 # You can adjust this threshold as needed

def get_box_info():
    return "Box info."

def resolveAIResponse(text):
    # import pdb; pdb.set_trace()
    # Convert labels to numerical values
    # TF-IDF vectorization
    vectorizer_x = TfidfVectorizer()
    X = vectorizer_x.fit_transform(greetings)

    vectorizer_y = TfidfVectorizer()
    Y = vectorizer_y.fit_transform(similar_questions)

    # Preprocess and vectorize the user's input
    input_text = text  # Use the provided input text
    input_vector_x = vectorizer_x.transform([input_text])
    input_vector_y = vectorizer_y.transform([input_text])

    # Calculate cosine similarity between the input and all greetings
    greet_cosine_similarities = cosine_similarity(input_vector_x, X)
    box_cosine_similarities = cosine_similarity(input_vector_y, Y)

    # Find the index of the most similar greeting
    most_similar_greet_idx = greet_cosine_similarities.argmax()
    most_similar_box_idx = box_cosine_similarities.argmax()

    # Get the corresponding response and its similarity score
    greet_max_similarity_score = greet_cosine_similarities[0, most_similar_greet_idx]
    greet_response = responses[most_similar_greet_idx]

    box_max_similarity_score = box_cosine_similarities[0, most_similar_box_idx]
 
    # Check if the similarity score is below the threshold
    if greet_max_similarity_score > box_max_similarity_score:
        max_similarity_score = greet_max_similarity_score
        response = greet_response, "greet"
    else:
        max_similarity_score = box_max_similarity_score
        response = get_box_info(), "box"    

    if max_similarity_score < cosine_similarity_threshold:
        default_response = "Sorry, I can not really answer that."
        return default_response, ""
    else:
        return response


# Example usage
user_input = "What can you see?"
response = resolveAIResponse(user_input)
print("Response:", response)
