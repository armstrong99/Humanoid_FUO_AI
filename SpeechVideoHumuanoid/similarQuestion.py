import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def is_question_similar(input_question, question_list):
    # Convert input question and question list to lowercase
    input_question = input_question.lower()
    question_list = [q.lower() for q in question_list]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(question_list)

    # Vectorize the input question
    input_vector = vectorizer.transform([input_question])

    # Calculate cosine similarity between the input question and all questions in the list
    cosine_similarities = cosine_similarity(input_vector, X)
    # Check if any similarity score is above the threshold
    if np.any(cosine_similarities > 0.5):
        return True
    else:
        return False


# # Example usage
# input_question = "Can you spot the box?"
# question_list = [
#     "Can you see the box?",
#     "Do you have visual access to the box?",
#     "Is the box in your line of sight?",
#     "What is your name?",
#     "How's the weather today?",
# ]

# is_similar = is_question_similar(input_question, question_list)
# print("Is the question similar?", is_similar)
