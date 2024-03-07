import logging
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service import user_user_recommendations_sparse, get_books_details, item_item_recommendations_sparse_by_title

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app definition
app = FastAPI()


# Models for request bodies
class UserRecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5


class IndexRecommendationRequest(BaseModel):
    book_title: str
    top_n: int = 5


# Function to load recommendations data
def load_recommendation_data():
    try:
        with open("recommendation.pkl", "rb") as pickle_in:
            loaded_functions = pickle.load(pickle_in)
        logger.info("Recommendation data loaded successfully.")
        return loaded_functions
    except Exception as e:
        logger.error(f"An error occurred while loading the recommendation data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Dependency that loads the recommendation data
def get_recommendation_functions():
    recommendation_data = load_recommendation_data()
    # logger.info(f"Recommendation_data: {recommendation_data}")
    return (recommendation_data['user_similarity'], recommendation_data['item_similarity'],
            recommendation_data['rating_df'], recommendation_data['top_n'],
            recommendation_data['user_item_matrix_sparse'], recommendation_data['item_mapping'],
            recommendation_data['title_to_isbn_mapping'], recommendation_data['isbn_to_index_mapping'])


@app.post("/recommendations/user/")
def get_user_recommendations(request_body: UserRecommendationRequest):
    (user_similarity, item_similarity, rating_df, top_n, user_item_matrix_sparse, item_mapping, title_to_isbn_mapping,
     isbn_to_index_mapping) = get_recommendation_functions()

    recommendations = user_user_recommendations_sparse(user_id=request_body.user_id,
                                                       user_similarity=user_similarity,
                                                       user_item_matrix=user_item_matrix_sparse,
                                                       item_mapping=item_mapping,
                                                       top_n=request_body.top_n)

    return get_books_details(recommendations, rating_df)


@app.post("/recommendations/item/")
def get_item_recommendations(request_body: IndexRecommendationRequest):
    (user_similarity, item_similarity, rating_df, top_n, user_item_matrix_sparse, item_mapping, title_to_isbn_mapping,
     isbn_to_index_mapping) = get_recommendation_functions()
    recommendations = item_item_recommendations_sparse_by_title(book_title=request_body.book_title,
                                                                item_similarity=item_similarity,
                                                                user_item_matrix=user_item_matrix_sparse,
                                                                item_mapping=item_mapping,
                                                                top_n=request_body.top_n,
                                                                title_to_isbn_mapping=title_to_isbn_mapping,
                                                                isbn_to_index_mapping=isbn_to_index_mapping)

    return get_books_details(recommendations, rating_df)
