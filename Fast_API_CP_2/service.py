from scipy.sparse import csr_matrix


def user_user_recommendations_sparse(user_id, user_similarity, user_item_matrix,
                                     item_mapping, top_n=5):
    # Ensure the user_item_matrix is in CSR format for efficient row querying
    if not isinstance(user_item_matrix, csr_matrix):
        user_item_matrix = csr_matrix(user_item_matrix)

    # Get the similarity scores for the target user directly from the sparse matrix
    similarity_scores = user_similarity.getrow(user_id).toarray().flatten()
    print(user_item_matrix.shape)
    print(similarity_scores.shape)
    # Predict scores by multiplying the similarity scores with the user-item matrix
    pred_scores = user_item_matrix.T.dot(similarity_scores).flatten()  # Removed .toarray() here

    # Get indices of already rated items to filter them out from recommendations
    rated_items = user_item_matrix.getrow(user_id).nonzero()[1]

    # Recommend items that the user hasn't rated yet
    recommendations = [(item, score) for item, score in enumerate(pred_scores) if item not in rated_items]

    # Sort the recommendations based on scores and return the top N
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Map indices to original ISBN numbers
    recommended_isbns = [item_mapping[item] for item, score in recommendations[:top_n]]

    return recommended_isbns


def get_books_details(isbn_list, rating_df):
    # Filter the books dataframe for the given ISBNs
    filtered_books = rating_df[rating_df['ISBN'].isin(isbn_list)]

    filtered_books = filtered_books.drop_duplicates(subset='ISBN')
    # Select required columns
    filtered_books = filtered_books[
        ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M',
         'Image-URL-L']]

    # Convert the dataframe to a dictionary format that matches the desired output
    books_details = filtered_books.to_dict(orient='records')

    return books_details


# def item_item_recommendations_sparse(item_index, item_similarity, user_item_matrix, item_mapping, top_n=5):
#     # Ensure the user_item_matrix is in CSR format for efficient column querying
#     if not isinstance(user_item_matrix, csr_matrix):
#         user_item_matrix = csr_matrix(user_item_matrix)
#
#     # Get the similarity scores for the target item directly from the sparse matrix
#     similarity_scores = item_similarity.getcol(item_index).toarray().flatten()
#
#     # Recommend items that are most similar
#     recommendations = [(item, score) for item, score in enumerate(similarity_scores) if item != item_index]
#
#     # Sort the recommendations based on scores and return the top N
#     recommendations.sort(key=lambda x: x[1], reverse=True)
#
#     # Map indices to original ISBN numbers
#     recommended_isbns = [item_mapping[item] for item, score in recommendations[:top_n]]
#
#     return recommended_isbns

def item_item_recommendations_sparse_by_title(book_title, title_to_isbn_mapping, item_similarity, user_item_matrix,
                                              isbn_to_index_mapping, item_mapping, top_n=5):
    # Convert book title to ISBN
    isbn = title_to_isbn_mapping.get(book_title)
    if isbn is None:
        return []  # Book title not found

    # Convert ISBN to item index
    item_index = isbn_to_index_mapping.get(isbn)
    if item_index is None:
        return []  # ISBN not found in the index mapping

    # Follow the existing process to get similarity scores and generate recommendations
    similarity_scores = item_similarity.getcol(item_index).toarray().flatten()
    recommendations = [(item, score) for item, score in enumerate(similarity_scores) if item != item_index]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    recommended_isbns = [item_mapping[item] for item, score in recommendations[:top_n]]

    return recommended_isbns
