import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Set up the page layout and logo
st.set_page_config(page_title="Amazon Market Basket Analysis", layout="wide")
st.image("MBA_logo.png", width=200)

# Sidebar with dropdown menu for project details and author information
st.sidebar.title("M.B.A - Amazon Dataset")
st.sidebar.write("Market Basket Analysis Tool for Amazon Data")
menu = st.sidebar.selectbox("Menu", ["Home", "About"])

if menu == "About":
    st.sidebar.subheader("Developed by: Prafull Raj")
    st.sidebar.write("""
    **Amazon Market Basket Analysis** provides insights into Amazon product data, including 
    collaborative and content-based product recommendations, and explores key metrics 
    like ratings, pricing, and user reviews. This tool helps Amazon sellers understand trends 
    and make data-driven decisions for their products.
    """)
else:
    st.sidebar.write("Use the app to explore Amazon data and get product recommendations.")

# Initialize session state for tracking progress
if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False

# Function to clean the dataset
def clean_data(df):
    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    return df

# Function to generate graphs and insights
def generate_visuals(df):
    st.subheader("Rating Distribution")
    sns.histplot(df['rating'], kde=True)
    st.pyplot(plt.gcf())
    
    st.subheader("Discounted vs Actual Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['discounted_price'], label='Discounted Price', color='green', kde=True, ax=ax)
    sns.histplot(df['actual_price'], label='Actual Price', color='blue', kde=True, ax=ax)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Top Categories by Count")
    fig, ax = plt.subplots()
    df['main_category'] = df['category'].str.split('|').str[0]
    df['main_category'].value_counts().head(5).plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

    st.subheader("Discount vs Rating Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x='discount_percentage', y='rating', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Average Rating by Category")
    fig, ax = plt.subplots()
    avg_rating_by_category = df.groupby('main_category')['rating'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_rating_by_category.values, y=avg_rating_by_category.index, ax=ax)
    st.pyplot(fig)

# Function to recommend products based on collaborative filtering (similar users)
def recommend_products_collaborative(user_id, num_recommendations=5):
    # Check if the user exists in the similarity matrix
    if user_id not in user_similarity_df_cleaned.index:
        return pd.DataFrame({"Error": ["User ID not found"]})

    # Get the most similar users to the given user
    similar_users = user_similarity_df_cleaned[user_id].sort_values(ascending=False).index[1:num_recommendations+1]
    
    # Find the products these users have rated highly and recommend them
    recommended_products = df_cleaned_users[df_cleaned_users['user_id'].isin(similar_users)][['product_name', 'rating']].sort_values(by='rating', ascending=False).head(num_recommendations)
    return recommended_products

# Function to recommend similar products based on content-based filtering
def recommend_similar_products(product_name, num_recommendations=5):
    idx = df_uploaded[df_uploaded['product_name'] == product_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return df_uploaded.iloc[sim_indices][['product_name', 'category', 'rating']]

# Upload Data Page
st.header("Amazon Market Basket Analysis")
st.write("Explore product data, pricing trends, ratings, and recommendations based on user behavior for Amazon products.")

uploaded_file = st.file_uploader("Upload Amazon Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    df_uploaded = clean_data(df_uploaded)  # Clean the data after uploading
    st.session_state['uploaded'] = True  # Update session state to allow progressing
    st.success("Data uploaded and cleaned successfully!")
    st.write(df_uploaded.head())

    # Generate graphs and insights
    generate_visuals(df_uploaded)

    # Collaborative Filtering Process
    df_cleaned_users = df_uploaded.copy()
    df_cleaned_users = df_cleaned_users.assign(user_id=df_cleaned_users['user_id'].str.split(','))
    df_cleaned_users = df_cleaned_users.explode('user_id').reset_index(drop=True)
    user_product_matrix_cleaned = df_cleaned_users.pivot_table(index='user_id', columns='product_name', values='rating', aggfunc='mean').fillna(0)
    user_product_sparse_cleaned = csr_matrix(user_product_matrix_cleaned)
    user_similarity_cleaned = cosine_similarity(user_product_sparse_cleaned)
    user_similarity_df_cleaned = pd.DataFrame(user_similarity_cleaned, index=user_product_matrix_cleaned.index, columns=user_product_matrix_cleaned.index)

    # Content-Based Filtering Process
    df_uploaded['combined_features'] = df_uploaded['product_name'] + ' ' + df_uploaded['category'] + ' ' + df_uploaded['about_product']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_uploaded['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Example recommendations for Collaborative Filtering and Content-Based Filtering
    st.header("Collaborative Filtering Recommendations")
    example_user_id = df_cleaned_users['user_id'].iloc[0]  # Taking first user for demonstration
    collaborative_recommendations = recommend_products_collaborative(example_user_id, 5)
    st.write(collaborative_recommendations)

    st.header("Content-Based Filtering Recommendations")
    example_product_name = df_uploaded['product_name'].iloc[0]  # Taking first product for demonstration
    content_based_recommendations = recommend_similar_products(example_product_name, 5)
    st.write(content_based_recommendations)

else:
    st.warning("Please upload a CSV file.")
