import streamlit as st
import model

# ---------------- UI DESIGN ----------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }

    h1 {
        text-align: center;
        color: #facc15;
    }

    /* ✅ FIX LABEL COLOR (IMPORTANT) */
    div[data-testid="stSelectbox"] label {
        color: white !important;
        font-size: 16px;
        font-weight: bold;
    }

    /* Dropdown text */
    div[data-baseweb="select"] {
        color: black;
    }

    div.stButton > button {
        background-color: #facc15;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------

new_df = model.new_df
similarity = model.similarity

# ---------------- TITLE ----------------

st.markdown("<h1>🎬 Smart Movie Recommender</h1>", unsafe_allow_html=True)

# ---------------- SELECT MOVIE ----------------

movie_list = ["Select a movie"] + list(new_df['title'].values)

selected_movie = st.selectbox(
    "Type or select a movie",
    movie_list,
    key="movie_select"
)

# ---------------- RECOMMEND FUNCTION ----------------

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)

    return recommended_movies

# ---------------- BUTTON ----------------

if st.button("Recommend"):
    if selected_movie == "Select a movie":
        st.warning("⚠️ Please select a movie first")
    else:
        recommendations = recommend(selected_movie)

        st.subheader("🎯 Recommended Movies:")
        for movie in recommendations:
            st.write("👉", movie)