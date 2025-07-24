# 🎬 Movie Recommendation System

A personalized movie recommendation system that suggests movies based on content similarity. Built using Python, pandas, scikit-learn, and deployed with Streamlit for an interactive user interface.

---

## 📌 Features

- ✅ Content-based filtering using movie metadata  
- ✅ Streamlit web interface for real-time recommendations  
- ✅ Preprocessed `.pkl` files for fast loading  
- ✅ Recommends 5 similar movies for any selected movie  
- ✅ Easy to run locally

---

## 🧠 How It Works

The system uses **content-based filtering**:

1. **Data Source**: `tmdb_5000_movies.csv`  
2. **Features Used**: Genres, keywords, overview, cast, crew  
3. **NLP Techniques**: TF-IDF, cosine similarity  
4. **Recommendation**: Top 5 most similar movies returned for a given title

---
## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Ishagarg05/Movie-Recommendation-System.git
cd Movie-Recommendation-System
