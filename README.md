# Movie Recommendation System

A comprehensive implementation of various movie recommendation algorithms using the MovieLens 100K dataset. This project demonstrates data mining and recommendation system techniques including collaborative filtering and graph-based approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project explores three different recommendation system approaches:

1. **User-Based Collaborative Filtering** - Recommends movies based on similar users' preferences
2. **Item-Based Collaborative Filtering** - Recommends movies similar to those a user has liked
3. **Graph-Based Recommendation (Pixie-Inspired)** - Uses weighted random walks on a bipartite user-movie graph

The implementation covers the complete pipeline from data exploration and cleaning to building sophisticated recommendation models.

## 📊 Dataset

The project uses the **MovieLens 100K** dataset, which contains:

- **943 users**
- **1,682 movies**
- **100,000 ratings** (scale: 1-5)

### Dataset Files

- `u.data` - User-movie ratings with timestamps
- `u.item` - Movie metadata (title, release date, IMDB link)
- `u.user` - User demographics (age, gender, occupation)

### Processed Files

The notebook generates cleaned CSV files:
- `ratings.csv` - Processed ratings with readable timestamps
- `movies.csv` - Movie information
- `users.csv` - User demographics

## ✨ Features

### Part 1: Data Exploration & Cleaning
- Load and parse MovieLens dataset files
- Handle missing values and data inconsistencies
- Convert Unix timestamps to readable datetime format
- Generate comprehensive dataset statistics
- Export processed data to CSV format

### Part 2: Collaborative Filtering
- **User-Based Filtering**
  - Compute user similarity using cosine similarity
  - Generate personalized recommendations based on similar users
  - Weight predictions by similarity scores
  
- **Item-Based Filtering**
  - Calculate item-item similarity matrix
  - Find movies similar to a given movie
  - Provide recommendations based on rating patterns

### Part 3: Graph-Based Recommendation
- Build bipartite user-movie interaction graph
- Implement weighted random walk algorithm (Pixie-inspired)
- Explore relationships through graph traversal
- Generate recommendations from graph structure

## 🚀 Installation

### Prerequisites

```bash
Python 3.7+
Jupyter Notebook
```

### Required Libraries

```bash
pip install pandas numpy scikit-learn
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Movie-Recommendation/
├── Movie-Recommendation.ipynb          # Main Jupyter notebook
├── README.md                           # This file
├── Pixie_Algorithm_Explanation.pdf    # Detailed algorithm explanation
├── Recommendation_Report.pdf           # Project report
├── u.data                              # Raw ratings data
├── u.item                              # Raw movie data
├── u.user                              # Raw user data
├── ratings.csv                         # Processed ratings
├── movies.csv                          # Processed movies
└── users.csv                           # Processed users
```

## 🔧 Implementation Details

### 1. Data Preprocessing

The notebook loads and processes raw MovieLens files:

```python
# Load ratings data
ratings = pd.read_csv("u.data", sep="\t", 
                     names=["user_id", "movie_id", "rating", "timestamp"])

# Convert timestamps
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
```

**Key Statistics:**
- Total Users: 943
- Total Movies: 1,682
- Total Ratings: 100,000
- Sparsity: ~93.7% (most user-movie pairs have no rating)

### 2. User-Based Collaborative Filtering

Creates a user-item matrix and computes user similarity:

```python
# Create user-movie rating matrix
user_movie_matrix = ratings.pivot(index='user_id', 
                                  columns='movie_id', 
                                  values='rating')

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
```

**Algorithm Steps:**
1. Find users similar to the target user
2. Get movies rated by similar users
3. Weight ratings by user similarity
4. Recommend top-N unrated movies

### 3. Item-Based Collaborative Filtering

Computes similarity between movies:

```python
# Transpose matrix to get movie similarities
item_similarity = cosine_similarity(user_movie_matrix.T.fillna(0))
```

**Algorithm Steps:**
1. Find movies similar to the query movie
2. Rank by similarity scores
3. Return top-N similar movies

### 4. Graph-Based Recommendation (Pixie-Inspired)

Builds a bipartite graph connecting users and movies:

```python
# Build adjacency list representation
graph = {}
for _, row in ratings.iterrows():
    user, movie = row['user_id'], row['movie_id']
    graph[user].add(movie)
    graph[movie].add(user)
```

**Random Walk Algorithm:**
1. Start from a user or movie node
2. Randomly walk to connected nodes
3. Track movie visit frequencies
4. Recommend most-visited movies

## 💻 Usage Examples

### User-Based Recommendations

```python
# Get recommendations for user 10
recommend_movies_for_user(user_id=10, num=5)
```

**Output:**
| Ranking | Movie Name |
|---------|-----------|
| 1 | In the Company of Men (1997) |
| 2 | Misérables, Les (1995) |
| 3 | Thin Blue Line, The (1988) |
| 4 | Braindead (1992) |
| 5 | Boys, Les (1997) |

### Item-Based Recommendations

```python
# Find movies similar to Jurassic Park
recommend_movies("Jurassic Park (1993)", num=5)
```

**Output:**
| Ranking | Movie Name |
|---------|-----------|
| 1 | Top Gun (1986) |
| 2 | Empire Strikes Back, The (1980) |
| 3 | Raiders of the Lost Ark (1981) |
| 4 | Indiana Jones and the Last Crusade (1989) |
| 5 | Speed (1994) |

### Graph-Based Recommendations

```python
# Random walk recommendations for user 1
weighted_pixie_recommend(start_point=1, walk_length=15, num=5)
```

**Output:**
| Ranking | Movie Name |
|---------|-----------|
| 1 | Spellbound (1945) |
| 2 | Air Force One (1997) |
| 3 | Streetcar Named Desire, A (1951) |
| 4 | Glimmer Man, The (1996) |

```python
# Find similar movies using random walks
weighted_pixie_recommend("Jurassic Park (1993)", walk_length=10, num=5)
```

**Output:**
| Ranking | Movie Name |
|---------|-----------|
| 1 | Young Frankenstein (1974) |
| 2 | Good Will Hunting (1997) |
| 3 | Kama Sutra: A Tale of Love (1996) |
| 4 | While You Were Sleeping (1995) |
| 5 | Ransom (1996) |

## 📈 Results

### Performance Comparison

| Method | Approach | Advantages | Limitations |
|--------|----------|------------|-------------|
| **User-Based CF** | Find similar users | Personalized, discovers new interests | Cold start problem, scalability |
| **Item-Based CF** | Find similar items | Stable, works with sparse data | Less serendipitous, popularity bias |
| **Pixie Random Walk** | Graph traversal | Handles sparsity, captures indirect relationships | Stochastic, parameter-dependent |

### Key Insights

1. **User-Based CF** excels at discovering diverse recommendations based on community preferences
2. **Item-Based CF** provides consistent, explainable recommendations based on content similarity
3. **Graph-Based** methods uncover hidden connections and work well with sparse interaction data

### Matrix Sparsity

The user-movie matrix has:
- **Total cells:** 1,586,126
- **Non-null ratings:** 100,000 (6.3%)
- **Missing values:** 1,486,126 (93.7%)

This extreme sparsity motivates the use of multiple recommendation strategies.

## 🛠 Technologies Used

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (cosine similarity)
- **Jupyter Notebook** - Interactive development environment

## 📚 Key Concepts

### Collaborative Filtering

Leverages user behavior patterns to make recommendations:
- **Memory-based:** Uses entire user-item matrix (implemented here)
- **Model-based:** Learns latent factors (e.g., matrix factorization)

### Cosine Similarity

Measures similarity between vectors based on angle:

```
similarity(A, B) = cos(θ) = (A · B) / (||A|| × ||B||)
```

Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite

### Random Walks

Probabilistic graph traversal technique:
- Start from a node
- Randomly move to neighbors
- Popular nodes get visited more often
- Models how users might explore related content

## 🎓 Learning Outcomes

This project demonstrates:

1. **Data preprocessing** for recommendation systems
2. **Matrix factorization** and similarity computation
3. **Collaborative filtering** algorithms
4. **Graph-based** recommendation techniques
5. **Cold start** and **sparsity** challenges
6. **Evaluation** of different recommendation approaches

## 📖 Additional Resources

- **MovieLens Dataset:** [GroupLens Research](https://grouplens.org/datasets/movielens/)
- **Collaborative Filtering:** [Netflix Prize](https://www.netflixprize.com/)
- **Pixie Algorithm:** [Pinterest Engineering Blog](https://medium.com/pinterest-engineering/pixie-a-system-for-recommending-3-billion-items-to-200-million-users-in-real-time-cf929b4c4e6b)
- **Graph-Based Recommendations:** [Stanford CS246](http://web.stanford.edu/class/cs246/)

## 🤝 Contributing

This is an academic project for **ITCS 6162: Data Mining** at UNC Charlotte. While this is a course assignment, suggestions and feedback are welcome!

## 📝 License

This project is created for educational purposes as part of coursework at UNC Charlotte.

## 👤 Author

**Course:** ITCS 6162 - Data Mining  
**Institution:** University of North Carolina at Charlotte  
**GitHub:** [https://github.com/Ialzouby/Movie-Recommendation](https://github.com/Ialzouby/Movie-Recommendation)

## 🙏 Acknowledgments

- **MovieLens** dataset provided by GroupLens Research
- **Pinterest's Pixie** algorithm for inspiration on graph-based recommendations
- **UNC Charlotte** Department of Computer Science
- Course instructors and teaching assistants

---

**Note:** This project implements fundamental recommendation algorithms for educational purposes. Production systems would require additional considerations including scalability, real-time updates, diversity, and bias mitigation.

