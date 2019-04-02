
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# =============================================================================
# This dataset (ml-latest-small) describes 5-star rating and free-text tagging 
# activity from [MovieLens](http://movielens.org), a movie recommendation service. 
# It contains 100836 ratings and 3683 tag applications across 9742 movies. 
# These data were created by 610 users between March 29, 1996 and September 24, 2018.
#  This dataset was generated on September 26, 2018
# =============================================================================

# Reading the csv file for movie ratings

#df = pd.read_csv('ratings.csv', sep='\t', names=['user_id','item_id','rating','titmestamp'])
df = pd.read_csv('ratings.csv')
df.head()
# We can see some statistical metrics
df.describe()

# Reading the csv file for movie titles
movie_titles = pd.read_csv('movies.csv')
# to be able to see the whole columns and rows in the terminal
pd.set_option('display.expand_frame_repr', False)
movie_titles.head()

# Merge 2 data sets into one (just the column of interest)
df = pd.merge(df, movie_titles, on='movieId') 
df.head()

# average rating of each movie
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

# Checking total number of ratings for a movie
# As we see not all the movies have a rating
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
ratings.head()

# get the plots in a different window
%matplotlib qt5 
#get the histogram  of ratings and number of ratings
ratings['rating'].hist(bins=50)
ratings['number_of_ratings'].hist(bins=60)

# check the relationship between the rating of a movie and the number of ratings.
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)

# creating the utility matrix
# columns -> movies titles, users->index, ratings-> values
movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
movie_matrix.head() 

#Visualize the top ten rated movies
ratings.sort_values('number_of_ratings', ascending=False).head(10)

#example for 3 movies to find their correlations with others
AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']
AFO_user_rating.head()
contact_user_rating.head()
forrest_gump_ratings = movie_matrix['(500) Days of Summer (2009)'] 
forrest_gump_ratings.head()


#Finding the correlation
#By using Corrwith omputes the pairwise correlation of rows or columns of 
#two dataframe objects
similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)
similar_to_air_force_one.head()
similar_to_forest_gump = movie_matrix.corrwith(forrest_gump_ratings)
similar_to_contact = movie_matrix.corrwith(contact_user_rating)
similar_to_contact.head()

# Get all the most correlated movies, show them and drop the Null values
corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])
corr_contact.dropna(inplace=True)
corr_contact.head()
corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
corr_AFO.dropna(inplace=True)
corr_AFO.head()
corr_forrest_gump = pd.DataFrame(similar_to_forest_gump, columns=['Correlation'])  
corr_forrest_gump.dropna(inplace=True)  
corr_forrest_gump.head() 

# Let's get the cosine similarity
# =============================================================================
# It is represented by the dot product of 2 vectors divided by the 
# product of their norms (v1 dot v2)/{||v1||*||v2||)
# =============================================================================
def cos_sim(a, b):
    # a and b must be vectors
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

def cosine_simi(vector,matrix):
   return ( np.sum(vector*matrix,axis=1) / 
           ( np.sqrt(np.sum(matrix**2,axis=1)) * np.sqrt(np.sum(vector**2)) ) )[::-1]

# =============================================================================
#  Let's compute the  angle
#  This will be calculated by arccos of the dot product of 2 vectors 
#  divided by the product of their norms  
# =============================================================================
def angle_similarity(cosine_sim):
    angle_in_radians = math.acos(cosine_sim)
    return math.degrees(angle_in_radians)


# remove the nan values
util = movie_matrix.fillna(0)
movie1 = AFO_user_rating.fillna(0) 
# transforms the dataframes into numpy arrays
#utility_matrix = np.array(movie_matrix.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0))
utility_matrix = util.values
movie11 = movie1.values


# Another way to get the simmilarity
from scipy import spatial
r = utility_matrix[:,10]
similar = 1 - spatial.distance.cosine(movie1, r)
similar1 = cos_sim(movie1,r)
similar2 = cosine_simi(movie11,r.reshape(1,-1))

# With the function from SciKit Learn
from sklearn.metrics.pairwise import cosine_similarity
#single evaluation of the cosine similarity vector vs 1 vector
similarity = cosine_similarity(movie11.reshape(1,-1),r.reshape(1,-1))

# getting the location index of a desired movie
movie_index = movie_matrix.columns.get_loc("'Hellboy': The Seeds of Creation (2004)")
# creating the isolated vector for that specific movie could be either one of the 2 following lines
#movie_selected  = utility_matrix[1,0:np.size(utility_matrix,0)].reshape(1,-1)
movie_selected = utility_matrix[:,movie_index].reshape(1,-1)
# get the transpose matrix of the utility matrix as needed to compute the cosine similarity function
utility_matrix_trans = utility_matrix.T
similarity2 = cosine_similarity(movie_selected, utility_matrix_trans)

##################################################################################

dataframe = pd.DataFrame.from_records(similarity2)
dataframe1 = pd.DataFrame(data=similarity2, columns='tre')
similar_df = pd.merge(dataframe1, movie_titles, on='movies1') 


indices = pd.Series(movie_titles.index, index=movie_titles['title'])
#idx = indices[title]

# Get the pairwsie similarity scores of all movies with that movie
sim_scores = list(enumerate(similarity))

# Sort the movies based on the similarity scores
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

# Get the scores of the 10 most similar movies
sim_scores = sim_scores[1:11]

# Get the movie indices
movie_indices = [i[0] for i in sim_scores]

# Return the top 10 most similar movies
return movie_titles['title'].iloc[movie_indices]


 