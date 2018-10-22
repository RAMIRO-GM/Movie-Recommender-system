
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
forrest_gump_ratings = movie_matrix['Forrest Gump (1994)'] 
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






 