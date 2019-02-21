# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import re
import random

class Movie:
    def __init__(self):
      # possible titles for the movie
      self.titles = []

      # a tuple where the first entry is the year the movie was made; 
      # for shows, first entry is start year and second entry is end year     
      self.year = None

def edit_distance(word1, word2):
  '''
  Case insensitive edit distance, excluding starting and trailing whitespace
  '''
  word1 = word1.strip().lower()
  word2 = word2.strip().lower()
  dp = [[0]*(len(word2) + 1) for i in range(len(word1) + 1)]

  for i in range(1, len(word1) + 1):
      dp[i][0] = i
  for j in range(1, len(word2) + 1):
      dp[0][j] = j

  for i in range(1, len(word1) + 1):
    for j in range(1, len(word2) + 1):
      if word1[i - 1] == word2[j - 1]:
        dp[i][j] = dp[i - 1][j - 1]
      else:
        dp[i][j] = min(dp[i - 1][j - 1] + 2,
                       dp[i - 1][j]     + 1,
                       dp[i][j - 1]     + 1)
  return dp[-1][-1]

def extract_titles_and_year(title):
  '''
  Returns a new Movie object with possible extracted movie titles and the movie
  year (start and end year for shows)
  '''
  mov = Movie()
  all_titles = []

  year_pat = '\((\d{4})-?(\d{4})?\)'
  pats = re.findall(year_pat, title)
  if pats:
    if pats[0][1] == '':
      mov.year = (pats[0][0],)
    else:
      mov.year = pats[0]
    if len(mov.year) == 1:
      title = title.replace('(' + mov.year[0] + ')', '')
    elif len(mov.year) == 2:
      title = title.replace('(' + mov.year[0] + '-' + mov.year[1] + ')', '')
  
  alt_name_pat = '\((.*)\)'
  pats = re.findall(alt_name_pat, title)
  if pats:
    alt_name = pats[0]
    title = title.replace('(' + alt_name + ')', '')
    alt_name = alt_name.replace('a.k.a.', '')

  # make sure full title is first in list
  all_titles.append(title)
  if pats:
    all_titles.append(alt_name)
#  if ':' in title:
#    all_titles.extend(title.split(':'))

  # strip whitespace, change [&+] to "and", move articles to end
  for i in range(len(all_titles)):
    t = move_article_to_end(all_titles[i].strip())
    t = t.replace('&', 'and')
    t = t.replace('+', 'and')
    all_titles[i] = t
  mov.titles = all_titles

  return mov
   
def move_article_to_end(title):
  '''
  Moves English articles (a, an, the) from the front of the title to the end,
  as is convention. Also move French articles (le, la, les) and Spanish
  articles (el, la, los)
  '''
  if bool(re.match('A ', title, re.I)):
    title = title[2:] + ', ' + 'A'
  elif bool(re.match('An ', title, re.I)):
    title = title[3:] + ', ' + 'An'
  elif bool(re.match('The ', title, re.I)):
    title = title[4:] + ', ' + 'The'
  elif bool(re.match('Le ', title, re.I)):
    title = title[3:] + ', ' + 'Le'
  elif bool(re.match('La ', title, re.I)):
    title = title[3:] + ', ' + 'La'
  elif bool(re.match('Les ', title, re.I)):
    title = title[4:] + ', ' + 'Les'
  elif bool(re.match('El ', title, re.I)):
    title = title[3:] + ', ' + 'El'
  elif bool(re.match('Los ', title, re.I)):
    title = title[4:] + ', ' + 'Los'
  return title

class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
      # The chatbot's default name is `moviebot`. Give your chatbot a new name.
      self.name = 'moviebot'

      self.creative = creative

      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, ratings = movielens.ratings()
      self.sentiment = movielens.sentiment()

      # a tuple with the index(es) of the current movie being discussed 
      self.current_movie = None
      # a tuple with the sentiment of the movie being discussed
      self.current_sentiment = None
      # a list of (movie, sentiment) tuples that the user has described and
      # the chatbot has processed
      self.user_movies = []
      self.user_movie_set = set()

      self.prefix_match_found = False

      # preprocess movie list by extracting possible titles and year
      self.movies = []
      for entry in self.titles:
        self.movies.append(extract_titles_and_year(entry[0]))
      #############################################################################
      # TODO: Binarize the movie ratings matrix.                                  #
      #############################################################################

      # Binarize the movie ratings before storing the binarized matrix.
      self.ratings = ratings
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
      """Return a message that the chatbot uses to greet the user."""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "How can I help you?"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Have a nice day!"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return goodbye_message


    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
      """Process a line of input from the REPL and generate a response.

      This is the method that is called by the REPL loop directly with user input.

      You should delegate most of the work of processing the user's input to
      the helper functions you write later in this class.

      Takes the input string from the REPL and call delegated functions that
        1) extract the relevant information, and
        2) transform the information into a response to the user.

      Example:
        resp = chatbot.process('I loved "The Notebok" so much!!')
        print(resp) // prints 'So you loved "The Notebook", huh?'

      :param line: a user-supplied line of text
      :returns: a string containing the chatbot's response to the user input
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method,         #
      # possibly calling other functions. Although modular code is not graded,    #
      # it is highly recommended.                                                 #
      #############################################################################
      response = ''
      # extract titles and matches
      matches = self.get_possible_matching_titles(line)
      # extract sentiment
      sentiment = self.extract_sentiment(line)

      if self.creative:
        if len(matches) == 0:
          return self.generate_response_to_irrelevant_input()
        else:
          response = "You are talking about: " +\
                     ', '.join('{}'.format(k) for k in matches)
      else:
        if len(matches) == 0:
          return self.generate_response_to_irrelevant_input() 
        elif len(matches) > 1:
          return 'Please tell me about one movie at a time.'
        else:
          title, idxs = matches[0]
          sentiment = self.extract_sentiment(line.replace(title, ''))
          if sentiment == 0:
            return "So did you like {} or hate it? Please tell me.".format(title)
          else:
            if len(idxs) > 1:
              return "I found multiple matches for {}. Can you be more specific? Maybe try telling me the year as well.".format(title)
            elif len(idxs) == 0:
              return "Hmm, I couldn't find a match for {}. Please tell me about some other movies you have watched!".format(title)
            else:
              if sentiment > 0:
                if idxs[0] in self.user_movie_set:
                  response = "I think you already told me about that movie."
                else:
                  response = "Great! So you liked {}. ".format(title)
                  self.process_movie(idxs[0], sentiment)
              else:
                response = "Okay, so you didn't like {}. ".format(title)
            
      # recommend once we have 5 movies
      if len(self.user_movies) >= 5:
        response += ' Recommending movies...'
      else:
        response += " " + self.generate_request_for_more_movies()   
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return response

    def generate_response_to_irrelevant_input(self):
      responses = [
                   "I'm sorry, but I want to hear about a movie you liked.",
                   "That's really cool and all, but can we go back to talking about movies? I want to know more about movies you enjoyed!",
                   "Maybe we can talk about that later. Let's get back to talking about movies. Why don't you tell me what you thought about a movie you watched recently?"
                  ]
      return responses[random.randint(0, len(responses) - 1)]

    def generate_request_for_more_movies(self):
      responses = [
                   "Please tell me about more movies you've watched!",
                   "Tell me another one of your favorite movies. This is so much fun!",
                   "What is another movie you liked?"
                  ]
      return responses[random.randint(0, len(responses) - 1)]

    def get_possible_matching_titles(self, line):
      possible_titles = self.extract_titles(line)
      matches = []
      if self.creative:
        self.prefix_match_found = False
        for title in possible_titles:
          movie_idxs = self.find_movies_by_title(title)
          if not self.prefix_match_found:
            movie_idxs.extend(self.find_movies_closest_to_title(title, max_distance=3))
            movie_idxs = sorted(list(set(movie_idxs)))
          matches.append((title, movie_idxs))
      else: 
        for title in possible_titles:
          matches.append((title, self.find_movies_by_title(title)))
      return matches

    def process_movie(self, movie_index, sentiment):
      self.user_movies.append((movie_index, sentiment))
      self.user_movie_set.add(movie_index)
      self.current_movie = None
      self.current_sentiment = None

    def extract_titles(self, text):
      """Extract potential movie titles from a line of text.

      Given an input text, this method should return a list of movie titles
      that are potentially in the text.

      - If there are no movie titles in the text, return an empty list.
      - If there is exactly one movie title in the text, return a list
      containing just that one movie title.
      - If there are multiple movie titles in the text, return a list
      of all movie titles you've extracted from the text.

      Example:
        potential_titles = chatbot.extract_titles('I liked "The Notebook" a lot.')
        print(potential_titles) // prints ["The Notebook"]

      :param text: a user-supplied line of text that may contain movie titles
      :returns: list of movie titles that are potentially in the text
      """
      potential_titles = []
      if self.creative:
        pat1 = '"(.*?)"'
        #pat2 = '((?:[A-HJ-Z0-9][^\s]*|I [A-Z0-9])(?:.*[A-Z0-9][^\s]*)?\s*(?:\(\d{4}\))?)'
        stop_words = 'at|as|of|on|to|with|and|the|in|from|&|\+|by|or|de|vs\.'
        pat2 = '((?:[A-HJ-Z0-9][^\s]*(?:\s+(?:[A-Z0-9.\-\(][^\s]*|' + stop_words + ')|$)|I [A-Z0-9])(?:.*[A-Z0-9][^\s]*)?\s*(?:\(\d{4}\))?)'
        potential_titles = re.findall(pat1, text)
        potential_titles.extend(re.findall(pat2, text))
        potential_titles = list(set(potential_titles))
      else:
        potential_titles = re.findall('"(.*?)"', text)
      return potential_titles

    def find_movies_by_title(self, title):
      """ Given a movie title, return a list of indices of matching movies.

      - If no movies are found that match the given title, return an empty list.
      - If multiple movies are found that match the given title, return a list
      containing all of the indices of these matching movies.
      - If exactly one movie is found that matches the given title, return a list
      that contains the index of that matching movie.

      Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]

      :param title: a string containing a movie title
      :returns: a list of indices of matching movies
      """
      candidates = []
      if self.creative:
        movie = extract_titles_and_year(title)
        for i in range(len(self.movies)):
          match_found = False
          for mt in self.movies[i].titles:
            for t in movie.titles:
              # if database title starts with query title
              if bool(re.match(t + '($|\s)', mt, re.I)):
                match_found = True
                self.prefix_match_found = True
                break
            if match_found:
              break
          if match_found:
            # if no year included in query, add all movies that match
            if not movie.year:
              candidates.append(i)
            # if year included in query, add only movies that match both
            # title AND year
            if movie.year and movie.year == self.movies[i].year:
              candidates.append(i)
              self.prefix_match_found = True
      else:
        movie = extract_titles_and_year(title)
        for i in range(len(self.movies)):
          if set(movie.titles).intersection(set(self.movies[i].titles)):
            if not movie.year:
              candidates.append(i)
            elif movie.year and movie.year == self.movies[i].year:
              candidates.append(i)
              return candidates
      return candidates


    def extract_sentiment(self, text):
      """Extract a sentiment rating from a line of text.

      You should return -1 if the sentiment of the text is negative, 0 if the
      sentiment of the text is neutral (no sentiment detected), or +1 if the
      sentiment of the text is positive.

      As an optional creative extension, return -2 if the sentiment of the text
      is super negative and +2 if the sentiment of the text is super positive.

      Example:
        sentiment = chatbot.extract_sentiment('I liked "The Titanic"')
        print(sentiment) // prints 1

      :param text: a user-supplied line of text
      :returns: a numerical value for the sentiment of the text
      """
      return 1

    def extract_sentiment_for_movies(self, text):
      """Creative Feature: Extracts the sentiments from a line of text
      that may contain multiple movies. Note that the sentiments toward
      the movies may be different.

      You should use the same sentiment values as extract_sentiment, described above.
      Hint: feel free to call previously defined functions to implement this.

      Example:
        sentiments = chatbot.extract_sentiment_for_text('I liked both "Titanic (1997)" and "Ex Machina".')
        print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

      :param text: a user-supplied line of text
      :returns: a list of tuples, where the first item in the tuple is a movie title,
        and the second is the sentiment in the text toward that movie
      """
      pass

    def find_movies_closest_to_title(self, title, max_distance=3):
      """Creative Feature: Given a potentially misspelled movie title,
      return a list of the movies in the dataset whose titles have the least edit distance
      from the provided title, and with edit distance at most max_distance.

      - If no movies have titles within max_distance of the provided title, return an empty list.
      - Otherwise, if there's a movie closer in edit distance to the given title 
        than all other movies, return a 1-element list containing its index.
      - If there is a tie for closest movie, return a list with the indices of all movies
        tying for minimum edit distance to the given movie.

      Example:
        chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

      :param title: a potentially misspelled title
      :param max_distance: the maximum edit distance to search for
      :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
      """
      candidates = []
      movie = extract_titles_and_year(title)
      for i in range(len(self.movies)):
        match_found = False
        #full_title = self.movies[i].titles[0]
        for mt in self.movies[i].titles:  
          for t in movie.titles:
            dist = edit_distance(t, mt)#full_title)
            if dist <= max_distance:
              match_found = True
              # if distance is smaller than all previous, discard previous
              if dist < max_distance:
                candidates = []
                max_distance = dist
              break
          if match_found:
            break
        if match_found:
          if not movie.year:
            candidates.append(i)
          if movie.year and movie.year == self.movies[i].year:
            candidates.append(i)
            return candidates
      return candidates

    def disambiguate(self, clarification, candidates):
      """Creative Feature: Given a list of movies that the user could be talking about 
      (represented as indices), and a string given by the user as clarification 
      (eg. in response to your bot saying "Which movie did you mean: Titanic (1953) 
      or Titanic (1997)?"), use the clarification to narrow down the list and return 
      a smaller list of candidates (hopefully just 1!)

      - If the clarification uniquely identifies one of the movies, this should return a 1-element
      list with the index of that movie.
      - If it's unclear which movie the user means by the clarification, it should return a list
      with the indices it could be referring to (to continue the disambiguation dialogue).

      Example:
        chatbot.disambiguate("1997", [1359, 2716]) should return [1359]
      
      :param clarification: user input intended to disambiguate between the given movies
      :param candidates: a list of movie indices
      :returns: a list of indices corresponding to the movies identified by the clarification
      """
      pass


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def binarize(self, ratings, threshold=2.5):
      """Return a binarized version of the given matrix.

      To binarize a matrix, replace all entries above the threshold with 1.
      and replace all entries at or below the threshold with a -1.

      Entries whose values are 0 represent null values and should remain at 0.

      :param x: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
      :param threshold: Numerical rating above which ratings are considered positive

      :returns: a binarized version of the movie-rating matrix
      """
      #############################################################################
      # TODO: Binarize the supplied ratings matrix.                               #
      #############################################################################

      # The starter code returns a new matrix shaped like ratings but full of zeros.
      binarized_ratings = np.zeros_like(ratings)

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return binarized_ratings


    def similarity(self, u, v):
      """Calculate the cosine similarity between two vectors.

      You may assume that the two arguments have the same shape.

      :param u: one vector, as a 1D numpy array
      :param v: another vector, as a 1D numpy array

      :returns: the cosine similarity between the two vectors
      """
      #############################################################################
      # TODO: Compute cosine similarity between the two vectors.
      #############################################################################
      similarity = 0
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return similarity


    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
      """Generate a list of indices of movies to recommend using collaborative filtering.

      You should return a collection of `k` indices of movies recommendations.

      As a precondition, user_ratings and ratings_matrix are both binarized.

      Remember to exclude movies the user has already rated!

      :param user_ratings: a binarized 1D numpy array of the user's movie ratings
      :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
        `ratings_matrix[i, j]` is the rating for movie i by user j
      :param k: the number of recommendations to generate
      :param creative: whether the chatbot is in creative mode

      :returns: a list of k movie indices corresponding to movies in ratings_matrix,
        in descending order of recommendation
      """

      #######################################################################################
      # TODO: Implement a recommendation function that takes a vector user_ratings          #
      # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
      #                                                                                     #
      # For starter mode, you should use item-item collaborative filtering                  #
      # with cosine similarity, no mean-centering, and no normalization of scores.          #
      #######################################################################################

      # Populate this list with k movie indices to recommend to the user.
      recommendations = []

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return recommendations


    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
      """Return debug information as a string for the line string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      debug_info = 'debug info'
      return debug_info


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      """Return a string to use as your chatbot's description for the user.

      Consider adding to this description any information about what your chatbot
      can do and how the user can interact with it.
      """
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')
