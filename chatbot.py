# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens

import numpy as np
import re
import random
import nltk
from PorterStemmer import PorterStemmer


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

def contains_anaphoric_expression(text):
  '''
  Checks a few patterns to make sure user text contains an anaphoric expression
  '''
  pat1 = '(\W|^)(it|the movie|that)(\W|$)'
  return bool(re.search(pat1, text, re.I))

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
      self.new_sentiment = {}
      self.p = PorterStemmer()

      # create a new sentiment dict with stemmed keys
      for key in self.sentiment:
        new_key = self.p.stem(key)
        self.new_sentiment[new_key] = self.sentiment[key]

      self.bin_ratings = self.binarize(ratings)

      # a tuple with the sentiment of the movie being discussed
      self.current_sentiment = None
      # the movie title entered by the user
      self.current_title = None
      # a list of current movie candidates
      self.current_idxs = []


      self.prev_movie = None
      self.prev_sentiment = None

      # a dict where dict[i] = j is the user's sentiment j for movie index i 
      # for movies that the user has described and the chatbot has processed
      self.user_movies = {}

      # a set of movie indexes that the user has already described
      self.user_movie_set = set()

      self.prefix_match_found = False
      self.disambig = False

      # if chatbot is in recommend mode, only respond to yes or no
      self.recommend_mode = False

      # a list of recommendations for the user
      self.recommendations = []
      self.recommend_idx = 0

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

      greeting_message = "Hi there! I'm Movie Chatbot. How can I help you?"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Have a nice day! It was fun talking to you!"

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

      swear_response = self.checkSwearWords(line)
      if swear_response:
        return swear_response
      caps_lock_response = self.checkAnger(line)
      if caps_lock_response:
        return caps_lock_response

      if self.recommend_mode:
        if re.match('yes', line.strip(), re.I):
          return self.give_recommendation()
        elif re.match('no', line.strip(), re.I):
          return "Okay, I guess I've given you enough recommendations!"
        else:
          return "Let's talk about that later. Do you want another recommendation?"

      clarification = False
      if self.creative:
        # deal with "Can you...?", "What is...?", etc. questions
        response_to_question = self.matches_question(line)
        if response_to_question:
          return response_to_question
        elif self.disambig:
          self.current_idxs = self.disambiguate(line, self.current_idxs)
          #print(self.current_idxs)
          if len(self.current_idxs) == 1:
            self.current_title = self.titles[self.current_idxs[0]][0]
            self.disambig = False
            clarification = True
          else:
            response = "Sorry, can you be a little more specific? I still found the following movies:\n"
            for i in self.current_idxs:
              response += "{}\n".format(self.titles[i][0])
            return response

      # extract titles and matches
      extracted_title_from_current_line = False
      if not self.current_title:
        matches = self.get_possible_matching_titles(line)
        extracted_title_from_current_line = True
        #print('Extracted title')
      else:
        matches = [(self.current_title, self.current_idxs)]
      #print('Current title:{}'.format(self.current_title))
      # extract sentiment
      extracted_sentiment_from_current_line = False
      if not self.current_sentiment:
        # remove title from line for sentiment extraction
        if matches:
          line = line.replace(matches[0][0], '')
        sentiment = self.extract_sentiment(line)
        extracted_sentiment_from_current_line = True
        self.current_sentiment = sentiment
        #print('Extracted sentiment')
      else:
        sentiment = self.current_sentiment
      #print('Current sentiment:{}'.format(self.current_sentiment))

      if self.creative:
        if not extracted_title_from_current_line and \
           extracted_sentiment_from_current_line:
           if not clarification and not contains_anaphoric_expression(line):
             #print('no anaphoric expression')
             return self.generate_response_to_irrelevant_input()

      if self.creative:
        if len(matches) == 0 and not self.current_title:
            return self.generate_response_to_irrelevant_input()
        elif len(matches) > 1:
          return 'Please tell me about one movie at a time.'
        elif len(matches) == 1:
          title, idxs = matches[0]
          self.current_idxs = idxs
          self.current_title = title
          if len(idxs) == 0:
            self.clear_current_movie()
            return "Hmm, I couldn't find a match for \"{}\". Please tell me about some other movies you have watched!".format(title)
          elif len(idxs) == 1:
            if idxs[0] in self.user_movie_set:
              response = "(I think you already told me about that movie, but I'll update what you tell me!)\n"
            
            if sentiment == 0:
              return response + "I'm a little confused. What did you think about \"{}\"?".format(self.titles[idxs[0]][0])
            if sentiment == 1:
              response += "Great, so you liked \"{}\".".format(self.titles[idxs[0]][0])
            elif sentiment == 2:
              response += "Wow, you really loved \"{}\"!".format(self.titles[idxs[0]][0])
            elif sentiment == -1:
              response += "Okay, you didn't like \"{}\".".format(self.titles[idxs[0]][0])
            elif sentiment == -2:
              response += "It seems like you hated \"{}\" with a passion! That's too bad.".format(self.titles[idxs[0]][0])
            self.process_movie(idxs[0], sentiment)
          else:
            response = "I found multiple movies. Which one are you talking about?\n"
            for i in idxs:
              response += '{}\n'.format(self.titles[i][0])
            self.disambig = True
            return response
      else:
        if len(matches) == 0:
          return self.generate_response_to_irrelevant_input() 
        elif len(matches) > 1:
          return 'Please tell me about one movie at a time.'
        else:
          title, idxs = matches[0]
          sentiment = self.extract_sentiment(line.replace(title, ''))
          if sentiment == 0:
            return "So did you like \"{}\" or hate it? Please tell me.".format(self.titles[idxs[0]][0])
          else:
            if len(idxs) > 1:
              return "I found multiple matches for \"{}\". Can you be more specific? Maybe try telling me the year as well.".format(title)
            elif len(idxs) == 0:
              return "Hmm, I couldn't find a match for \"{}\". Please tell me about some other movies you have watched!".format(title)
            else:
              if sentiment > 0:
                if idxs[0] in self.user_movie_set:
                  response = "(I think you already told me about that movie, but I'll update what you tell me!)\n"
                else:
                  response = "Great! So you liked \"{}\". ".format(self.titles[idxs[0]][0])
                self.process_movie(idxs[0], sentiment)
              elif sentiment < 0:
                if idxs[0] in self.user_movie_set:
                  response = "I think you already told me about that movie."
                else:
                  response = "Okay, so you didn't like \"{}\". ".format(self.titles[idxs[0]][0])
                  self.process_movie(idxs[0], sentiment)
              else:
                return "I'm not sure if you liked or didn't the movie. Can you tell me a movie and what you thought about it?"
            
      # recommend once we have 5 movies
      if len(self.user_movies) >= 5:
        self.recommend_mode = True
        user_ratings = np.zeros(len(self.titles))
        for m in self.user_movies:
          user_ratings[m] = self.user_movies[m]
        self.recommendations = self.recommend(user_ratings, self.bin_ratings, k=10, creative=self.creative)
        self.recommend_idx = 0
        return self.give_recommendation()
      else:
        response += " " + self.generate_request_for_more_movies()   
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return response
    
    def give_recommendation(self):
      recommend_sentences = ["Why don't you check out \"{}\"? ",
                             "I think you might enjoy \"{}\"! ",
                             "\"{}\" might suit your tastes! "
                             ]
      if self.recommend_idx < len(self.recommendations):
        response = ''
        if self.recommend_idx == 0:
          response += "Okay, based on what you told me, I think you would like \"{}\"! ".format(self.titles[self.recommendations[self.recommend_idx]][0])
        else:
          response += random.choice(recommend_sentences).format(self.titles[self.recommendations[self.recommend_idx]][0])
        response += 'Would you like another recommendation?'
        self.recommend_idx += 1
      else:
        response = "Sorry, I don't have any more recommendations!"

      return response

    def matches_question(self, text):
      '''
      Returns response to question
      '''
      question_responses = [
                            "I don't know. Ask Google.",
                            "I'd like to know as well.",
                            "Let me think about that. I'll get back to you in a billion years."
                           ] 
      match = re.findall('(.*)\?', text, re.I)
      if match:
        return self.flip_question(text) + ' ' + random.choice(question_responses) 
      else:
        return None      

    def flip_question(self, text):
      '''
      Flips the perspective of the question
      '''
      table = {
               'I': 'you',
               'me': 'you',
               'my': 'your',
               'your': 'my',
               'myself': 'yourself',
               'yourself': 'myself'
              }
      # some common prepositions
      prep_set = {'of','with','at','from','including','until','against',
                  'among','towards','upon', 'to'}
      words = re.split('\s|\?', text)
      words.pop()# remove empty string at end
      last_word = None
      for i in range(len(words)):
        if words[i] in table:
          words[i] = table[words[i]]
        elif words[i] == 'you' or words[i] == 'You':
          if last_word in prep_set:
            words[i] = 'me'
          else:
            words[i] = 'I'
        last_word = words[i]
      return ' '.join(words) + '?'

    def generate_response_to_irrelevant_input(self):
      responses = [
                   "I'm sorry, but I want to hear about a movie you liked.",
                   "That's really cool and all, but can we go back to talking about movies? I want to know more about movies you enjoyed!",
                   "Maybe we can talk about that later. Let's get back to talking about movies. Why don't you tell me what you thought about a movie you watched recently?"
                  ]
      return random.choice(responses)

    def generate_request_for_more_movies(self):
      responses = [
                   "Please tell me about more movies you've watched!",
                   "Tell me another one of your favorite movies. This is so much fun!",
                   "What is another movie you liked?"
                  ]
      return random.choice(responses)

    def get_possible_matching_titles(self, line):
      possible_titles = self.extract_titles(line)
      matches = []
      if self.creative:
        self.prefix_match_found = False
        for title in possible_titles:
          movie_idxs = self.find_movies_by_title(title)
          #print(movie_idxs)
          if not self.prefix_match_found:
            movie_idxs.extend(self.find_movies_closest_to_title(title, max_distance=3))
            #print(movie_idxs)
            movie_idxs = sorted(list(set(movie_idxs)))
          matches.append((title, movie_idxs))
      else: 
        for title in possible_titles:
          matches.append((title, self.find_movies_by_title(title)))
      return matches

    def process_movie(self, movie_index, sentiment):
      self.user_movies[movie_index] = sentiment
      self.user_movie_set.add(movie_index)
      self.prev_idx = movie_index
      self.prev_sentiment = self.current_sentiment
      self.clear_current_movie()    

    def clear_current_movie(self):
      self.current_sentiment = None
      self.current_title = None
      self.current_idxs = None

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
        stop_words = 'at|as|of|on|to|with|and|the|in|from|&|\+|by|or|de|vs\.'
        pat2 = '((?:[A-HJ-Z0-9]\S*(?:\s+(?:[A-Z0-9\.\-\(]\S*|' + stop_words + ')?|$)|I [A-Z0-9])(?:.*[A-HJ-Z0-9]\S*|.*[A-Z]\S+)?\s*(?:\(\d{4}\))?)'
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
          for dbt in self.movies[i].titles:
            for qt in movie.titles:
              # if database title starts with query title
              if bool(re.match(qt + '($|\W)', dbt, re.I)):
                match_found = True
                break
            if match_found:
              break
          if match_found:
            # if no year included in query, add all movies that match
            if not movie.year:
              candidates.append(i)
              self.prefix_match_found = True
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

    class Example:
      """Represents a document with a label. klass is 'pos' or 'neg' by convention.
          words is a list of strings.
      """
      def __init__(self):
          self.klass = ''
          self.words = []

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

      #process train data 

      negationSet = {"n't", "never", "not", "no"}
      strongerSet = {"really", "very", "love", "hate", "terrible", "truly",
                     "despise", "great", "fantastic", "amazing", "extremely",
                     "horrible", "disgusting", "stunning", "adore"}
      punct = "\W+"
      
      newSet = set()
      for word in negationSet:
        newSet.add(self.p.stem(word))
      negationSet = newSet

      newSet = set()
      for word in strongerSet:
        newSet.add(self.p.stem(word))
      strongerSet = newSet
      

      textWords = nltk.word_tokenize(text)
       
      opp = False
      pos_num = 0
      neg_num = 0
      strength_val = 1
      num_sentiment_words = 0

      for word in textWords:
        word = self.p.stem(word)
        if word in negationSet:
          opp = True
          continue
        if re.match(punct, word):
          opp = False
          strength_val = 1
          continue
        if word in strongerSet:
          strength_val = 2
        if word in self.new_sentiment:
          if self.new_sentiment[word] == 'pos' and not opp:
            pos_num += strength_val
          elif self.new_sentiment[word] == 'pos' and opp:
            neg_num += strength_val
          elif self.new_sentiment[word] == 'neg' and not opp:
            neg_num += strength_val
          else:
            pos_num += strength_val
          num_sentiment_words += 1

      thresh = 0.25
      if num_sentiment_words == 0:
        sentiment = 0
      else:
        avg = (pos_num - neg_num)/float(num_sentiment_words)
        if avg > 1:
          sentiment = 2
        elif thresh < avg <= 1:
          sentiment = 1
        elif -thresh <= avg <= thresh:
          sentiment = 0
        elif -1 <= avg < -thresh:
          sentiment = -1
        else:
          sentiment = -2

        if not self.creative:
          if sentiment > 1:
            sentiment = 1
          elif sentiment < -1:
            sentiment = -1

      return sentiment

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
        for dbt in self.movies[i].titles:  
          for qt in movie.titles:
            dist = edit_distance(qt, dbt)
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
      filtered_idxs = []
      for idx in candidates:
        if bool(re.search('(\W|^)' + clarification + '(\W|$)', self.titles[idx][0], 
                          re.I)):
          filtered_idxs.append(idx)
      # try looking for phrases like 'first one' or '2nd movie'
      if not filtered_idxs:
        if bool(re.search('(\W|^)(first|1st)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 1:
          filtered_idxs = [candidates[0]]
        elif bool(re.search('(\W|^)(second|2nd)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 2:
          filtered_idxs = [candidates[1]]
        elif bool(re.search('(\W|^)(third|3rd)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 3:
          filtered_idxs = [candidates[2]]
        elif bool(re.search('(\W|^)(fourth|4th)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 4:
          filtered_idxs = [candidates[3]]
        elif bool(re.search('(\W|^)(fifth|5th)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 5:
          filtered_idxs = [candidates[4]]
        elif bool(re.search('(\W|^)(sixth|6th)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 6:
          filtered_idxs = [candidates[5]]
        elif bool(re.search('(\W|^)(seventh|7th)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 7:
          filtered_idxs = [candidates[6]]
        elif bool(re.search('(\W|^)(eighth|8th)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 8:
          filtered_idxs = [candidates[7]]
        elif bool(re.search('(\W|^)(ninth|9th)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 9:
          filtered_idxs = [candidates[8]]
        elif bool(re.search('(\W|^)(tenth|10th)(\W|$)', clarification, re.I)) and \
           len(candidates) >= 10:
          filtered_idxs = [candidates[9]]
      
      if not filtered_idxs:
        return candidates
      else:
        return filtered_idxs


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def binarize(self, ratings, threshold=2.5, creative=False):
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
      if creative: 
        high_thresh = 4
        low_thresh = 5-high_thresh
        binarized_ratings = np.where(ratings >= high_thresh, 2.0, 0.0) + np.where((ratings > threshold) & (ratings < high_thresh),1.0,0.0) + np.where((ratings <= threshold) & (ratings > low_thresh),-1.0,0.0) + np.where((ratings != 0.0) & (ratings <= low_thresh), -2.0, 0.0) 
      else:
        binarized_ratings = np.where(ratings > threshold, 1.0, 0.0) + np.where((ratings != 0.0) & (ratings <= threshold), -1.0, 0.0) 


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
      u_norm = np.linalg.norm(u)
      v_norm = np.linalg.norm(v)
      dot_prod = np.dot(u, v)
      similarity = dot_prod
      if u_norm == 0.0 or v_norm == 0.0:
        return 0.0
      else:
        similarity = float(dot_prod)/(u_norm*v_norm)
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

      unseen_movies = np.where(user_ratings == 0)[0]
      seen_movies = np.where(user_ratings != 0)[0]

      ratings_unseen = []

      for i in unseen_movies:
        unseen_ratings = ratings_matrix[i, :]
        weights = []
        ratings = []
        for j in seen_movies:
          seen_ratings = ratings_matrix[j, :]
          weight = self.similarity(unseen_ratings, seen_ratings)
          weights.append(weight)
          ratings.append(user_ratings[j])
        estimated_rating = float(np.dot(weights, ratings))
        ratings_unseen.append([i, estimated_rating])


      ratings_unseen.sort(key = lambda x:x[1], reverse = True)

      recommendations = []  

      for i in range(k):
        recommendations.append(ratings_unseen[i][0])

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return recommendations


    def checkAnger(self, string):
      response = ''

      words = string.split()
      upperCase = all([(word.isupper()) for word in words])
          
      upperCaseResponses = ["Any reason you are yelling at me?!",
                            "Is your caps lock key stuck or something?",
                            "It looks like you were busy capslocking >_>."]
      if upperCase:    
        return random.choice(upperCaseResponses)
      else:
        return ''

    def checkSwearWords(self, string):
      swearSet = {"fuck", "fucking", "shit", "damn", "bitch", "crap", "piss",
                  "dick", "cock", "pussy","asshole","fag","bastard","slut","douche",
                  "bollocks","arsehole","bloody"}
      words = set(string.lower().split())
      if words & swearSet:
        return 'Wash your mouth with soap!'
      else:
        return ''


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
      Creative Mode:
      1. Identifies movies in quotation marks. Also identifies outside of quotation marks if first word is capitalized correctly and either date is included or last word is correctly capitalized (eg. "The american president (1995)", "The great Escape")
      2. Handles alternate/foreign titles
      3. Handles foreign articles (le, la, les, el, los)
      4. Handles [+&] in title
      5. Disambiguates by matching title, year, or position in given list ("The first one", "The 2nd movie"). Includes dialogue for disambiguation
      6. Extracts fine-grained sentiment by taking the weighted average of sentiment words and using a threshold
      7. Spelling correction for regular/foreign/alternate titles
      8. Understands reference to things said previously. If it asks to clarify sentiment, requires anaphoric expression. Detects duplicate movies you've told it about.
      9. Responds to irrelevant input 
     10. Responds to arbitrary questions. Reflects back questions such as "Can you help me?"-->"Can I help you?"
     11. Responds to anger by detecting swear words or caps lock
      """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')
