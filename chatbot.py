import numpy as np
import argparse
import joblib
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util

class Chatbot:
    """Class that implements the chatbot for HW 6."""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = "CinemAI" # TODO: Give your chatbot a new name.

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')

        # Load sentiment words
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        self.count_vectorizer = None
        self.model = None
        # Train the classifier
        self.train_logreg_sentiment_classifier() # stores in model and count_vectorizer

        # TODO: put any other class variables you need here
        self.user_ratings = {}

        self.state = 'input' # ['input', 'clarification', 'recommendation']
        self.clarification_storage = ([], 0) # movies we're choosing from, sentiment already expressed
        self.recommendation_storage = [] # list of recs we haven't yet given

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return "Hi, I'm CinemAI, your friendly neighborhood movie chatbot! \n\
            I can provide movie recommendations based on the movies you like or dislike. \n\
            You can write \":quit\" or press Ctrl-C anytime to stop."

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Let's get started! Tell me some of your preferences."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Bye bye!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function.
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################
    def rec(self, line: str) -> str:
        if line.lower() == 'yes':
            if len(self.recommendation_storage) > 0:
                response = f"Wonderful! I suggest you watch {self.recommendation_storage[0]}. Would you like another recommendation? [yes or :quit] "
                self.recommendation_storage.pop(0)
            else:
                response = "Sorry, that was all the recs I had! :'( Enter ':quit' to quit."
        else:
            response = f"Enter 'yes' to receive a recommendation or ':quit' to quit."

        return response

    def process(self, line: str) -> str:
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        Arguments:
            - line (str): a user-supplied line of text

        Returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        # response = "I (the chatbot) processed '{}'".format(line)


        if self.state == 'recommendation':
            return self.rec(line)

        if self.state == 'input':
            titles = self.extract_titles(line) # list of all movies in the line
            # after extracting title attempt to autocorrect
            line = self.spellcheck(line)
            sentiment = self.predict_sentiment_statistical(line)
            if len(titles) == 0:
                return "Sorry, I did not detect any movie titles in your response. \nPlease check your spelling and make sure to include the title in quotations."
            else:
                title = titles[0] # just look at the first one for now
                indices = self.find_movies_idx_by_title(title)


        elif self.state == 'clarification':
            indices = self.disambiguate_candidates(line, self.clarification_storage[0])
            sentiment = self.clarification_storage[1] # TODO: mess up if have to clarify

        if sentiment == 1:
            feeling = "liked"
        elif sentiment == -1:
            feeling = "disliked"
        else:
            feeling = "neutral"


        if len(indices) == 0:
            response = f"Sorry, I am not familiar with the movie '{title}'"
        elif len(indices) == 1:
            title = self.titles[indices[0]][0]
            if feeling == "neutral":
                response = f"I'm sorry, I'm not sure how you feel about '{title}'. Please try again and be more descriptive."
            else:
                response = f"I see you {feeling} the movie '{title}'. \nIf this is not correct, please try again and be more descriptive.\nIf it is correct, great! Please tell me about another movie: "
            self.user_ratings[indices[0]] = sentiment

            if len(self.user_ratings) >= 5:
                recs = self.recommend_movies(self.user_ratings, 5)
                print("recs: ", recs)
                response += f"\nThanks! That's enough information for me to make a recommendation.\
                I recommend you watch {recs[0]}. Would you like another recommendation? [yes or :quit] "
                self.state = 'recommendation'
                self.recommendation_storage = recs[1:]
            else:
                self.state = 'input'
        else:

            movies = self.titles[indices[0]][0]
            for index in indices[1:]:
                movies += f" or {self.titles[index][0]}"

            self.state = 'clarification'
            self.clarification_storage = (indices, sentiment)
            response = f"That could refer to {movies}. Can you please clarify?"


        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    def extract_titles(self, user_input: str) -> list:
        """Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]

        Arguments:
            - user_input (str) : a user-supplied line of text

        Returns:
            - (list) movie titles that are potentially in the text

        Hints:
            - What regular expressions would be helpful here?
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        regex = "\"([^\"]*)\""
        return re.findall(regex, user_input)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def find_movies_idx_by_title(self, title:str) -> list:
        """ Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title

        Returns:
            - a list of indices of matching movies

        Hints:
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful:
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more than that try to think
              of a more concise approach
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        return self.titles_with_the_and_a(title)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def disambiguate_candidates(self, clarification:str, candidates:list) -> list:
        """Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)


        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this
        should return multiple elements in the list which the clarification could
        be referring to.

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              movieboth> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)"
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '

        Arguments:
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns:
            - a list of indices corresponding to the movies identified by the clarification

        Hints:
            - You should use self.titles somewhere in this function
            - You might find one or more of the following helpful:
              re.search, re.findall, re.match, re.escape, re.compile
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################

        dates = re.findall("(\d{4})", clarification)

        titles = self.extract_titles(clarification)

        if len(dates) > 0:
            candidates = [candidate for candidate in candidates if dates[0] in self.titles[candidate][0]]

        if len(candidates) > 1 and len(titles) > 0:
            movies = [candidate for candidate in candidates if titles[0] in self.titles[candidate][0].lower()]
            if len(movies) > 0:
                candidates = movies

        if len(candidates) > 1:
            movies = [candidate for candidate in candidates if clarification.lower() in self.titles[candidate][0].lower()]
            if len(movies) > 0:
                candidates = movies

        return candidates
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 3. Sentiment                                                             #
    ###########################################################################

    def tokenize(self, user_input: str) -> list:
        regex = r"[A-Za-z]+|\$[\d\.]+|\S+"
        return re.findall(regex, user_input.lower())

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to
        predict sentiment.

        Use the sentiment words from data/sentiment.txt which we have already loaded for you in self.sentiment.
        Then count the number of tokens that are in the positive sentiment category (pos_tok_count)
        and negative sentiment category (neg_tok_count)

        This function should return
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neutral): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1

        Arguments:
            - user_input (str) : a user-supplied line of text
        Returns:
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints:
            - Take a look at self.sentiment (e.g. in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################

        # remove movie title from input
        split = re.split("\"", user_input)
        if len(split) > 2:
            user_input = split[0] + split[2]

        # get list of tokens from user input
        tokens = self.tokenize(user_input)
        counter = 0
        for token in tokens:
            if token in self.sentiment:
                if self.sentiment[token] == "neg":
                    counter -= 1
                else:
                    counter += 1
        if counter < 0:
            return -1
        elif counter == 0:
            return 0
        else:
            return 1

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset

        You'll have to transform the class labels (y) such that:
            -1 inputed into sklearn corresponds to "rotten" in the dataset
            +1 inputed into sklearn correspond to "fresh" in the dataset

        To run call on the command line:
            python3 chatbot.py --train_logreg_sentiment

        Hints:
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Review how you used sklearn to train a logistic regression classifier for HW 5.
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset.
        """
        #load training data
        texts, y = util.load_rotten_tomatoes_dataset()

        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################

        y = np.array([1 if elem == "Fresh" else -1 for elem in y])
        self.count_vectorizer =  CountVectorizer(stop_words='english', max_features = 1000, min_df = 20)
        # transform texts into nparray
        X = self.count_vectorizer.fit_transform(texts).toarray()

        self.model = sklearn.linear_model.LogisticRegression(penalty=None, max_iter = 1000)
        self.model.fit(X, y)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def predict_sentiment_statistical(self, user_input: str) -> int:
        """ Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments:
            - user_input (str) : a user-supplied line of text
        Returns: int
            -1 if the trained classifier predicts -1
            1 if the trained classifier predicts 1
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints:
            - Be sure to lower-case the user input
            - Don't forget about a case for the 0 class!
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################

        # remove movie title from input
        split = re.split("\"", user_input)
        if len(split) > 2:
            user_input = split[0] + split[2]

        text = [user_input.lower()]
        X = self.count_vectorizer.transform(text).toarray()
        for elem in X[0]:
            # only use predict if there are any words in our vocab
            if elem != 0:
                return self.model.predict(X)[0]

        # return 0 if no words in our vocab
        return 0
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the
        recommended movie titles.

        Be sure to call util.recommend() which has implemented collaborative
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.

        This function must have at least 5 ratings to make a recommendation.

        Arguments:
            - user_ratings (dict):
                - keys are indices of movies
                  (corresponding to rows in both data/movies.txt and data/ratings.txt)
                - values are 1, 0, and -1 corresponding to positive, neutral, and
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example:
            bot_recommends = chatbot.recommend_movie({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)',
            'Problem Child (1990)']

        Hints:
            - You should be using self.ratings somewhere in this function
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing.
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        user_ratings_full = [0] * self.ratings.shape[0] # number of movies
        for index in user_ratings:
            user_ratings_full[index] = user_ratings[index]
        user_ratings_full = np.array(user_ratings_full)
        rec_ind = util.recommend(user_ratings_full, self.ratings, num_return)
        res = []
        for index in rec_ind:
            res.append(self.titles[index][0])
        return res
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def titles_with_the_and_a(self, title:str) -> list:
        """
        This function can extract titles that have "the" or "a/an" at the beginning.
        """
        pattern = "([Tt]he\s|[Aa]n\s|[Aa]\s)?(.*)"
        processed_title = re.findall(pattern, title.lower())[0][1].strip()



        movie_list = [i for i in range(len(self.titles)) if processed_title == self.titles[i][0].lower()]

        if len(movie_list) > 0:
            return movie_list

        movie_list = [i for i in range(len(self.titles)) if processed_title == re.split("\s\(\d", self.titles[i][0].lower())[0]]
        if len(movie_list) > 0:
            return movie_list

        return [i for i in range(len(self.titles)) if processed_title in self.titles[i][0].lower()]

    def spellcheck(self, line: str) -> str:
        """
        TODO: delete and replace with your function.
        Be sure to put an adequate description in this docstring.
        """
        from spellchecker import SpellChecker
        spell = SpellChecker()
        tok_line = self.tokenize(line)
        misspelled = spell.unknown(tok_line)
        for i in range(len(tok_line)):
            if tok_line[i] in misspelled:
                tok_line[i] = spell.correction(tok_line[i])
        return ' '.join(tok_line)

    def function3():
        """
        Any additional functions beyond two count towards extra credit
        """
        pass


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
