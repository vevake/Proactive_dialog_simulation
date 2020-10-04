# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from simdial.domain import Domain, DomainSpec
from simdial.generator import Generator
from simdial import complexity
import string
import sys
import numpy as np

class RestSpec(DomainSpec):
    name = "restaurant"
    greet = "Welcome to restaurant recommendation system."
    nlg_spec = {"area": {"inform": ["I am at %s.", "%s.", "I'm interested in food at %s.", "At %s.", "In %s."],
                        "request": ["Which city are you interested in?", "Which place?"]},

                "food": {"inform": ["I like %s food.", "%s food.", "%s restaurant.", "%s."],
                              "request": ["What kind of food do you like?", "What type of restaurant?"]},

                "open": {"inform": ["The restaurant is %s.", "It is %s right now."],
                         "request": ["Tell me if the restaurant is open.", "What's the hours?"],
                         "yn_question": {'open': ["Is the restaurant open?"],
                                         'closed': ["Is it closed?"]
                                         }},

                "parking": {"inform": ["The restaurant has %s.", "This place has %s."],
                            "request": ["What kind of parking does it have?.", "How easy is it to park?"],
                            "yn_question": {'street parking': ["Does it have street parking?"],
                                            "valet parking": ["Does it have valet parking?"]
                                            }},

                "pricerange": {"inform": ["The restaurant serves %s food.", "The price is %s."],
                          "request": ["What's the average price?", "How expensive it is?"],
                          "yn_question": {'expensive': ["Is it expensive?"],
                                          'moderate': ["Does it have moderate price?"],
                                          'cheap': ["Is it cheap?"]
                                          }},

                "default": {"inform": ["Restaurant %s is a good choice."],
                            "request": ["I need a restaurant.",
                                        "I am looking for a restaurant.",
                                        "Recommend me a place to eat."]}
                }


    usr_slots = [("food", "food preference", ["afghan", "african", "afternoon tea", "asian oriental", "australasian", "australian", "austrian", "barbeque", "basque", "belgian", "bistro", "brazilian", "british", "canapes", "cantonese", "caribbean", "catalan", "chinese", "christmas", "corsica", "creative", "crossover", "cuban", "danish", "eastern european", "english", "eritrean", "european", "french", "fusion", "gastropub", "german", "greek", "halal", "hungarian", "indian", "indonesian", "international", "irish", "italian", "jamaican", "japanese", "korean", "kosher", "latin american", "lebanese", "light bites", "malaysian", "mediterranean", "mexican", "middle eastern", "modern american", "modern eclectic", "modern european", "modern global", "molecular gastronomy", "moroccan", "new zealand", "north african", "north american", "north indian", "northern european", "panasian", "persian", "polish", "polynesian", "portuguese", "romanian", "russian", "scandinavian", "scottish", "seafood", "singaporean", "south african", "south indian", "spanish", "sri lankan", "steakhouse", "swedish", "swiss", "thai", "the americas", "traditional", "turkish", "tuscan", "unusual", "vegetarian", "venetian", "vietnamese", "welsh", "world"]),
                 ("area", "area preference", ["centre", "north", "west", "south", "east"]),
                 ("pricerange", "average price per person", ["cheap", "moderate", "expensive"])]
    
    sys_slots = [("open", "if it's open now", ["open", "closed"]),
                 ("parking", "if it has parking", ["street parking", "valet parking", "no parking"])]

    db_size = 200



if __name__ == "__main__":
    # pipeline here
    # generate a fix 500 test set and 5000 training set.
    # generate them separately so the model can choose a subset for train and
    # test on all the test set to see generalization.
#     advice_prob = sys.argv[1]

    for i in range(10, 11):
        advice_prob = float(i)/10 if i > 0 else 0.0
        test_size = 10
        train_size = 0
        gen_bot = Generator()

        rest_spec = RestSpec()

        # restaurant
        print('\n')
#         print("No. of rows in KB: {}".format(rest_spec.db_size))
        print("Advice accept prob = {}".format(advice_prob))
        gen_bot.gen_corpus("test", rest_spec, complexity.CleanSpec, test_size, advice_prob=advice_prob)
#     gen_bot.gen_corpus("test", rest_spec, complexity.MixSpec, test_size)
#     gen_bot.gen_corpus("train", rest_spec, complexity.CleanSpec, train_size)
#     gen_bot.gen_corpus("train", rest_spec, complexity.MixSpec, train_size)