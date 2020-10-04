# -*- coding: utf-8 -*-
# Author: Tiancheng Zhao
# Date: 9/13/17

import numpy as np
from simdial.agent.core import SystemAct, UserAct, BaseUsrSlot
from simdial.agent import core
import json
import copy


class AbstractNlg(object):
    """
    Abstract class of NLG
    """

    def __init__(self, domain, complexity):
        self.domain = domain
        self.complexity = complexity

    def generate_sent(self, actions, **kwargs):
        """
        Map a list of actions to a string.

        :param actions: a list of actions
        :return: uttearnces in string
        """
        raise NotImplementedError("Generate sent is required for NLG")

    def sample(self, examples):
        return np.random.choice(examples)


class SysCommonNlg(object):
    templates = {SystemAct.GREET: ["Hello.", "Hi.", "Greetings.", "How are you doing?"],
                 SystemAct.ASK_REPEAT: ["Can you please repeat that?", "What did you say?"],
                 SystemAct.ASK_REPHRASE: ["Can you please rephrase that?", "Can you say it in another way?"],
                 SystemAct.GOODBYE: ["Goodbye.", "See you next time."],
                 SystemAct.CLARIFY: ["I didn't catch you."],
                 SystemAct.REQUEST+core.BaseUsrSlot.NEED: ["What can I do for you?",
                                                           "What do you need?",
                                                           "How can I help?"],
                 SystemAct.REQUEST+core.BaseUsrSlot.HAPPY: ["What else can I do?",
                                                            "Are you happy about my answer?",
                                                            "Anything else?"],
                 SystemAct.EXPLICIT_CONFIRM+"dont_care": ["Okay, you dont_care, do you?",
                                                          "You dont_care, right?"],
                 SystemAct.IMPLICIT_CONFIRM+"dont_care": ["Okay, you dont_care.",
                                                          "Alright, dont_care."],
                 SystemAct.RESTART: ["We do not have any matches for you."]}

class SysAdviceNlg(object):
    templates = {":food": ["Most of the restaurants are <FOOD> .",
                          "We have large collection of restaurants such as <FOOD> , <FOOD> and <FOOD> ."],
                 
                 ":area": ["There is a wide selection of restaurants in the <AREA> ."],
                 
                 ":pricerange": ["Most of the restaurants are in <PRICERANGE> pricerange."],
                 
                 
                 "food:area": ["All <FOOD> restaurants are in <AREA> area.",
                               "Most of the <FOOD> restaurants are in <AREA> and <AREA> ."],
                 
                 "food:pricerange": ["All <FOOD> restaurants are in <PRICERANGE> pricerange.",
                                     "Most of the <FOOD> restaurants are in <PRICERANGE> and <PRICERANGE> pricerange."],
                 
                 "area:pricerange": ["All restaurants in <AREA> area are in <PRICERANGE> pricerange.",
                                      "Most of the restaurants in the <AREA> are in <PRICERANGE> and <PRICERANGE> pricerange."],
                 
                 
                 
                 
                 "food:area_pricerange": ["All <FOOD> restaurants are in <AREA> area and are in <PRICERANGE> pricerange.",
                               "Most of the <FOOD> restaurants are in <AREA> and are <PRICERANGE> ."],
                 
                 "area:food_pricerange": ["All restaurants in <AREA> are serve <FOOD> and are <PRICERANGE>.",
                                     "Most of the restaurants in <AREA> are <FOOD> and are <PRICERANGE> ."],
                 
                 "pricerange:food_area": ["All <PRICERANGE> restaurants serve <FOOD> and are in <AREA> .",
                                      "Most of the <PRICERANGE> restaurants serve <FOOD> and are in <AREA> ."],                 
                 
                 
                                                   
                 
                 "area:food": ["All restaurants in <AREA> area are <FOOD> .",
                               "There is a wide range of restaurants in the <AREA> such as <FOOD> , <FOOD> , <FOOD> ."],
                 
                 "pricerange:food": ["All restaurants in <PRICERANGE> pricerange are <FOOD>.",
                                     "Most of the <AREA> restaurants are <FOOD> and <FOOD> ."],
                 
                 "pricerange:area": ["All restaurants in <PRICERANGE> pricerange are in the <AREA> .",
                                      "Most of the {} restaurants are in <AREA> and <AREA> ."],
                 
                 "food_area:pricerange": ["All <FOOD> restaurants in <AREA> area are in <PRICERANGE> pricerange.",
                                          "Most of the <FOOD> restaurants in the <AREA> are in <PRICERANGE> and <PRICERANGE> pricerange."],
                 
                 "area_food:pricerange": ["All <FOOD> restaurants in <AREA> area are <PRICERANGE> .",
                                          "Most of the <FOOD> restaurants in <AREA> are <PRICERANGE> and <PRICERANGE> ."],
                 
                 "area_pricerange:food": ["All <PRICERANGE> restaurants in <AREA> area are <FOOD> .",
                                          "Most of the <PRICERANGE> restaurants in <AREA> are <FOOD> and <FOOD> ."],                 
                 
                 "food_pricerange:area": ["All <PRICERANGE> <FOOD> restaurants are in <AREA> area.",
                                          "Most of the <PRICERANGE> <FOOD> restaurants are in <AREA> and <AREA> ."]}
    
class SysNlg(AbstractNlg):
    """
    NLG class to generate utterances for the system side.
    """

    def generate_sent(self, actions, domain=None, templates=SysCommonNlg.templates, stat_query=([], []), adv_templates=SysAdviceNlg.templates):
        """
         Map a list of system actions to a string.

        :param actions: a list of actions
        :param templates: a common NLG template that uses the default one if not given
        :return: uttearnces in string
        """
        str_actions = []
        lexicalized_actions = []
        for a in actions:
            a_copy = copy.deepcopy(a)
            requested_slot = None
            if a.act == SystemAct.GREET:
                if domain:
                    str_actions.append(domain.greet)
                else:
                    str_actions.append(self.sample(templates[a.act]))

            elif a.act == SystemAct.QUERY:
                usr_constrains = a.parameters[0]
                sys_goals = a.parameters[1]

                # create string list for KB_SEARCH
                search_dict = {}
                for k, v in usr_constrains:
                    slot = self.domain.get_usr_slot(k)
                    if v is None:
                        search_dict[k] = 'dont_care'
                    else:
                        search_dict[k] = slot.vocabulary[v]

                a_copy.parameters[0] = search_dict
                a_copy.parameters[1] = sys_goals
                str_actions.append(json.dumps({"QUERY": search_dict,
                                               "GOALS": sys_goals}))

            elif a.act == SystemAct.INFORM:
                sys_goals = a.parameters[1]

                # create string list for RET + Informs
                informs = []
                sys_goal_dict = {}
                for k, (v, e_v) in sys_goals.items():
                    slot = self.domain.get_sys_slot(k)
                    sys_goal_dict[k] = slot.vocabulary[v]

                    if e_v is not None:
                        prefix = "Yes, " if v == e_v else "No, "
                    else:
                        prefix = ""
                    informs.append(prefix + slot.sample_inform()
                                   % slot.vocabulary[v])
                a_copy['parameters'] = [sys_goal_dict]
                str_actions.append(" ".join(informs))

            elif a.act == SystemAct.REQUEST:
                slot_type, _ = a.parameters[0]
                if slot_type in [core.BaseUsrSlot.NEED, core.BaseUsrSlot.HAPPY]:
                    str_actions.append(self.sample(templates[SystemAct.REQUEST+slot_type]))
                else:
                    target_slot = self.domain.get_usr_slot(slot_type)
                    requested_slot = slot_type
                    if target_slot is None:
                        raise ValueError("none slot %s" % slot_type)
                    str_actions.append(target_slot.sample_request())

            elif a.act == SystemAct.EXPLICIT_CONFIRM:
                slot_type, slot_val = a.parameters[0]
                if slot_val is None:
                    str_actions.append(self.sample(templates[SystemAct.EXPLICIT_CONFIRM+"dont_care"]))
                    a_copy.parameters[0] = (slot_type, "dont_care")
                else:
                    slot = self.domain.get_usr_slot(slot_type)
                    str_actions.append("Do you mean %s?"
                                       % slot.vocabulary[slot_val])
                    a_copy.parameters[0] = (slot_type, slot.vocabulary[slot_val])

            elif a.act == SystemAct.IMPLICIT_CONFIRM:
                slot_type, slot_val = a.parameters[0]
                if slot_val is None:
                    str_actions.append(self.sample(templates[SystemAct.IMPLICIT_CONFIRM+"dont_care"]))
                    a_copy.parameters[0] = (slot_type, "dont_care")
                else:
                    slot = self.domain.get_usr_slot(slot_type)
                    str_actions.append("I believe you said %s."
                                       % slot.vocabulary[slot_val])
                    a_copy.parameters[0] = (slot_type, slot.vocabulary[slot_val])

            elif a.act in templates.keys():
                str_actions.append(self.sample(templates[a.act]))

            else:
                raise ValueError("Unknown dialog act %s" % a.act)

            lexicalized_actions.append(a_copy)
        
        stat_query, advice = stat_query
#         print(stat_query, advice)
        suggestion = {}
        def add_to_suggestion(sug, i):
            if i==0:
                if "#food" not in sug:
                    sug["#food"] = c
            elif i==1:
                if "#area" not in sug:
                    sug["#area"] = c
            elif i==2:
                if "#pricerange" not in sug:
                    sug["#pricerange"] = c
            return sug
        
        advice_nlg = ''
        names = ['food', 'area', 'pricerange']
        given = []            
        t = ''        
        for i, q in enumerate(stat_query):
            if q is not None:
                 given.append(i)

        if len(given)==0 and len(advice)>0:         
            #if no constraint is imposed
            to_convey = 0
#             if target_slot is not None:
#                 from_ad = 
            from_ad = np.random.choice(len(advice))
            if requested_slot is not None:
                if requested_slot == '#food':
                    ad = [i for i, ad in enumerate(advice) if ad[0]!='']
                    from_ad = ad[0]
                elif requested_slot == '#area':
                    ad = [i for i, ad in enumerate(advice) if ad[1]!='']
                    from_ad = ad[0]
                elif requested_slot == '#pricerange':
                    ad = [i for i, ad in enumerate(advice) if ad[2]!='']
                    from_ad = ad[0] 
            
            for i, c in enumerate(advice[from_ad]):
                if c!='':
                    to_convey = i
                    suggestion = add_to_suggestion(suggestion, i)
                    break

            if to_convey is not None:
                t += ':'+names[to_convey]
                advice_nlg = adv_templates[t][0]
                advice_nlg = advice_nlg.split()
                for i, word in enumerate(advice_nlg):
                    if word[0]=='<' and word[-1]=='>':
                        if word == '<FOOD>':
                            target_slot = self.domain.get_usr_slot('#food')
                            advice_nlg[i] = target_slot.vocabulary[int(advice[from_ad][0])]
                        elif word == "<AREA>":
                            target_slot = self.domain.get_usr_slot('#area')
                            advice_nlg[i] = target_slot.vocabulary[int(advice[from_ad][1])]
                        elif word == '<PRICERANGE>':
                            target_slot = self.domain.get_usr_slot('#pricerange')
                            advice_nlg[i] = target_slot.vocabulary[int(advice[from_ad][2])]
#             print(advice_nlg)
        
        elif len(advice) > 1:
            for i, q in enumerate(stat_query):
                if q is not None:
                    given.append(i)                 
                    t += names[i]+'_'
            t = t[:-1] + ':'
            
            if advice[0][-1] == advice[1][-1]:# and not any(a==b for a, b in zip(advice[0][-1], advice[1][-1])):
                to_convey = set()
                
                for i, c in enumerate(advice[0][:-1]):
                    if i not in given and c!= '':
                        to_convey.add(i)
                        suggestion = add_to_suggestion(suggestion, i)
                        
                for i, c in enumerate(advice[1][:-1]):
                    if i not in given and c!= '':
                        to_convey.add(i)
                        suggestion = add_to_suggestion(suggestion, i)
                
#                 print("to_convey: ", list(to_convey))
                
                to_convey = list(to_convey)
                sorted(to_convey)
                for c in to_convey:
                    t += names[c] + '_'
                
                t = t[:-1]                
                advice_nlg = adv_templates[t][0]
                advice_nlg = advice_nlg.split()                
                for i, word in enumerate(advice_nlg):
                    if word[0]=='<' and word[-1]=='>':
                        if word == '<FOOD>':
                            target_slot = self.domain.get_usr_slot('#food')
                            if advice[0][0] != '':
                                advice_nlg[i] = target_slot.vocabulary[int(advice[0][0])]
                            else:
                                advice_nlg[i] = target_slot.vocabulary[int(advice[1][0])]
                        
                        elif word == "<AREA>":
                            target_slot = self.domain.get_usr_slot('#area')
                            if advice[0][1] != '':
                                advice_nlg[i] = target_slot.vocabulary[int(advice[0][1])]
                            else:
                                advice_nlg[i] = target_slot.vocabulary[int(advice[1][1])]
                        
                        elif word == '<PRICERANGE>':
                            target_slot = self.domain.get_usr_slot('#pricerange')
                            if advice[0][2] != '':
                                advice_nlg[i] = target_slot.vocabulary[int(advice[0][2])]
                            else:
                                advice_nlg[i] = target_slot.vocabulary[int(advice[1][2])]
#                 print(advice_nlg)
            
            else:
                to_convey = set()
                for i, c in enumerate(advice[0][:-1]):
                    if i not in given and c!= '':
                        to_convey.add(i)
                        suggestion = add_to_suggestion(suggestion, i)
                for i, c in enumerate(advice[1][:-1]):
                    if i not in given and c!= '':
                        to_convey.add(i)
                        suggestion = add_to_suggestion(suggestion, i)
                
#                 print("to_convey: ", list(to_convey))                
                to_convey = list(to_convey)
                sorted(to_convey)
                for c in to_convey:
                    t += names[c] + '_'
                    
                t = t[:-1]
                if len(to_convey) > 1:
                    advice_nlg = adv_templates[t][1]
                    advice_nlg = advice_nlg.split()                
                    for i, word in enumerate(advice_nlg):
                        if word[0]=='<' and word[-1]=='>':
                            if word == '<FOOD>':
                                target_slot = self.domain.get_usr_slot('#food')
                                if advice[0][0] != '':
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[0][0])]
                                else:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[1][0])]
                            elif word == "<AREA>":
                                target_slot = self.domain.get_usr_slot('#area')
                                if advice[0][1] != '':
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[0][1])]
                                else:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[1][1])]
                            elif word == '<PRICERANGE>':
                                target_slot = self.domain.get_usr_slot('#pricerange')
                                if advice[0][2] != '':
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[0][2])]
                                else:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[1][2])]
                else:
                    advice_nlg = adv_templates[t][1]
                    advice_nlg = advice_nlg.split()
                    flag = [False, False, False]
                    for i, word in enumerate(advice_nlg):
                        if word[0]=='<' and word[-1]=='>':
                            if word == '<FOOD>':
                                target_slot = self.domain.get_usr_slot('#food')
                                if not flag[0]:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[0][0])]
                                    flag[0] = True
                                else:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[1][0])]
                            elif word == "<AREA>":
                                target_slot = self.domain.get_usr_slot('#area')
                                if not flag[1]:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[0][1])]
                                    flag[1] = True
                                else:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[1][1])]
                            elif word == '<PRICERANGE>':
                                target_slot = self.domain.get_usr_slot('#pricerange')
                                if not flag[2]:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[0][2])]
                                    flag[2] = True
                                else:
                                    advice_nlg[i] = target_slot.vocabulary[int(advice[1][2])]                    
#                 print(advice_nlg)
        elif len(advice)==1:
            for i, q in enumerate(stat_query):
                if q is not None:
                    given.append(i)                 
                    t += names[i]+'_'
            t = t[:-1] + ':'
            to_convey = None
            for i, c in enumerate(advice[0][:-1]):
                if i not in given and c!= '':
                    to_convey = i
                    suggestion = add_to_suggestion(suggestion, i)
                    break
                
#                 print("to_convey: ", to_convey)                
            if to_convey is not None:
                t += names[to_convey]                
                advice_nlg = adv_templates[t][0]
                advice_nlg = advice_nlg.split()                
                for i, word in enumerate(advice_nlg):
                    if word[0]=='<' and word[-1]=='>':
                        if word == '<FOOD>':
                            target_slot = self.domain.get_usr_slot('#food')
                            advice_nlg[i] = target_slot.vocabulary[int(advice[0][0])]

                        elif word == "<AREA>":
                            target_slot = self.domain.get_usr_slot('#area')
                            advice_nlg[i] = target_slot.vocabulary[int(advice[0][1])]

                        elif word == '<PRICERANGE>':
                            target_slot = self.domain.get_usr_slot('#pricerange')
                            advice_nlg[i] = target_slot.vocabulary[int(advice[0][2])]

#                 print(advice_nlg)            
                                    
        return " ".join(str_actions) + '. ' + ' '.join(advice_nlg), lexicalized_actions, suggestion

class UserNlg(AbstractNlg):
    """
    NLG class to generate utterances for the user side.
    """

    def generate_sent(self, actions):
        """
         Map a list of user actions to a string.

        :param actions: a list of actions
        :return: uttearnces in string
        """
        str_actions = []
        for a in actions:
            if a.act == UserAct.KB_RETURN:
                sys_goals = a.parameters[1]
                sys_goal_dict = {}
                for k, v in sys_goals.items():
                    slot = self.domain.get_sys_slot(k)
                    sys_goal_dict[k] = slot.vocabulary[v]

                str_actions.append(json.dumps({"RET": sys_goal_dict}))
            elif a.act == UserAct.GREET:
                str_actions.append(self.sample(["Hi.", "Hello robot.", "What's up?"]))

            elif a.act == UserAct.GOODBYE:
                str_actions.append(self.sample(["That's all.", "Thank you.", "See you."]))

            elif a.act == UserAct.REQUEST:
                slot_type, _ = a.parameters[0]
                target_slot = self.domain.get_sys_slot(slot_type)
                str_actions.append(target_slot.sample_request())

            elif a.act == UserAct.INFORM:
                has_self_correct = a.parameters[-1][0] == BaseUsrSlot.SELF_CORRECT
                slot_type, slot_value = a.parameters[0]
                target_slot = self.domain.get_usr_slot(slot_type)

                def get_inform_utt(val):
                    if val is None:
                        return self.sample(["Anything is fine.", "I don't care.", "Whatever is good."])
                    else:
                        return target_slot.sample_inform() % target_slot.vocabulary[val]

                if has_self_correct:
                    wrong_value = target_slot.sample_different(slot_value)
                    wrong_utt = get_inform_utt(wrong_value)
                    correct_utt = get_inform_utt(slot_value)
                    connector = self.sample(["Oh no,", "Uhm sorry,", "Oh sorry,"])
                    str_actions.append("%s %s %s" % (wrong_utt, connector, correct_utt))
                else:
                    str_actions.append(get_inform_utt(slot_value))

            elif a.act == UserAct.CHAT:
                str_actions.append(self.sample(["What's your name?", "Where are you from?"]))

            elif a.act == UserAct.YN_QUESTION:
                slot_type, expect_id = a.parameters[0]
                target_slot = self.domain.get_sys_slot(slot_type)
                expect_val = target_slot.vocabulary[expect_id]
                str_actions.append(target_slot.sample_yn_question(expect_val))

            elif a.act == UserAct.CONFIRM:
                str_actions.append(self.sample(["Yes.", "Yep.", "Yeah.", "That's correct.", "Uh-huh."]))

            elif a.act == UserAct.DISCONFIRM:
                str_actions.append(self.sample(["No.", "Nope.", "Wrong.", "That's wrong.", "Nay."]))

            elif a.act == UserAct.SATISFY:
                str_actions.append(self.sample(["No more questions.", "I have all I need.", "All good."]))

            elif a.act == UserAct.MORE_REQUEST:
                str_actions.append(self.sample(["I have more requests.", "One more thing.", "Not done yet."]))

            elif a.act == UserAct.NEW_SEARCH:
                str_actions.append(self.sample(["I want to search a new one.", "New request.", "A new search."]))
            
            elif a.act == UserAct.RESTART:
                str_actions.append(json.dumps({"RET": self.sample(["No match found. Would you like to another?", "There is no restaurant matching your constraint. Please change your constraint and try again."])}))

            else:
                raise ValueError("Unknown user act %s for NLG" % a.act)

        return " ".join(str_actions)

    def add_hesitation(self, sents, actions):
        pass

    def add_self_restart(self, sents, actions):
        pass
