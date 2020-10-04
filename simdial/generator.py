# -*- coding: utf-8 -*-
# author: Tiancheng Zhao

from simdial.agent.user import User
from simdial.agent.system import System
from simdial.channel import ActionChannel, WordChannel
from simdial.agent.nlg import SysNlg, UserNlg
from simdial.complexity import Complexity
from simdial.domain import Domain
#import progressbar
import json
import numpy as np
import sys
import os
import re

class Generator(object):
    """
    The generator class used to generate synthetic slot-filling human-computer conversation in any domain. 
    The generator can be configured to generate data with varying complexity at: propositional, interaction and social 
    level. 
    
    The required input is a domain specification dictionary + a configuration dict.
    """

    @staticmethod
    def pack_msg(speaker, utt, **kwargs):
        resp = {k: v for k, v in kwargs.items()}
        resp["speaker"] = speaker
        resp["utt"] = utt
        return resp

    @staticmethod
    def pprint(dialogs, usr_goals, in_json, domain_spec, output_file=None):
        """
        Print the dailog to a file or STDOUT
        
        :param dialogs: a list of dialogs generated
        :param output_file: None if print to STDOUT. Otherwise write the file in the path
        """
        f = sys.stdout if output_file is None else open(output_file, "wb")
        
        cnt_others, cnt_inf, cnt_req, cnt_query = 0., 0., 0., 0.
        
        if in_json:
            combo = {'dialogs': dialogs}#, 'meta': domain_spec.to_dict()}
            json.dump(combo, f, indent=2)
        
        else:
            for idx, (d, usr_g) in enumerate(zip(dialogs, usr_goals)):
                f.write("## DIALOG %d ##\n" % idx)
                f.write("User goal: " + str(usr_g)+'\n')
                for turn in d:
                    speaker, utt, actions = turn["speaker"], turn["utt"], turn["actions"]
#                     print(actions)
                    act = [a.dump_string() for a in actions]
                    act = " ".join(act)
#                     print(act, utt)
#                     print(act, any(s in act for s in ['#open', '#parking']))        
                    if any(s in act for s in ['query']):
                        cnt_query += 1

                    elif any(s in act for s in ['#open', '#parking', 'kb_return']):
                        cnt_req += 1               
                    
                    elif any(s in act for s in ['#food', '#area', '#pricerange']):
                        cnt_inf += 1                                    
                    
                    else:
                        cnt_others += 1
                    
                    if utt:
                        str_actions = utt
                    else:
                        str_actions = " ".join([a.dump_string() for a in actions])
                    
#                     if speaker == "SYS":
#                         f.write('Act: %s, Sugg: %s\n' %(act, turn["suggestions"]))
#                     else:
#                         f.write('Act: %s\n' %(act))
                    
                    if '{"QUERY"' not in str_actions and '{"RET"' not in str_actions:
                        if speaker == "USR":
    #                         f.write("%s(%f)-> %s\n" % (speaker, turn['conf'], str_actions))
                            f.write("%s -> %s\n" % (speaker, str_actions))
                        else:
                            f.write("%s -> %s\n" % (speaker, str_actions))

                f.write("\n")
                
        if output_file is not None:
            f.close()
        
        print(cnt_inf/len(dialogs), cnt_req/len(dialogs), cnt_others/len(dialogs), cnt_query/len(dialogs))
        
    @staticmethod
    def print_stats(dialogs):
        """
        Print some basic stats of the dialog.
        
        :param dialogs: A list of dialogs generated.
        """
        print("%d dialogs" % len(dialogs))
        all_lens = [len(d) for d in dialogs]
        print("Avg len {} Max Len {}".format(np.mean(all_lens), np.max(all_lens)))

        total_cnt = 0.
        kb_cnt = 0.
        ratio = []
        for d in dialogs:
            local_cnt = 0.
            for t in d:
                total_cnt +=1
                if 'QUERY' in t['utt']:
                    kb_cnt += 1
                    local_cnt += 1
            ratio.append(local_cnt/len(d))
#         print(kb_cnt/total_cnt)
#         print(np.mean(ratio))

    def gen(self, domain, complexity, num_sess=1, advice_prob=1.0):
        """
        Generate synthetic dialogs in the given domain. 

        :param domain: a domain specification dictionary
        :param complexity: an implmenetaiton of Complexity
        :param num_sess: how dialogs to generate
        :return: a list of dialogs. Each dialog is a list of turns.
        """
        dialogs = []
        usr_goals = []
        action_channel = ActionChannel(domain, complexity)
        word_channel = WordChannel(domain, complexity)

        # natural language generators
        sys_nlg = SysNlg(domain, complexity)
        usr_nlg = UserNlg(domain, complexity)

        #bar = progressbar.ProgressBar(max_value=num_sess)
        for i in range(num_sess):
#             print("Dial:{}".format(i))
            #bar.update(i)
            usr = User(domain, complexity, advice_prob)
            sys = System(domain, complexity)
            usr_goals.append(usr.usr_cons_readable)
            
            # begin conversation
            noisy_usr_as = []
            dialog = []
            conf = 1.0
            while True:
                # make a decision
                sys_r, sys_t, sys_as, sys_s, stat_query = sys.step(noisy_usr_as, conf)
#                 print('sys: ', sys_as)
                sys_utt, sys_str_as, suggestions = sys_nlg.generate_sent(sys_as, domain=domain, stat_query=stat_query)
                dialog.append(self.pack_msg("SYS", sys_utt, actions=sys_str_as, domain=domain.name, state=sys_s, suggestions=suggestions))
#                 print(suggestions)
            
                if sys_t:
                    break

                usr_r, usr_t, usr_as = usr.step(sys_as, suggestions)
#                 print('usr: ', usr_as)

                # passing through noise, nlg and noise!
                noisy_usr_as, conf = action_channel.transmit2sys(usr_as)
                usr_utt = usr_nlg.generate_sent(noisy_usr_as)
                noisy_usr_utt = word_channel.transmit2sys(usr_utt)

                dialog.append(self.pack_msg("USR", noisy_usr_utt, actions=noisy_usr_as, conf=conf, domain=domain.name))

            dialogs.append(dialog)

        return dialogs, usr_goals

    def gen_corpus(self, name, domain_spec, complexity_spec, size, advice_prob=1.0):
        if not os.path.exists(name):
            os.mkdir(name)

        # create meta specifications
        domain = Domain(domain_spec)
        complex = Complexity(complexity_spec)

        # generate the corpus conditioned on domain & complexity
        corpus, usr_goals = self.gen(domain, complex, num_sess=size, advice_prob=advice_prob)

        # txt_file = "{}-{}-{}.{}".format(domain_spec.name,
        #                                complexity_spec.__name__,
        #                                size, 'txt')

        json_file = "{}-{}-{}.{}".format(domain_spec.name,
                                         complexity_spec.__name__,
                                         size, 'txt')

        json_file = os.path.join(name, json_file)
        self.pprint(corpus, usr_goals, False, domain_spec, json_file)
        self.print_stats(corpus)
