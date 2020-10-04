import numpy as np
import logging
import pandas as pd
import json

class slot_vocab():
    def __init__(self, usr_slots):
        self.slot_vocab = {}
        self.slot_vocab_inv = {}
        for slot, desc, values in usr_slots:
            self.slot_vocab[slot] = {}
            self.slot_vocab_inv[slot] = {}
            for idx, value in enumerate(values):
                if value not in self.slot_vocab[slot]:
                    self.slot_vocab[slot][value] = idx
                    self.slot_vocab_inv[slot][idx]  = value
                    
    def get_slot_id(self, slot, value):
        if value in self.slot_vocab[slot]:
            return self.slot_vocab[slot][value]
        else:
            self.slot_vocab[slot][value] = len(self.slot_vocab[slot])
            self.slot_vocab_inv[slot][len(self.slot_vocab_inv[slot])] = value
            return self.slot_vocab[slot][value]
        
    def get_slot_value(self, slot, idx):
        if idx in self.slot_vocab_inv[slot]:
            return self.slot_vocab_inv[slot][idx]
        
        else:
            raise IndexError

class Database(object):
    """
    A table-based database class. Each row is an entry and each column is an attribute. Each attribute
    has vocabulary size called modality.
    
    :ivar usr_dirichlet_priors: the prior for each attribute : 2D list [[]*modality]
    :ivar num_usr_slots: the number of columns: Int
    :ivar usr_modalities: the vocab size of each column : List
    :ivar usr_pdf: the PDF for each columns : 2D list
    :ivar num_rows: the number of entries
    :ivar table: the content : 2D list [[] *num_rows]
    :ivar indexes: for efficient SELECT : [{attribute_word -> corresponding rows}]
    """

    logger = logging.getLogger(__name__)

    def __init__(self, usr_dirichlet_priors, sys_dirichlet_priors, num_rows, usr_slots):
        """
        :param usr_dirichlet_priors: 2D list [[]_0, []_1, ... []_k] for each searchable attributes
        :param sys_dirichlet_priors: 2D llst for each entry (non-searchable attributes)
        :param num_rows: the number of row in the database
        """
        self.usr_dirichlet_priors = usr_dirichlet_priors
        self.sys_dirichlet_priors = sys_dirichlet_priors

        self.num_usr_slots = len(usr_dirichlet_priors)
        self.usr_modalities = [len(p) for p in usr_dirichlet_priors]

        self.num_sys_slots = len(sys_dirichlet_priors)
        self.sys_modalities = [len(p) for p in sys_dirichlet_priors]

        # sample attr_pdf for each attribute from the dirichlet prior
        self.usr_pdf = [np.random.dirichlet(d_p) for d_p in self.usr_dirichlet_priors]
        self.sys_pdf = [np.random.dirichlet(d_p) for d_p in self.sys_dirichlet_priors]
#         self.num_rows = num_rows

        # begin to generate the table
        slot_voc = slot_vocab(usr_slots)
        usr_table, usr_index = self.load_kb(slot_voc)   
        self.num_rows = len(usr_table[0])
        print(self.num_rows)
#         usr_table, usr_index = self._gen_table(self.usr_pdf, self.usr_modalities, self.num_usr_slots, num_rows)
        sys_table, sys_index = self._gen_table(self.sys_pdf, self.sys_modalities, self.num_sys_slots, self.num_rows)
        
        # append the UID in the first column
        sys_table.insert(0, range(self.num_rows))        
        
        self.table = np.array(usr_table).transpose()
        self.indexes = usr_index
        self.sys_table = np.array(sys_table).transpose()
        
#         print(len(usr_table[0]), usr_table)
        self.db_stat = self._get_stat(self.table)
        cons_table, cons_index = self._gen_table(self.usr_pdf, self.usr_modalities, self.num_usr_slots, 5000)#num_rows)
#         self.cons_table = self.table
#         self.cons_index = self.indexes
        self.cons_table = np.array(cons_table).transpose()
        self.cons_index = cons_index
#         print(usr_table, usr_index)
#         print(len(usr_table[0]), self.cons_table, self.cons_table[0], cons_index)
        kb_table = self.table.tolist()
        c = 0
        for row in self.cons_table:
            if row.tolist() in kb_table:
                c += 1
            
        print(c, float(c)/5000)
        exit()        
#         with open('db.txt', 'w') as out_file:
#             json.dump(usr_table, out_file)
        
#         exit()
        
    @staticmethod
    def _get_stat(db):
        data = pd.DataFrame(db, columns=['food', 'area', 'pricerange'])
        food = [x for x in list(data.food.unique()) if str(x)!='nan']
        food = sorted(food)

        area = [x for x in list(data.area.unique())]
        area = sorted(area)

        pricerange = [x for x in list(data.pricerange.unique())]
        pricerange = sorted(pricerange)

        stat = pd.DataFrame(columns=['food', 'area', 'pricerange', 'count'])

        for f in food:
            curr_db = data[data.food==f]
            stat.loc[len(stat)] = [f, '', '', len(curr_db)]

        for a in area:
            curr_db = data[data.area==a]
            stat.loc[len(stat)] = ['', a, '', len(curr_db)]

        for p in pricerange:
            curr_db = data[data.pricerange==p]
            stat.loc[len(stat)] = ['', '', p, len(curr_db)] 

        for f in food:
            curr_db = data[data.food==f]
            totalcount = len(curr_db)

            for a in area:
                count = len(curr_db[curr_db.area==a])        
                if count > 0:
                    stat.loc[len(stat)] = [f, a, '', count]
                    curr_db_ = curr_db[curr_db.area==a]

                    for p in pricerange:
                        count = len(curr_db_[(curr_db_.pricerange==p)])
                        if count > 0:
                            stat.loc[len(stat)] = [f, a, p, count]


        for a in area:
            curr_db = data[data.area==a]
            totalcount = len(curr_db)

            for p in pricerange:
                count = len(curr_db[curr_db.pricerange==p])        
                if count > 0:
                    stat.loc[len(stat)] = ['', a, p, count]

        for p in pricerange:
            curr_db = data[data.pricerange==p]
            totalcount = len(curr_db)

            for f in food:
                count = len(curr_db[curr_db.food==f])        
                if count > 0:
                    stat.loc[len(stat)] = [f, '', p, count]
        
        return stat
            
    @staticmethod
    def _gen_table(pdf, modalities, num_cols, num_rows):
        list_table = []
        indexes = []
        for idx in range(num_cols):
            col = np.random.choice(range(modalities[idx]), p=pdf[idx], size=num_rows)
            list_table.append(col)
            # indexing
            index = {}
            for m_id in range(modalities[idx]):
                matched_list = np.squeeze(np.argwhere(col == m_id)).tolist()
                matched_list = set(matched_list) if type(matched_list) is list else {matched_list}
                index[m_id] = matched_list
            indexes.append(index)
        return list_table, indexes

    @staticmethod
    def load_kb(slot_voc):
        db = pd.read_json('db.json')
        db = db.reset_index()
        slots = ['food', 'area', 'pricerange']
        list_table = [np.array([slot_voc.get_slot_id(slot, v) for v in db[slot].values]) for slot in slots]
        indexes = [{i:set(db.index[db[slot]==slot_voc.get_slot_value(slot, i)].tolist()) for i in range(len(slot_voc.slot_vocab[slot]))} for slot in slots]
        return list_table, indexes
    
    def sample_unique_row(self):
        """
        :return: a unique row in the searchable table
        """
        unique_rows = np.unique(self.cons_table, axis=0)
        idxes = range(len(unique_rows))
        np.random.shuffle(idxes)
        return unique_rows[idxes[0]]
        return row

    def select(self, query, return_index=False):
        """
        Filter the database entries according the query.
        
        :param query: 1D [] equal to the number of attributes, None means don't care
        :param return_index: if return the db index
        :return return a list system_entries and (optional)index that satisfy all constrains
        
        """
        valid_idx = set(range(self.num_rows))
        for q, a_id in zip(query, range(self.num_usr_slots)):
#             print(valid_idx, q, a_id, self.indexes)
            if q:
                valid_idx = valid_idx.intersection(self.indexes[a_id][q])
                if len(valid_idx) == 0:
                    break
        valid_idx = list(valid_idx)        
#         print('v_idx: ', valid_idx, query)
        if return_index:
            return self.sys_table[valid_idx, :], valid_idx
        else:
            return self.sys_table[valid_idx, :]

    def pprint(self):
        """
        print statistics of the database in a beautiful format. 
        """

        self.logger.info("DB contains %d rows (%d unique ones), with %d attributes"
                         % (self.num_rows, len(np.unique(self.table, axis=0)), self.num_usr_slots))
    
    
    def get_advice(self, constraints, constraint_order=[]):           
        belief_state = {'area': None, 'pricerange': None, 'food': None}
        cols = {0:'food', 1:'area', 2:'pricerange'}
#         for k, v in constraints.iteritems():            
#             belief_state[k] = v
        for i, c in enumerate(constraints):
            belief_state[cols[i]] = c
        
        search_query = {s: None for s in belief_state}
        flag = False
        for s, v in belief_state.items():
            if v is not None:
                search_query[s] = v
                flag = True
#             else:
#                 search_query.pop(s)
        
        t = self.db_stat
        for s,v in search_query.items():
            if v is not None:
                t = t[t[s]==v]
            else:
                if flag:
                    t = t[t[s]!='']

        t = t.sort_values(by=['count'], ascending=False)
        t = t.reset_index(drop=True)
#         print(t)
        
        advice = []        
        for idx, row in t.iterrows():
            advice.append([row.food, row.area, row.pricerange, row['count']])
            if idx >=5:
                break
  
#         if len(advice) == 0:
#             for i in range(len(belief_state)):
# #                 search_query = [v for j, (s,v) in enumerate(belief_state.items()) if v in uniq_constraint[:-(i+1)]]
# #                 search_query = {s: v for j, (s,v) in enumerate(belief_state.items()) if j!=i}
#                 c = [None, None, None]
#                 for i, v in enumerate(constraints[:-1]):
#                     c[i] = v
                
#                 advice = self.get_advice(c)
#                 if len(advice) > 0:
#                     break
                    
#         if len(advice) == 0:
#             advice = self.get_advice({})
        return advice
        