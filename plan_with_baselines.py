import numpy as np
from collections import Counter, defaultdict

from convokit import Corpus, Utterance, Speaker
from convokit import BoWTransformer, ConvoKitMatrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from strategy_manipulation import remove_strategies_from_utt
from perception_utils import estimate_perception

from settings import UPPER_BOUND

###### Retrieval solution ######

def init_transformer(train_corpus):

    # preparing tfidf vectorizer 
    tfidf_vect = TfidfVectorizer()

    tfidf = BoWTransformer(obj_type='utterance', vector_name='tfidf', \
                           text_func=lambda utt:utt.meta['content'], vectorizer=tfidf_vect)

    tfidf.fit(train_corpus)
    
    return tfidf 


# assumes both corpuses have marker-deleted "content" utterance metadata
def add_similarity_scores(test_corpus, train_corpus, tfidf):

    train_corpus = tfidf.transform(train_corpus) 
    test_corpus = tfidf.transform(test_corpus)
    
    train_tfidf = train_corpus.get_vector_matrix('tfidf')
    test_tfidf = test_corpus.get_vector_matrix('tfidf') 
    
    similarity_scores = cosine_similarity(test_tfidf.matrix, train_tfidf.matrix)
    
    # add to corpus
    test_corpus.set_vector_matrix(name='similarity', \
                                  matrix= similarity_scores, \
                                  columns = train_tfidf.ids, ids = test_tfidf.ids)
    
    # associate each utterance 
    for utt in test_corpus.iter_utterances():
        
        utt.add_vector('similarity')
    
    return test_corpus
    
# find the most semantically similar reference utterance showing the same polarity
# that also satisfies other constraints 
def get_retrieval_plan(utt, train_corpus, train_ids, at_risk_strategies, upper_bound=UPPER_BOUND): 
    
    scores = utt.get_vector('similarity')[0]
    
    sorted_idx = np.argsort(scores)
    inspection_order = sorted_idx[::-1]

    # currently used strategies 
    cur_plan = utt.meta['strategy_set']
    
    # current use from the subjunctive-indicative pair
    sub_ind_pair = set(["Subjunctive", "Indicative"])
    cur_si_status = len(cur_plan.intersection(sub_ind_pair))

    # inspecting from the most semantically similar utterances 
    for idx in inspection_order:
        
        ref_utt = train_corpus.get_utterance(train_ids[idx])
                  
        # consider only reference utterance that is of the same polarity 
        if ref_utt.meta['polarity'] == utt.meta['polarity']:

            ref_plan = ref_utt.meta['strategy_set'] - at_risk_strategies
            ref_si_status = len(ref_plan.intersection(sub_ind_pair))

            # check the number of strategies to add meets the requirement
            added_strategies = set(ref_plan) - set(cur_plan) 

            if len(added_strategies) <= upper_bound and ref_si_status == cur_si_status:

                return ref_plan

    # if no such plan can be found, return set with at risk strategies removed 
    return cur_plan - at_risk_strategies


###### Greedy solutions ######

def get_greedy_plan(strategy_set, sender_model, receiver_model, at_risk_strategies, upper_bound = UPPER_BOUND):
    
    # estimated intended politeness level
    intended = estimate_perception(sender_model, strategy_set)
    
    # strategy coefs, from either side
    sender_coefs, receiver_coefs = sender_model['coefs'], receiver_model['coefs']
    
    if intended >= 0: 
    
        # negativity constraint 
        neg_strategies = {k for k,v in receiver_coefs.items() if v < 0}
        combined_risk_set = neg_strategies.union(at_risk_strategies)

        lookup = construct_lookup_table(sender_coefs, receiver_coefs, combined_risk_set)
    
    else: 
        
        lookup = construct_lookup_table(sender_coefs, receiver_coefs, at_risk_strategies)
        
    # strategies to be substituted 
    strategy_perception_gaps = {k: abs(sender_coefs[k] - receiver_coefs[k]) for k in strategy_set}
    
    # substitute from strategies with the largest perception gap
    strategy_sorted = sorted(strategy_perception_gaps.items(), key=lambda x:x[1], reverse=True)
    
    cnt, sols = 0, set()
    
    for name, _ in strategy_sorted:
        
        # when new strategies can still be added, find best replacement 
        if cnt <= upper_bound:
            
            replacement = lookup[name]
            
            # check if this is a new strategy 
            if replacement != name and replacement not in sols:
                cnt += 1
        
            sols.add(replacement)
        
        # when upper bound is reached, keep using original strategy
        elif name not in at_risk_strategies:
            
            sols.add(name)
        
        # if original strategy isn't safe, just drop it 
        else:
            continue 
    
    return sols


def find_closest_sub(coef, receiver_coefs, safe_strategy_set):
    
    # find strategy whose receiver coefs is the closest
    coef_diffs = {k: abs(coef - receiver_coefs[k]) for k in safe_strategy_set}
    
    sorted_subs = sorted(coef_diffs.items(), key=lambda item: item[1])
    sub, _ = sorted_subs[0]

    return sub


def construct_lookup_table(sender_coefs, receiver_coefs, at_risk_strategies): 
    
    lookup = defaultdict()
    
    # special strategies 
    # (following the constraing used in the ILP setting) 
    sub_ind_pair = set(['Subjunctive', "Indicative"])
    
    # consider subjunctive and indicative seperately 
    excluded = at_risk_strategies.union(sub_ind_pair)
    
    # candidate strategies to search for substitues  
    safe_strategies = {strategy: coef for strategy, coef in receiver_coefs.items() \
                       if strategy not in excluded}
    
    safe_from_pair = sub_ind_pair - at_risk_strategies
    
    # there needs to be at least 1 safe strategy to construct a plan from 
    assert len(safe_strategies) > 0
    
    # finding closest subsitute
    for strategy, coef in sender_coefs.items():
        
        # by pass special cases
        if strategy in sub_ind_pair:
            continue 
        
        # keep track of the greedy substitute 
        lookup[strategy] = find_closest_sub(coef, receiver_coefs, safe_strategies)

        
    # handing subjunctive and indicative 
    # if both are not safe, no sub
    
    assert len(safe_from_pair) > 0
    
    for strategy in sub_ind_pair:
        
        lookup[strategy] = find_closest_sub(sender_coefs[strategy], \
                                            receiver_coefs, safe_from_pair)
    
    return lookup 


