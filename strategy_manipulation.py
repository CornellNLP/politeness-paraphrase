from convokit import Utterance, PolitenessStrategies
from collections import defaultdict

import json
import numpy as np
from generation_utils import load_model, preditction_with_beam_search
from perception_utils import estimate_perception

from settings import MARKER_DELETION_MODE, PUNCS, GEN_MODEL_PATH, PERCEPTION_MODEL_PATH

# pre-trained generation model
TOKENIZER, DEVICE, MODEL = load_model(GEN_MODEL_PATH)

# average perception model 
with open(PERCEPTION_MODEL_PATH, 'r') as f:
    AVERAGE_MODEL = json.load(f)

    
###### REMOVING STRATEGIES ### 

# locate the span of text seperated by within sentence punctuations
# anchor_tuple: (marker, tok_idx) tuples, obtained by extracting markers
# return a set of tok_idx to be removed

def locate_segment(tokens, anchor_tuple, stopping_puncs = PUNCS):
    
    anchor_tok, anchor_idx = anchor_tuple
    
    if anchor_tok not in [tok['tok'] for tok in tokens]:
        print('The specified marker is not present in the sentence.')
        return []
    
    # search from both ends until segment boundary 
    span, start_idx, end_idx = [], anchor_idx-1, anchor_idx
    
    # moving start_idx leftward   
    while start_idx >= 0 and tokens[start_idx]['tok'] not in stopping_puncs:
        
        span.append(start_idx)
        start_idx -= 1
    
    # moving end_idx rightward   
    while (end_idx < len(tokens)):
        
        cur_tok = tokens[end_idx]['tok']
        
        if cur_tok in stopping_puncs:
            span.append(end_idx)
            break
            
        span.append(end_idx)
        end_idx += 1

    return span


# utterance should have strategy annotations 
def remove_strategies_from_utt(utt, strategies_to_remove, \
                               marker_attribute_name='markers', \
                               removed_attribute_name='post_del_context'):
    
    removal_info_by_sent = defaultdict(set)
    
    # markers being used in the current utterance
    strategy_markers = {k:v for k,v in utt.retrieve_meta(marker_attribute_name).items() if len(v) > 0}
            
    # list of tokens (root info not needed)
    sent_toks = [x['toks'] for x in utt.meta['parsed']]
    
    # remove markers 
    strategies_to_proc = set([s for s in strategies_to_remove if s in strategy_markers])
    
    for strategy in strategies_to_proc:
        
        for (tok, sent_idx, tok_idx) in strategy_markers[strategy]:
            
            if MARKER_DELETION_MODE[strategy] == 'token': 
                removal_info_by_sent[sent_idx].add(tok_idx)
            
            else:
                anchor_tuple = (tok, tok_idx)
                span = locate_segment(sent_toks[sent_idx], anchor_tuple)
                removal_info_by_sent[sent_idx] = removal_info_by_sent[sent_idx].union(span)                

    res_sents = []
    for idx, sent in enumerate(sent_toks):
        
        # ensure sentence do not start with puncutations 
        res_toks = [tok for j, tok in enumerate(sent) if j not in removal_info_by_sent[idx]]
        
        if len(res_toks) > 0 and res_toks[0]['dep'] == 'punct':
            res_toks = res_toks[1:]
    
        # remove punctuations if it is at the start
        res_sents.append(" ".join([x['tok'] for x in res_toks]).strip())
    
    utt.add_meta(removed_attribute_name, " ".join(res_sents).strip())
    
    return utt.meta[removed_attribute_name]


###### ADDING STRATEGIES ### 

def convert_to_training_format(strategy, post_del_content, text):
    
    return "<STR> <{}> <CONTEXT> {} <START> {} <END>".format(strategy, post_del_content, text)


def format_input(strategy, content):

    return "<STR> <{}> <CONTEXT> {} <START>".format(strategy, content)


    
def select_candidate(candidates, intended_politeness, \
                     politeness_transformer, spacy_nlp, perception_model):
    
    # process all candidates 
    annotated_utts = [politeness_transformer.transform_utterance(text, spacy_nlp) for text in candidates]
    
    strategy_sets = [{k for k,v in utt.meta[politeness_transformer.strategy_attribute_name].items() if v == 1} for utt in annotated_utts]
    
    # estimated perceptions 
    misalignment = [abs(intended_politeness - estimate_perception(perception_model, strategies)) for strategies in strategy_sets]
    
    # find the candidate that results in minimal gap
    min_idx = np.argsort(misalignment)[0]
    
    return candidates[min_idx]

# # utterance should have been annotated with strategy-removed context and intended politeness
def add_strategies_to_utt(utt, strategies_to_add, \
                          politeness_transformer, spacy_nlp, \
                          perception_model = AVERAGE_MODEL, \
                          content_attribute_name='context', \
                          output_attribute_name="paraphrase", \
                          tokenizer = TOKENIZER, model = MODEL, device = DEVICE):
    
    # generate candidate solutions 
    candidates = [utt.meta[content_attribute_name]]
    
    for strategy in strategies_to_add:

        res = []
        instances = [format_input(strategy, content) for content in candidates]

        for instance in instances:
            res.extend(preditction_with_beam_search(instance, tokenizer, model, device))
    
        candidates = res
    
    if 'intended_politeness' not in utt.meta:
        intended_politeness = estimate_perception(perception_model, utt.meta['strategy_set'])
    else:
        intended_politeness = utt.meta['intended_politeness']
    
    
    # select paraphrase that minimizes the expected perception gap
    # note that naturalness of the output is not taken into account in selection 
    # (it's possible that one of the candidate actually sounds more natural but is not chosen)
    paraphrase = select_candidate(candidates, intended_politeness, politeness_transformer, spacy_nlp, perception_model)
    
    utt.meta[output_attribute_name] = paraphrase
    
    return paraphrase