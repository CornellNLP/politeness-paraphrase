import pandas as pd
from sklearn.linear_model import LinearRegression
from convokit import Corpus, Utterance, Speaker

# scale scores from 1-25 range to -3 - 3 range
def scale_func(val, a = 0.25, b = -3.25):
    return a*val + b


def get_strategies_df(corpus, strategy_attribute_name):
    
    strategies_dict = {utt.id: utt.meta[strategy_attribute_name] for utt in corpus.iter_utterances()}
    
    df = pd.DataFrame.from_dict(strategies_dict, orient="index")
    
    # return df with alphabetically ordered feature name
    return df.sort_index(axis=1)


def train_model(X, y):
    
    return LinearRegression().fit(X, y)


def get_model_info(df, y):

    lr_model = train_model(df.values, y)
    
    coefs = dict(zip(df.keys(), lr_model.coef_))
    intercept = lr_model.intercept_
    
    return {"coefs": coefs, "intercept": intercept}


# train individualized model
# using coefs from average model for strategies that are under-annotated
def get_ind_model_info(df_feat, scores, avg_coefs, min_cnt = 15):
    
    feat_counts = dict(df_feat.sum(axis=0))
    
    good_feats = [k for k,v in feat_counts.items() if v > min_cnt]
    
    # if certain strategies have too little annotations from the individual
    # we consider directly using the "average-person" view 
    low_count_feats = {k for k,v in feat_counts.items() if v <= min_cnt}

    # adjust scores for low counts
    for feat in low_count_feats: 
    
        feat_ids = df_feat[df_feat[feat]==1].index
    
        for idx in feat_ids:

            # embed mean coef into "score" (i.e., fixing coefs for these low-count features)
            scores[idx] -= avg_coefs[feat]
    
    df_feat_sel = df_feat[good_feats]

    df = df_feat_sel
    y = [scores[idx] for idx in df_feat_sel.index]
    
    ind_model = get_model_info(df, y)
    ind_model['coefs'].update({feat: avg_coefs[feat] for feat in low_count_feats})
    
    return ind_model


# estimate the level of politeness according to a perception model
def estimate_perception(model, strategy_set):
    
    return sum([model['coefs'][s] for s in strategy_set]) + model['intercept']