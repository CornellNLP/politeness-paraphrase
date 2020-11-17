### Models ###  

# Note: change to where the downloaded model is stored 
GEN_MODEL_PATH = "/reef/exp_liye/politeness_paraphrase/models/politeness_gen.bin"

PERCEPTION_MODEL_PATH = "data/perceptions/average.json"

### MARKER MANIPULATION SETTINGS ### 

STRATEGIES = ['Actually','Adverb.Just','Affirmation','Apology','By.The.Way',
 'Conj.Start', 'Filler', 'For.Me', 'For.You', 'Gratitude', 'Greeting',
 'Hedges', 'Indicative', 'Please', 'Please.Start',
 'Reassurance', 'Subjunctive', 'Swearing']

# deletion mode for each strategy
MARKER_DELETION_MODE = {'Actually': "token",
                        'Adverb.Just': "token",
                        'Affirmation': "segment",
                        'Apology': "segment",
                        'By.The.Way': "token",
                        'Indicative': "token",
                        'Conj.Start': "token",
                        'Subjunctive': "token",
                        'Filler': "token",
                        'For.Me': "token",
                        'For.You': "token",
                        'Gratitude': "segment",
                        'Hedges': "token",
                        'Greeting': "token",
                        'Please': "token",
                        'Please.Start': "token",
                        'Reassurance': "segment",
                        'Swearing': "token"}

# common punctuations that segments sentences
PUNCS = [',', ";", "-", ":"] 

### CONSTRAINTS ### 

# max number of strategies to be added 
UPPER_BOUND = 3

