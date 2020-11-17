from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import torch

SPECIAL_TOKENS = ['<STR>','<CONTEXT>','<START>','<END>', \
                      '<Actually>',
                      '<Adverb.Just>',
                      '<Affirmation>',
                      '<Apology>',
                      '<By.The.Way>',
                      '<Indicative>',
                      '<Conj.Start>',
                      '<Subjunctive>',
                      '<Filler>',
                      '<For.Me>',
                      '<For.You>',
                      '<Gratitude>',
                      '<Hedges>',
                      '<Greeting>',
                      '<Please>',
                      '<Please.Start>',
                      '<Reassurance>',
                      '<Swearing>']

# Adapted from https://github.com/agaralabs/transformer-drg-style-transfer/blob/master/OpenAI_GPT_Pred.ipynb
# Author: Bhargav5
def preditction_with_beam_search(ref_text, tokenizer, model, device, beam_width=3):
    """
    This function decodes sentences using Beam Seach. 
    It will output #sentences = beam_width. This function works on a single example.
    
    ref_text : string : Input sentence
    beam_width : int : Width of the output beam
    vocab_length : int : Size of the Vocab after adding the special tokens
    """
    vocab_length = max(tokenizer.special_tokens.values()) + 1
    
    done = [False for i in range(beam_width)] # To track which beams are already decoded
    stop_decode = False
    decoded_sentences=[] # List of decoded sentences at any given time
    
    sm = torch.nn.Softmax(dim=-1) # To calculate Softmax over the final layer Logits
    tokens = tokenizer.tokenize(ref_text) # Tokenize the input text
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens) # Convert tokens to ids
    index_tokens = [indexed_tokens for i in range(beam_width)] # Replication of Input ids for all the beams
    
    #index_tokens = [indexed_tokens for i in range(beam_width)]
    torch_tensor = torch.tensor(index_tokens).to(device)
    beam_indexes = [[] for i in range(beam_width)] # indexes of the current decoded beams
    best_scoes = [0 for i in range(beam_width)] # A list of lists to store Probability values of each decoded token of best beams
    count = 0
    
    # Enforce shorter sents 
    # while count < model.config.n_positions and not stop_decode:
    while count < len(ref_text.split())*2 and not stop_decode:
        if count == 0: # For the first step when only one sentence is availabe
            with torch.no_grad():
                # Calculate output probability distribution over the Vocab,
                preds = sm(model(torch_tensor)) #  shape = [beam_bidth, len(input_sen)+1,Vocab_length]
            top_v, top_i = preds[:,-1,:].topk(beam_width) # Fatch top indexes and it's values
            [beam_indexes[i].append(top_i[0][i].tolist()) for i in range(beam_width)] # Update the Beam indexes
            # Update the best_scores, for first time just add the topk values directly
            for i in range(beam_width):
                best_scoes[i] = top_v[0][i].item()
            count += 1
        else: # After first step
            # Prepare the current_state by concating original input and decoded beam indexes
            current_state = torch.cat((torch_tensor, torch.tensor(beam_indexes).to(device)), dim=1)
            # Prediction on the current state
            with torch.no_grad():
                preds = sm(model(current_state))
            # Multiply new probability predictions with corresponding best scores
            # Total socres = beam_width * Vocab_Size
            flatten_score = (preds[:,-1,:]*torch.tensor(best_scoes).to(device).unsqueeze(1)).view(-1)
            # Fatch the top scores and indexes 
            vals, inx = flatten_score.topk(beam_width)
            # best_score_inx saves the index of best beams after multiplying the probability of new prediction
            
            # best_scoes_inx = (inx / vocab_length).tolist()
            best_scoes_inx = torch.floor_divide(inx, vocab_length).tolist()
            
            best_scoes = vals.tolist()
            # Unflatten the index 
            correct_inx = (inx % vocab_length).tolist()
            
            # Check if done for all the Beams
            for i in range(beam_width):
                if correct_inx[i] == tokenizer.special_tokens["<END>"]:
                    done[i] = True
            # Update the best score for each the current Beams
            for i in range(beam_width):
                if not done[i]:
                    best_scoes[i] = vals.tolist()[i]
            # Check is All the Beams are Done
            if (sum(done) == beam_width):
                stop_decode = True
            # Prepapre the new beams
            temp_lt=[0 for i in range(beam_width)]
            for i,x in enumerate(best_scoes_inx):
                temp_lt[i] = beam_indexes[x] + [correct_inx[i]]
            # Update the Beam indexes
            beam_indexes = temp_lt
            del temp_lt
            count += 1
            
    # Decode All the beam indexes to till <END> token only and convert into sentence
    for i in range(beam_width):
        try:
            end_index = beam_indexes[i].index(tokenizer.special_tokens["<END>"])
        except ValueError:
            end_index = len(beam_indexes[i])
            
        decoded_sentences.append(tokenizer.decode(beam_indexes[i][:end_index]))
        
    return decoded_sentences


def load_model(model_path, special_tokens=SPECIAL_TOKENS):
    
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', special_tokens=special_tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', num_special_tokens=len(special_tokens))
    
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return tokenizer, device, model

