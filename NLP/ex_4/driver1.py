
import tagger2
import tagger
import taggercopy
import torch

chosen_model = 'HMM'

tagged_sentences = {}

train_path = "data/en-ud-train.upos.tsv"
dev_path = "data/en-ud-dev.upos.tsv"

train_data = tagger.load_annotated_corpus(train_path)
dev_data = tagger.load_annotated_corpus(dev_path)


# [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] = tagger.learn_params(train_data)

# gold_sentence = dev_data[0]
# pred_sentence = [w[0] for w in gold_sentence]
# print(f"tested sentence is {gold_sentence} of length {len(pred_sentence)}")



# baseline_tagged_sentence = tagger.baseline_tag_sentence
# tagged_sentences['baseline'] = baseline_tagged_sentence
# HMM_tagged_sentence= tagger.hmm_tag_sentence
# tagged_sentences['HMM'] = HMM_tagged_sentence

# pred_model = tagged_sentences[chosen_model]
# tagged_sentence = pred_model(pred_sentence, A,B)
# correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)

# print(f"correct: {correct}, correctOOV: {correctOOV}, OOV: {OOV}")






# models = {'baseline': [perWordTagCounts, allTagCounts],
#           'hmm': [A,B]
#           }

# ####### bilstm
# dict_params = tagger.get_best_performing_model_params()
# model_bilstm = tagger.initialize_rnn_model(dict_params)
# tagger.train_rnn(model_bilstm, train_data)

# ####### bilstm case based
# dict_params = tagger.get_best_performing_model_params()
# model_case_base = tagger.initialize_rnn_model(dict_params)
# tagger.train_rnn(model_case_base, train_data)

# models["blstm"] = model_bilstm
# models["cblstm"] = model_case_base





# score_nom, score_denom = 0, 0
# for gold_sentence in dev_data:
#     pred_sentence = [w[0] for w in gold_sentence]
    
#     tagged_sentence = pred_model(pred_sentence, A,B)
#     correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)
#     score_nom += correct
#     score_denom += len(pred_sentence)

# print(f"baseline score is {score_nom/score_denom}")

#### Check BILSTM: 

####### bilstm
dict_params = taggercopy.get_best_performing_model_params()
model = taggercopy.initialize_rnn_model(dict_params)
taggercopy.train_rnn(model, train_data)



score_nom, score_denom = 0, 0
for gold_sentence in dev_data:
    pred_sentence = [w[0] for w in gold_sentence]
    tagged_sentence = taggercopy.rnn_tag_sentence(pred_sentence, model)
    correct, correctOOV, OOV = taggercopy.count_correct(gold_sentence, tagged_sentence)
    score_nom += correct
    score_denom += len(pred_sentence)

print(f"baseline score for bilstm is {score_nom/score_denom}")










