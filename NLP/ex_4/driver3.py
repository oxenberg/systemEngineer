import tagger as pt
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dev_path = r'en-ud-dev.upos.tsv'
train_path = r'en-ud-train.upos.tsv'
emb_path = r'glove.6B.100d.txt'

train_data = pt.load_annotated_corpus(train_path)
dev_data = pt.load_annotated_corpus(dev_path)
params = pt.get_best_performing_model_params()  # test get_best_performing_model_params 
split_sentences = [list(zip(*sentence)) for sentence in dev_data]  # [ [ [token, ...], [tag, ...] ], ...]


def report_results(sentences, model):
    results = []
    for sentence, tags in sentences:
        tagged_sentence = pt.tag_sentence(sentence, model)  # test tag_sentence
        
        correct, correctOOV, OOV = pt.count_correct(list(zip(sentence, tags)), tagged_sentence)  # test count_correct
        
        oov_acc = correctOOV / OOV if OOV != 0 else np.nan
        
        results.append((correct / len(sentence), oov_acc))
        
    res = np.array(results)
    
    print(f'\n{list(model.keys())[0]} tagger:\n\tTotal Accuracy: {res[:, 0].mean():.2f}\n\tOOV Accuracy: {np.nanmean(res[:, 1]):.2f}\n')
    
    
### LSTMs ###

params['max_vocab_size'] = -1  # test unlimited vocab size
params['min_frequency'] = 1  # test full vocab
params['input_rep'] = 1  # change to cblstm model
case_model = pt.initialize_rnn_model(params)
pt.train_rnn(case_model, train_data, dev_data)  # test with dev_data

params['max_vocab_size'] = 4000  # test (max_vocab < vocab) case
params['min_frequency'] = 2
params['input_rep'] = 0  # change to vanilla model
vanilla_model = pt.initialize_rnn_model(params)
pt.train_rnn(vanilla_model, train_data)  # test without dev_data

# TODO - add tests for various num_of_layers, output_dimension

case_model = {'cblstm': case_model}
vanilla_model = {'blstm': vanilla_model}

### Baseline & HMM ###

pt.allTagCounts, pt.perWordTagCounts, pt.transitionCounts, pt.emissionCounts, pt.A, pt.B = pt.learn_params(train_data)

hmm_model = {'hmm': [pt.A, pt.B]}
baseline_model = {'baseline': [pt.perWordTagCounts, pt.allTagCounts]}


report_results(split_sentences, hmm_model)
report_results(split_sentences, baseline_model)
report_results(split_sentences, case_model)
report_results(split_sentences, vanilla_model)
