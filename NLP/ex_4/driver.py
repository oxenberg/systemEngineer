
import tagger

chosen_model = 'HMM'

tagged_sentences = {}

train_path = "data/en-ud-train.upos.tsv"
dev_path = "data/en-ud-dev.upos.tsv"

train_data = tagger.load_annotated_corpus(train_path)
dev_data = tagger.load_annotated_corpus(dev_path)


[allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] = tagger.learn_params(train_data)

gold_sentence = dev_data[0]
pred_sentence = [w[0] for w in gold_sentence]
print(f"tested sentence is {gold_sentence} of length {len(pred_sentence)}")

baseline_tagged_sentence = tagger.baseline_tag_sentence
tagged_sentences['baseline'] = baseline_tagged_sentence
HMM_tagged_sentence= tagger.hmm_tag_sentence
tagged_sentences['HMM'] = HMM_tagged_sentence

pred_model = tagged_sentences[chosen_model]
tagged_sentence = pred_model(pred_sentence, A,B)
correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)

print(f"correct: {correct}, correctOOV: {correctOOV}, OOV: {OOV}")


score_nom, score_denom = 0, 0
for gold_sentence in dev_data:
    pred_sentence = [w[0] for w in gold_sentence]
    tagged_sentence = pred_model(pred_sentence, A,B)
    correct, correctOOV, OOV = tagger.count_correct(gold_sentence, tagged_sentence)
    score_nom += correct
    score_denom += len(pred_sentence)

print(f"baseline score is {score_nom/score_denom}")



