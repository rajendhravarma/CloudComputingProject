from joblib import dump, load

# # Load the Vocab model
# loaded_vocab = load('title_vocab_v4.joblib')

# # Load the algorithm models
# LogisticRegression_Model = load('LogisticRegression_v2.joblib')

unique_tags_dict = load('unique_tags_dict')
tag_id_cf = [810, 4074, 6701, 8596, 9303, 24182, 26612, 27974, 31382]
labels_cf= {i:unique_tags_dict[i] for i in tag_id_cf}
print(labels_cf)