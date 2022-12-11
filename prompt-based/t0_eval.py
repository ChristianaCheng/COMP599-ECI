from sklearn.metrics import precision_recall_fscore_support
pred_fname = "t0_data/test0.9_onlyinp_noguide_int.txt"

def compute_metrics(labels,preds):
  precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
  return {'f1': f1,'precision': precision,'recall': recall}

def file_to_list(fpath):
	target = open(fpath,"r")
	labels = target.read().split('\n')
	labels = [int(i) for i in labels if i != ""]
	return labels

labels = file_to_list("t0_data/test_intra_labels.txt")
preds = file_to_list(pred_fname)

print(compute_metrics(labels,preds))

