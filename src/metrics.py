
import json, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix
from pytorch_lightning.utilities.rank_zero import rank_zero_only  

def overall_accuracy(npcm):
    oa = np.trace(npcm)/npcm.sum()
    return 100*oa 

def class_IoU(npcm,n_class):
    ious = 100 * np.diag(npcm) / (np.sum(npcm, axis=1) + np.sum(npcm, axis=0) - np.diag(npcm))
    ious[np.isnan(ious)] = 0
    return ious, np.mean(ious)

def class_precision(npcm):
    precision = 100 * np.diag(npcm) / np.sum(npcm, axis=0)
    precision[np.isnan(precision)] = 0
    return precision, np.mean(precision)

def class_recall(npcm):
    recall = 100 * np.diag(npcm) / np.sum(npcm, axis=1)
    recall[np.isnan(recall)] = 0
    return recall, np.mean(recall)

def class_fscore(precision, recall):
    fscore = 2 * (precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    return fscore, np.mean(fscore)

@rank_zero_only
def metrics(config: dict, path_preds: str, remove_preds=False):
    gt_csv = pd.read_csv(config["paths"]["test_csv"], header=None)
    truth_images = gt_csv.iloc[:,0].to_list()
    truth_msks = gt_csv.iloc[:,1].to_list()
    preds_msks = [Path(path_preds.as_posix(), i.split('/')[-1].replace(i.split('/')[-1], 'PRED_'+i.split('/')[-1])).as_posix() for i in truth_images]
    assert len(truth_msks) == len(preds_msks), '[WARNING !] mismatch number of predictions and test files.'  
    print('-- Calculating metrics --')
    patch_confusion_matrices = []
    for u in range(len(truth_msks)):
        try:
            target = np.array(Image.open(truth_msks[u]))-1 
            preds = np.array(Image.open(preds_msks[u]))         
            patch_confusion_matrices.append(confusion_matrix(target.flatten(), preds.flatten(), labels=list(range(int(len(config["classes"]))))))
        except Exception as e:
            print(f"Error at index {u}: {e}")
    sum_confmat = np.sum(patch_confusion_matrices, axis=0)  
    #### CLEAN REGARDING WEIGHTS FOR METRICS CALC : 
    weights = np.array([config["classes"][i][0] for i in config["classes"]])
    unused_classes = np.where(weights == 0)[0]
    confmat_cleaned = np.delete(sum_confmat, unused_classes, axis=0)  # remove rows
    confmat_cleaned = np.delete(confmat_cleaned, unused_classes, axis=1)  # remove columns) == 0)[0]
    
    per_c_ious, avg_ious = class_IoU(confmat_cleaned, len(np.nonzero(weights)[0]))
    ovr_acc = overall_accuracy(confmat_cleaned)
    per_c_precision, avg_precison = class_precision(confmat_cleaned)
    per_c_recall, avg_recall= class_recall(confmat_cleaned)
    per_c_fscore, avg_fscore = class_fscore(per_c_precision, per_c_recall)
    
    metrics = {
    'Avg_metrics_name' : ['mIoU', 'Overall Accuracy', 'Fscore', 'Precision', 'Recall'],
    'Avg_metrics' : [avg_ious, ovr_acc, avg_fscore, avg_precison, avg_recall],
    'classes' : list(np.array([config["classes"][i][1] for i in config["classes"]])[np.nonzero(weights)[0]]),
    'per_class_iou' : list(per_c_ious),
    'per_class_fscore' : list(per_c_fscore),
    'per_class_precision' : list(per_c_precision),
    'per_class_recall' : list(per_c_recall),  
    }

    out_folder_metrics = Path('/'.join(path_preds.as_posix().split('/')[:-1]), 'metrics')
    out_folder_metrics.mkdir(exist_ok=True, parents=True)
    np.save(out_folder_metrics.as_posix()+'/confmat.npy', sum_confmat)
    json.dump(metrics, open(out_folder_metrics/Path('metrics.json'), 'w'))
    
    print('')
    print('Global Metrics: ')
    print('-'*90)
    for metric_name, metric_value in zip(metrics['Avg_metrics_name'], metrics['Avg_metrics']):
        print(f"{metric_name:<20s} {metric_value:<20.4f}")
    print('-'*90 + '\n\n')
    # Separate classes into used and unused based on weight
    used_classes = {k: v for k, v in config["classes"].items() if v[0] != 0}
    unused_classes = {k: v for k, v in config["classes"].items() if v[0] == 0}
    # class metrics
    def print_class_metrics(class_dict, metrics_available=True):
        for class_index, class_info in class_dict.items():
            class_weight, class_name = class_info
            if metrics_available:
                i = metrics['classes'].index(class_name)  # Get the index of the class in the metrics
                print("{:<25} {:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    class_name, class_weight, metrics['per_class_iou'][i], 
                    metrics['per_class_fscore'][i], metrics['per_class_precision'][i], 
                    metrics['per_class_recall'][i]))
            else:
                print("{:<25} {:<15}".format(class_name, class_weight))
    print("{:<25} {:<15} {:<10} {:<10} {:<10} {:<10}".format('Class', 'Weight', 'IoU', 'F-score', 'Precision', 'Recall'))
    print('-'*65)
    print_class_metrics(used_classes)
    print("\nUnused Classes:")
    print("{:<25} {:<15}".format('Class', 'Weight'))
    print('-'*65)
    print_class_metrics(unused_classes, metrics_available=False)
    print('\n\n')

    if remove_preds:
        shutil.rmtree(path_preds)