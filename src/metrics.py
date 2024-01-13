import numpy as np
import pandas as pd
import logging
from PIL import Image
from sklearn.metrics import confusion_matrix
from pathlib import Path





def generate_miou(config: dict, path_preds: str) -> list:

    def calc_miou(cm_array):
        m = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
        m = np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()
        return m.astype(float), ious      

    #################################################################################################

    gt_csv = pd.read_csv(config["paths"]["test_csv"], header=None)

    truth_images = gt_csv.iloc[:,0].to_list()
    truth_msks = gt_csv.iloc[:,1].to_list()

    preds_msks = []
    for i in truth_images:
        preds_msks.append(Path(path_preds.as_posix(), i.split('/')[-1].replace(i.split('/')[-1], 'PRED_'+i.split('/')[-1])).as_posix())

    try: 
        assert len(truth_msks) == len(preds_msks), '[WARNING !] mismatch number of predictions and test files.'
    except AssertionError as e:
        logging.error(str(e))
        raise SystemExit()         

    ######
    ###### TO ADD : FILTER FROM WEIGHTS FOR mIOU CALC ? 

    patch_confusion_matrices = []

    for u in range(len(truth_msks)):
        target = np.array(Image.open(truth_msks[u]))-1 
        preds = np.array(Image.open(preds_msks[u]))         
        patch_confusion_matrices.append(confusion_matrix(target.flatten(), preds.flatten(), labels=list(range(int(len(config["classes"]))))))

    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    miou, ious = calc_miou(sum_confmat) 

    return sum_confmat, miou, ious


