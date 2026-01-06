import numpy as np
import sklearn.metrics as sk

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1., debug=False):
    """
    Calculate FPR at a given TPR (recall) level.
    
    Args:
        y_true: ground truth labels (1 for positive/ID, 0 for negative/OOD)
        y_score: predicted scores (higher = more likely ID)
        recall_level: target TPR level (e.g., 0.95 for TPR95)
        pos_label: label for positive class
        debug: print debug information
    
    Returns:
        FPR at the given TPR level
    """
    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    
    # Total positives and negatives
    n_pos = np.sum(y_true)
    n_neg = np.sum(~y_true)
    
    if debug:
        print(f"  [FPR95 Debug] n_pos={n_pos}, n_neg={n_neg}")
        print(f"  [FPR95 Debug] score range: [{float(y_score.min()):.4f}, {float(y_score.max()):.4f}]")
    
    if n_pos == 0 or n_neg == 0:
        return 1.0
    
    # sort scores and corresponding truth values (descending order)
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(y_true_sorted)
    fp_cumsum = np.cumsum(~y_true_sorted)
    
    # Calculate TPR and FPR at each threshold
    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg
    
    if debug:
        # Find where TPR crosses 95%
        cross_idx = np.where(tpr >= recall_level)[0]
        if len(cross_idx) > 0:
            idx = cross_idx[0]
            print(f"  [FPR95 Debug] TPR95 at idx={idx}: threshold={float(y_score_sorted[idx]):.4f}, TPR={float(tpr[idx]):.4f}, FPR={float(fpr[idx]):.4f}")
            print(f"  [FPR95 Debug] TP={int(tp_cumsum[idx])}, FP={int(fp_cumsum[idx])}")
        else:
            print(f"  [FPR95 Debug] Could not achieve TPR >= {recall_level}")
            print(f"  [FPR95 Debug] Max TPR: {float(tpr.max()):.4f}")
    
    # Find the threshold where TPR >= recall_level
    valid_indices = np.where(tpr >= recall_level)[0]
    
    if len(valid_indices) == 0:
        # If we can't achieve the target recall, return worst case
        if debug:
            print(f"  [FPR95 Debug] Returning 1.0 (cannot achieve target recall)")
        return 1.0
    
    # Use the FIRST index where TPR >= recall_level
    # This gives us the FPR when TPR just reaches 95%
    idx = valid_indices[0]
    
    if debug:
        print(f"  [FPR95 Debug] Using idx={idx} (first valid), TPR={float(tpr[idx]):.4f}, FPR={float(fpr[idx]):.4f}")
    
    return float(fpr[idx])


def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    # Concatenate and flatten to 1D
    examples = np.concatenate((in_examples, out_examples)).flatten()

    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)
    recall_level = 0.95

    # Handle potential errors in FPR95 calculation
    try:
        fpr = fpr_and_fdr_at_recall(labels, examples, recall_level, debug=True)
        # Check for NaN/inf
        if not np.isfinite(fpr):
            print(f"Warning: FPR95 is {fpr}, setting to 1.0")
            fpr = 1.0
        # Clip to valid range
        fpr = np.clip(fpr, 0.0, 1.0)
    except Exception as e:
        print(f"Warning: FPR95 calculation failed: {e}, setting to 1.0")
        fpr = 1.0

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples_neg = -1 * examples  # Already flattened above
    aupr_out = sk.average_precision_score(labels_rev, examples_neg)
    return auroc, aupr_in, aupr_out, fpr

def evaluate_all(in_scores, out_scores):
    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))
    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    return {
        'auroc': auroc,
        'aupr_in': aupr_in,
        'aupr_out': aupr_out,
        'fpr95': fpr95
    }

## from knn-ood github
# def cal_metric(known, novel, method=None):
#     tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
#     results = dict()
#     # FPR
#     mtype = 'FPR'
#     results[mtype] = fpr_at_tpr95
#
#     # AUROC
#     mtype = 'AUROC'
#     tpr = np.concatenate([[1.], tp/tp[0], [0.]])
#     fpr = np.concatenate([[1.], fp/fp[0], [0.]])
#     results[mtype] = -np.trapz(1.-fpr, tpr)
#
#     # AUIN
#     mtype = 'AUIN'
#     denom = tp+fp
#     denom[denom == 0.] = -1.
#     pin_ind = np.concatenate([[True], denom > 0., [True]])
#     pin = np.concatenate([[.5], tp/denom, [0.]])
#     results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
#
#     # AUOUT
#     mtype = 'AUOUT'
#     denom = tp[0]-tp+fp[0]-fp
#     denom[denom == 0.] = -1.
#     pout_ind = np.concatenate([[True], denom > 0., [True]])
#     pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
#     results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
#
#     return results['AUROC'], results['AUIN'], results['AUOUT'], results['FPR']
#
# def get_curve(known, novel, method=None):
#     tp, fp = dict(), dict()
#     fpr_at_tpr95 = dict()
#
#     known.sort()
#     novel.sort()
#
#     end = np.max([np.max(known), np.max(novel)])
#     start = np.min([np.min(known),np.min(novel)])
#
#     all = np.concatenate((known, novel))
#     all.sort()
#
#     num_k = known.shape[0]
#     num_n = novel.shape[0]
#
#     if method == 'row':
#         threshold = -0.5
#     else:
#         threshold = known[round(0.05 * num_k)]
#
#     tp = -np.ones([num_k+num_n+1], dtype=int)
#     fp = -np.ones([num_k+num_n+1], dtype=int)
#     tp[0], fp[0] = num_k, num_n
#     k, n = 0, 0
#     for l in range(num_k+num_n):
#         if k == num_k:
#             tp[l+1:] = tp[l]
#             fp[l+1:] = np.arange(fp[l]-1, -1, -1)
#             break
#         elif n == num_n:
#             tp[l+1:] = np.arange(tp[l]-1, -1, -1)
#             fp[l+1:] = fp[l]
#             break
#         else:
#             if novel[n] < known[k]:
#                 n += 1
#                 tp[l+1] = tp[l]
#                 fp[l+1] = fp[l] - 1
#             else:
#                 k += 1
#                 tp[l+1] = tp[l] - 1
#                 fp[l+1] = fp[l]
#
#     j = num_k+num_n-1
#     for l in range(num_k+num_n-1):
#         if all[j] == all[j-1]:
#             tp[j] = tp[j+1]
#             fp[j] = fp[j+1]
#         j -= 1
#
#     fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)
#
#     return tp, fp, fpr_at_tpr95
#
# def evaluate_all(in_scores, out_scores):
#     in_examples = in_scores.reshape((-1,))
#     out_examples = out_scores.reshape((-1,))
#     auroc, aupr_in, aupr_out, fpr95 = cal_metric(in_examples, out_examples)
#     return auroc, aupr_in, aupr_out, fpr95