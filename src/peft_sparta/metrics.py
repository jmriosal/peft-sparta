def mcc(cm):
    """
    Matthew's correlation coefficient (MCC) a.k.a. Phi coefficient
    of a 2x2 confusion matrix is its Pearson correlation coefficient 
    Use: measures the quality of a binary classifier
    """
    assert cm.shape == (2,2)
    numerator = cm[0,0]*cm[1,1]-cm[0,1]*cm[1,0]  
    denominator = (cm.sum(0).prod()*cm.sum(1).prod()).sqrt()
    if denominator == 0 and numerator == 0: # limit case
        mcc = 0
    else:
        mcc = (numerator / denominator).item()
    return mcc


def f1_score(cm):
    assert cm.shape == (2,2)
    return (2*cm[1,1]/(2*cm[1,1]+cm[0,1]+cm[1,0])).item()
    

def bacc(cm):
    """balanced accuracy"""
    return (cm.diag() / cm.sum(dim=1)).mean().item()
