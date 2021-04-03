def rounder(test_preds):
    
    coef = [0.5, 1.5, 2.5, 3.5]
    new_preds = []
    for i, pred in enumerate(test_preds):
        
        if pred < coef[0]:
            new_preds.append(0)
        elif pred >= coef[0] and pred < coef[1]:
            new_preds.append(1)
        elif pred >= coef[1] and pred < coef[2]:
            new_preds.append(2)
        elif pred >= coef[2] and pred < coef[3]:
            new_preds.append(3)
        else:
            new_preds.append(4)
    return new_preds