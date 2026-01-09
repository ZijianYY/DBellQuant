import torch
from utils.autosearch import structural_searching
from utils.autosearch import structural_searching_new
from utils.autosearch import structural_searching_w
from utils.mask import generate_structural_mask

'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution(tmp, H=None, metric="magnitude", up_lim=30):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split, mask3 = structural_searching(target_weights, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3

def structural_guassian_distribution_new(tmp,inp,name,blocki, H=None, metric="magnitude", up_lim=30):
    if metric == "hessian":
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split, mask3 = structural_searching_new(target_weights, inp, name, blocki,up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3

def structural_guassian_distribution_weight(tmp, H=None, metric="magnitude", up_lim=30):

    target_weights = tmp


    # optimal_split, mask3 = structural_searching_w(target_weights, up_lim)
    # mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    # return mask1, mask2, mask3
    optimal_split, mask3 = structural_searching(target_weights, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3

def structural_guassian_distribution_weight_2(tmp, H=None, metric="magnitude", up_lim=30):
    
    target_weights = tmp

    true_counts = target_weights.abs().sum(dim=0)

    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, 20)

    mask3 = torch.full((target_weights.shape[0], target_weights.shape[1]), False).to(target_weights.device)
    mask3[:, top_braq_2_columns] = True


    # optimal_split, mask3 = structural_searching_w(target_weights, up_lim)
    # mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    # return mask1, mask2, mask3
    # optimal_split, mask3 = structural_searching(target_weights, up_lim)
    # mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)

    mask4 = ~mask3

    print( mask3.sum() / mask3.numel())
    return mask3, mask4