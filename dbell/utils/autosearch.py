from re import L
import numpy as np
from pyparsing import line
import torch
from binary import high_order_residual
from utils.mask import generate_structural_mask

def unravel_index(indices, shape):
    """将 1D 索引转换为多维坐标"""
    coords = []
    for dim in reversed(shape):
        coords.append(indices % dim)
        indices = indices // dim
    coords = torch.stack(coords[::-1], dim=-1)  # 按正确顺序堆叠
    return tuple(coords.t())  # 转置后返回元组

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    
    # error_matrix = origin_matrix - quantized_matrix
    
    # # 将 origin_matrix 展平并找到前 100 个最大值及其索引
    # topk_values, topk_indices = torch.topk(origin_matrix.flatten(), 100)
    
    # # 创建与 origin_matrix 相同形状的权重矩阵，初始值为 1
    # weights = torch.ones_like(origin_matrix)
    
    # # 按照 topk_indices 更新权重矩阵对应位置的值为 2
    # # 使用 unravel_index 将一维索引转换为多维索引
    # topk_indices_multi_dim = unravel_index(topk_indices, origin_matrix.shape)
    # weights[topk_indices_multi_dim] = 2
    
    # # 按权重计算加权 MSE
    # weighted_mse = torch.mean(weights * (error_matrix ** 2))
    
    return mse
    
    
    # a = ((origin_matrix - quantized_matrix) ** 2)
    # top_values, top_indices = torch.topk(a.view(-1), 20)
    # mse = torch.sum(top_values)
    
    # abs_origin = torch.abs(origin_matrix)
    # abs_origin_flat = abs_origin.view(-1)  # 展平为 1D 张量

    # # 获取绝对值最大的 100 个值的索引
    # top_100_values, top_100_indices = torch.topk(abs_origin_flat, k=100, largest=True)

    # # 将 1D 索引转换为原始 shape 的多维坐标
    # top_100_coords = unravel_index(top_100_indices, origin_matrix.shape)

    # # 3. 在 quantized_matrix 中找到对应位置的值
    # quantized_values_at_top_100 = quantized_matrix[top_100_coords]

    # # 4. 计算损失
    # # 获取 origin_matrix 中对应的 100 个值
    # origin_values_at_top_100 = origin_matrix[top_100_coords]

    # # 示例：使用平方差作为损失
    # loss = torch.mean((origin_values_at_top_100 - quantized_values_at_top_100) ** 2)


    # return mse

def calculate_percentage_and_variance_original(weights, abs_weights, bin_edges):
    percentages = []
    variances = []
    accum_percentages = [0]
    total_elements = abs_weights.numel()
    for i in range(len(bin_edges) - 1):
        bin_mask = (abs_weights >= bin_edges[i]) & (abs_weights < bin_edges[i + 1])
        bin_weights = weights[bin_mask]
        percentages.append(bin_weights.numel() / total_elements * 100)
        accum_percentages.append(accum_percentages[-1] + percentages[-1])
        variances.append(torch.var(bin_weights))
    return percentages, variances, accum_percentages

'''
Include main method to search the rate for 2-bit salient data columns and the optimal split for 1-bit data
'''
def structural_searching(origin_matrix, up_lim=30):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)
        group4 = high_order_residual(origin_matrix, ~mask3, order=1)
        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)
        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    search_matrix = origin_matrix * (~mask3)

    flat_abs_tensor = torch.abs(search_matrix).view(-1)
    percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    ).to(origin_matrix.device)

    # search for the optimal split for the second group, high order=1,, non-structured search
    for split_value in percentile_values:
        mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
        group1 = high_order_residual(origin_matrix, mask1, order=1)
        group2 = high_order_residual(origin_matrix, mask2, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value
        tmp = torch.max(torch.abs(search_matrix)).item()
    
    return optimal_split, mask3

def structural_searching_new(origin_matrix,inp,name, blocki, up_lim=30):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)

        group4 = high_order_residual(origin_matrix, ~mask3, order=1)
        quantize_error_0 = error_computing(inp[:,:,blocki*128:(blocki+1)*128].float() @ origin_matrix.T, inp[:,:,blocki*128:(blocki+1)*128].float() @(group4+group3).T)
        # error.append(quantize_error_0.item())
        lines.append(i)
        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    search_matrix = origin_matrix * (~mask3)

    flat_abs_tensor = torch.abs(search_matrix).view(-1)
    percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    ).to(origin_matrix.device)

    # search for the optimal split for the second group, high order=1,, non-structured search
    for split_value in percentile_values:
        mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
        group1 = high_order_residual(origin_matrix, mask1, order=1)
        group2 = high_order_residual(origin_matrix, mask2, order=1)

        quantize_error = error_computing(inp[:,:,blocki*128:(blocki+1)*128].float() @origin_matrix.T, inp[:,:,blocki*128:(blocki+1)*128].float() @(group1+group2+group3).T)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value
        tmp = torch.max(torch.abs(search_matrix)).item()
    
    return optimal_split, mask3

def find_optimal_split(group_max, origin_matrix, border):
    optimal_split = None
    minimal_value = float('inf')
    searching_steps = torch.arange(0.1,0.8,0.01)
    searching_steps = searching_steps * group_max

    group3 = high_order_residual(origin_matrix, torch.abs(origin_matrix) > border, order=2)
    for split_value in searching_steps:

        group1 = high_order_residual(origin_matrix, (torch.abs(origin_matrix) > split_value) & (torch.abs(origin_matrix) <= border), order=1)
        group2 = high_order_residual(origin_matrix, torch.abs(origin_matrix) <= split_value, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value

    return optimal_split, minimal_value

def structural_searching_w(origin_matrix, up_lim=30):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    # for i in range(1, up_lim):
    #     mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    #     mask3[:, top_braq_2_columns[:i]] = True
    #     group3 = high_order_residual(origin_matrix, mask3, order=2)

    #     group4 = high_order_residual(origin_matrix, ~mask3, order=1)
    #     quantize_error_0 = error_computing(origin_matrix, group4+group3)
    #     error.append(quantize_error_0.item())
    #     lines.append(i)
    #     if quantize_error_0 < minimal_value_0:
    #         minimal_value_0 = quantize_error_0
    optimal_split_0 = 20

    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    search_matrix = origin_matrix * (~mask3)

    flat_abs_tensor = torch.abs(search_matrix).view(-1)
    # percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentiles = torch.tensor([0.5]).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    ).to(origin_matrix.device)
    optimal_split = percentile_values
    # search for the optimal split for the second group, high order=1,, non-structured search
    # for split_value in percentile_values:
    #     mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
    #     group1 = high_order_residual(origin_matrix, mask1, order=1)
    #     group2 = high_order_residual(origin_matrix, mask2, order=1)

    #     quantize_error = error_computing(origin_matrix, group1+group2+group3)
    #     if quantize_error < minimal_value:
    #         minimal_value = quantize_error
    #         optimal_split = split_value
    #     tmp = torch.max(torch.abs(search_matrix)).item()
    
    return optimal_split, mask3