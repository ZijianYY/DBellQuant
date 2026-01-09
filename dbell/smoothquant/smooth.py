import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
# from transformers.models.mixtral.modeling_mixtral import (
#     MixtralDecoderLayer,
#     MixtralRMSNorm,
# )
# from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
import numpy as np
import matplotlib.pyplot as plt
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0]
    weight_scales = weight_scales.to(torch.float32).clamp(min=1e-5)
    # weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales,index, a, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    # weight_scales = torch.cat(
    #     [fcs[0].weight.abs().max(dim=0, keepdim=True)[0] ], dim=0
    # )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # scales = torch.full((4096,), 2).to(device=device, dtype=dtype)
    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    # #     # 将 tensor 分成每 128 个一组
    # groups = weight_scales.view(-1, 128)

    # # 找到每一组中最大的两个值的索引
    # _, max_indices = torch.topk(groups, k=4, dim=1)

    # # 将组内索引转换为原始 tensor 的索引
    # original_indices = max_indices + (torch.arange(groups.size(0), device=weight_scales.device).view(-1, 1) * 128)

    # original_indices_flat1 = original_indices.flatten()
    
    # # groups2 = act_scales.view(-1, 128)

    # # # 找到每一组中最大的两个值的索引
    # # _, max_indices2 = torch.topk(groups2, k=4, dim=1)

    # # # 将组内索引转换为原始 tensor 的索引
    # # original_indices2 = max_indices2 + (torch.arange(groups2.size(0), device=act_scales.device).view(-1, 1) * 128)

    # # original_indices_flat2 = original_indices2.flatten()
    
    # # top_values1, top_indices1 = torch.topk(weight_scales, k=20)
    # # top_values2, top_indices2 = torch.topk(act_scales, k=20)
    # # index1 = set(top_indices1)
    # # index2 = set(top_indices2)
    # index = set(index)
    # original_indices_flat1 = set(original_indices_flat1)
    # # original_indices_flat2 = set(original_indices_flat2)
    # # index = list(index1 | index2 | index)
    # index = list(original_indices_flat1 | index)
    
    # scales[index] *= a
    # # scales[index] = scales[index].mean()  / scales[index]
    # # 将不在 index 的元素设置为 1
    # # mask = torch.ones(scales.size(), device='cuda')
    # # mask[index] = 0  # 设置 index 位置为 0
    # # scales[mask.bool()] = 1  # 将 mask 为 True 的位置设置为 1


    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales
        
@torch.no_grad()
def smooth_ln_fcs_mamba_like(ln, fcs, act_scales, index, a ,alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, MixtralRMSNorm))
    assert isinstance(ln, (RMSNorm, RMSNormGated))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # scales = torch.full((4096,), 3).to(device=device, dtype=dtype)
    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )


    # groups = weight_scales.view(-1, 128)

    # # 找到每一组中最大的两个值的索引
    # _, max_indices = torch.topk(groups, k=4, dim=1)

    # # 将组内索引转换为原始 tensor 的索引
    # original_indices = max_indices + (torch.arange(groups.size(0), device=weight_scales.device).view(-1, 1) * 128)

    # original_indices_flat1 = original_indices.flatten()


    # index = set(index)
    # original_indices_flat1 = set(original_indices_flat1)
    # # original_indices_flat2 = set(original_indices_flat2)
    # # index = list(index1 | index2 | index)
    # index = list(original_indices_flat1 | index)
    
    scales[index] *= a
    
    # top_values1, top_indices1 = torch.topk(weight_scales, k=20)
    # top_values2, top_indices2 = torch.topk(act_scales, k=20)
    # index1 = set(top_indices1)
    # index2 = set(top_indices2)
    # index = set(index)
    # index = list(index1 | index2 | index)
    
    # scales[index] *= a
    
    # scales[index] *= a
    # mask = torch.ones(scales.size(), device='cuda')
    # mask[top_indices] = 0  # 设置 index 位置为 0
    # scales[mask.bool()] = 1  # 将 mask 为 True 的位置设置为 1
    # print(scales)
    # scales /= 2
    # print(scales)
    # scales[index] *= 3
    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, index, a, alpha=0.5):
    scale = {}
# def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        
            if isinstance(module, OPTDecoderLayer):
                attn_ln = module.self_attn_layer_norm
                qkv = [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ]
                qkv_input_scales = scales[name + ".self_attn.q_proj"]
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

                ffn_ln = module.final_layer_norm
                fc1 = module.fc1
                fc1_input_scales = scales[name + ".fc1"]
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            elif isinstance(module, BloomBlock):
                attn_ln = module.input_layernorm
                qkv = module.self_attention.query_key_value
                qkv_input_scales = scales[name + ".self_attention.query_key_value"]
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

                ffn_ln = module.post_attention_layernorm
                fc1 = module.mlp.dense_h_to_4h
                fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            # elif isinstance(module, FalconDecoderLayer):
            #     qkv = module.self_attention.query_key_value
            #     qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            #     fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            #     fc1 = module.mlp.dense_h_to_4h

            #     if (
            #         not module.config.new_decoder_architecture
            #         and module.config.parallel_attn
            #     ):
            #         attn_ln = module.input_layernorm
            #         smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
            #     else:
            #         attn_ln = (
            #             module.ln_attn
            #             if module.config.new_decoder_architecture
            #             else module.input_layernorm
            #         )
            #         ffn_ln = (
            #             module.ln_mlp
            #             if module.config.new_decoder_architecture
            #             else module.post_attention_layernorm
            #         )
            #         smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            #         smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
                
                if int(name.split('.')[-1]) < 50:
                    #     a = 2.4
                    # if int(name.split('.')[-1]) < 5:
                    #     a = 2.2
                    # else:
                    #     a = 1.2
                       
                    alpha = 0.89
                    a_all = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse_alpha.pt")

                    a = a_all[int(name.split('.')[-1])]

                    ratio1 = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse.pt")

                    alpha = ratio1[int(name.split('.')[-1])]

                    # ratio2 = torch.load("/home/zijian/projects/BiLLM/ratio_search_mse_ratio2.pt")

                    # alpha2 = ratio2[int(name.split('.')[-1])]
                # if int(name.split('.')[-1]) == 11:
                    attn_ln = module.input_layernorm  # attention forward norm
                    qkv = [
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ]

                    # print(name)
                    q_index = set(index[name.split('.')[-1] + ".self_attn.q_proj"])
                    # print(q_index)
                    k_index = set(index[name.split('.')[-1] + ".self_attn.k_proj"])
                    v_index = set(index[name.split('.')[-1] + ".self_attn.v_proj"])
                    intersect = list(q_index&k_index&v_index)
                    union = list(q_index | k_index | v_index )
                    qkv_input_scales = scales[name + ".self_attn.q_proj"]
                    scale1 = smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, union,a,alpha)
                    index_num1 = name.split('.')[-1]
                    key1 = f"{index_num1}.q_proj"
                    scale[key1] = scale1

                    ffn_ln = module.post_attention_layernorm  # feed forward norm
                    fcs = [module.mlp.gate_proj, module.mlp.up_proj]
                    # fcs = [ module.mlp.up_proj]
                    
                    gate_index = set(index[name.split('.')[-1] + ".mlp.gate_proj"])
                    up_index = set(index[name.split('.')[-1] + ".mlp.up_proj"])
                    intersect = list(gate_index&up_index)
                    union = list(gate_index | up_index)
                    
                    
                    fcs_input_scales = scales[name + ".mlp.gate_proj"]

                    scale2 = smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, union,a, alpha)
                    
                    
                else:    
                    alpha = 0.89       
                    attn_ln = module.input_layernorm  # attention forward norm
                    qkv = [
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ]

                    qkv_input_scales = scales[name + ".self_attn.q_proj"]
                    smooth_ln_fcs_llama_like_new(attn_ln, qkv, qkv_input_scales, alpha)

                    ffn_ln = module.post_attention_layernorm  # feed forward norm
                    fcs = [module.mlp.gate_proj, module.mlp.up_proj]
                    fcs_input_scales = scales[name + ".mlp.gate_proj"]

                    smooth_ln_fcs_llama_like_new(ffn_ln, fcs, fcs_input_scales, alpha)
                    
                    # index_num2 = name.split('.')[-1]
                    # key2 = f"{index_num2}.up_proj"
                    # scale[key2] = union
                    
                    # v_proj = module.self_attn.v_proj # feed forward norm
                    # o_proj = module.self_attn.o_proj # feed forward norm
                    # # fcs = [ module.mlp.up_proj]
            

                    # smooth_ln_fcs_llama_like_vo(v_proj, o_proj)
                # elif isinstance(module, MixtralDecoderLayer):
                #     attn_ln = module.input_layernorm  # attention forward norm
                #     qkv = [
                #         module.self_attn.q_proj,
                #         module.self_attn.k_proj,
                #         module.self_attn.v_proj,
                #     ]

                #     qkv_input_scales = scales[name + ".self_attn.q_proj"]
                #     smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

                #     ffn_ln = module.post_attention_layernorm  # feed forward norm
                #     fcs = [module.block_sparse_moe.gate]
                #     for expert in module.block_sparse_moe.experts:
                #         fcs.append(expert.w1)
                #         fcs.append(expert.w3)
                #     fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

                #     smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
            elif isinstance(module, Block):
                in_norm = module.norm
                in_proj = module.mixer.in_proj
                # print(name)

                input_scales = scales[name + ".mixer.in_proj"]
                input_scales = input_scales
                inproj_index = index[name[-1] + ".mixer.in_proj"]
                smooth_ln_fcs_mamba_like(in_norm, in_proj, input_scales,inproj_index,a, alpha)

                # out_norm = module.mixer.norm
                # out_proj = module.mixer.out_proj
                # fcs_input_scales = scales[name + ".mixer.out_proj"]
                # outproj_index = index[name[-1] + ".mixer.out_proj"]

                # smooth_ln_fcs_mamba_like(out_norm, out_proj, fcs_input_scales, outproj_index, a , alpha)

    return scale
    # torch.save(scale, "omni_channel.pt")
# @torch.no_grad()
def smooth_lm_new(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            # smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            smooth_ln_fcs_llama_like_train(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            # smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            smooth_ln_fcs_llama_like_train(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            
            # ratio = torch.load("/home/zijian/projects/BiLLM/ratio_search.pt")

            # alpha = ratio[int(name.split('.')[-1])]
            # print(alpha)
            # if int(name.split('.')[-1]) > 20:
                alpha = 0.75
                
                attn_ln = module.input_layernorm  # attention forward norm
                qkv = [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ]

                qkv_input_scales = scales[name + ".self_attn.q_proj"]
                # qkv_input_scales = scales[int(name[-1])]["qkv_smooth_scale"]
                smooth_ln_fcs_llama_like_new(attn_ln, qkv, qkv_input_scales, alpha)

                ffn_ln = module.post_attention_layernorm  # feed forward norm
                fcs = [module.mlp.gate_proj, module.mlp.up_proj]
                fcs_input_scales = scales[name + ".mlp.gate_proj"]
                # fcs_input_scales = scales[int(name[-1])]["fc1_smooth_scale"]

                smooth_ln_fcs_llama_like_new(ffn_ln, fcs, fcs_input_scales, alpha)

            # else:
            #     scales = torch.load("/home/zijian/projects/BiLLM/smooth_scale_new.pt")
            #     attn_ln = module.input_layernorm  # attention forward norm
            #     qkv = [
            #         module.self_attn.q_proj,
            #         module.self_attn.k_proj,
            #         module.self_attn.v_proj,
            #     ]

            
            #     # qkv_input_scales = scales[name + ".self_attn.q_proj"]
            #     qkv_input_scales = scales[name.split('.')[-1] + ".q_proj"]
            #     # print(qkv_input_scales)
            #     smooth_ln_fcs_llama_like_new(attn_ln, qkv, qkv_input_scales, alpha)

            #     ffn_ln = module.post_attention_layernorm  # feed forward norm
            #     fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            #     # fcs_input_scales = scales[name + ".mlp.gate_proj"]
            #     fcs_input_scales = scales[name.split('.')[-1] + ".up_proj"]

            #     smooth_ln_fcs_llama_like_new(ffn_ln, fcs, fcs_input_scales, alpha)


# Step 3: 定义损失函数
def proximity_loss(weights_scaled):
    """
    对矩阵的每一行计算 (b + a 或 -b + a) 的偏离损失，然后取所有行的平均。
    """
    # 动态计算每一行的 a 和 b
    a = torch.mean(weights_scaled, dim=1, keepdim=True)  # 每行的均值 a (shape: [num_rows, 1])
    b = torch.mean(torch.abs(weights_scaled - a), dim=1, keepdim=True)  # 每行的绝对差均值 b (shape: [num_rows, 1])

    # 目标值
    target1 = b + a  # 每行的目标值 b + a (shape: [num_rows, 1])
    target2 = -b + a  # 每行的目标值 -b + a (shape: [num_rows, 1])

    # 对每个元素计算与目标值的最小距离
    # loss = 1000 * torch.min(torch.abs(weights_scaled - target1), torch.abs(weights_scaled - target2))
    loss = torch.where(
    torch.abs(weights_scaled - target1) < torch.abs(weights_scaled - target2),
    torch.abs(weights_scaled - target1) / (torch.abs(target1)  ),
    torch.abs(weights_scaled - target2) / (torch.abs(target2)  )
)

    # # 返回总损失（所有行的损失均值），以及每行的 a 和 b
    targetabs = 100 * torch.mean(torch.abs(target1))+torch.mean(torch.abs(target2))
    # targetabs = torch.mean(torch.abs(target1))
    
    loss_def = torch.mean(torch.min(torch.abs(weights_scaled - target1), torch.abs(weights_scaled - target2)))
    return 100 *torch.mean(loss), loss_def
    # return loss_def, loss_def

def search_top_k(fcs, act_scales, alpha):
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    # scales = (
    #     (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
    #     .clamp(min=1e-5)
    #     .to(device)
    #     .to(dtype)
    # )
    
    # groups = weight_scales.view(-1, 128)
    
    loss_list = []

    # for i in range(1,51):
    for a in  range(60,96):
        alpha = a/100
        
        scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
        
        
        
        loss_fcs = []
        # 找到每一组中最大的两个值的索引
        # _, max_indices = torch.topk(groups, k=i, dim=1)

        # # 将组内索引转换为原始 tensor 的索引
        # original_indices = max_indices + (torch.arange(groups.size(0), device=weight_scales.device).view(-1, 1) * 128)

        # original_indices_flat = original_indices.flatten()


        # # index = set(index)
        # # original_indices_flat1 = set(original_indices_flat1)
        # # # original_indices_flat2 = set(original_indices_flat2)
        # # # index = list(index1 | index2 | index)
        # # index = list(original_indices_flat1)
        
        # scales[original_indices_flat] *= 2
        
        loss = 0
        for fc in fcs:
            # loss = 0
            weights_scaled = fc.weight.mul(scales.view(1, -1)) 
            blocksize = 128
            columns = fc.weight.shape[1]
            for blocki, col_st in enumerate(range(0, columns, blocksize)):
                col_ed = min(col_st + blocksize, columns)
                n_cols = col_ed - col_st

                st = col_st
                ed = col_ed
                weight_block = weights_scaled[:, st:ed]
                loss1, loss2 = proximity_loss(weight_block)
                
                loss += loss2
            del weights_scaled
            del weight_block
            # loss_fcs.append(loss.detach())
        loss_list.append(loss.detach())
    print(1)
    
def smooth_ln_fcs_llama_like_train(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    final_scales_list = []
    # search_top_k(fcs, act_scales, alpha)
    
    for fc in fcs:
        
        weight_scales = fc.weight.abs().max(dim=0, keepdim=True)[0].to(torch.float32)
        
        # act_scales = torch.ones_like(act_scales)

        # weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
        scales = (
            (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(torch.float32)
        )
        # scales = scales.view(-1)
        
        # weight_scales = weight_scales.view(-1)
        # groups = weight_scales.view(-1, 128)

        # # 找到每一组中最大的两个值的索引
        # _, max_indices = torch.topk(groups, k=8, dim=1)

        # # 将组内索引转换为原始 tensor 的索引
        # original_indices = max_indices + (torch.arange(groups.size(0), device=weight_scales.device).view(-1, 1) * 128)

        # original_indices_flat1 = original_indices.flatten()


        # # index = set(index)
        # original_indices_flat1 = set(original_indices_flat1)
        # # original_indices_flat2 = set(original_indices_flat2)
        # # index = list(index1 | index2 | index)
        # index = list(original_indices_flat1)
        
        # scales[index] *= 2
        # scales = scales.view(1,-1)
        # scales = torch.ones_like(scales) *2

        # weight = weight[:, :128]
        # scales = scales[:, :128]
        # scales = torch.tensor(scales)
        # scales = scales.clone().detach().requires_grad_(True)
        # print(scales)
        scales_new = torch.zeros_like(scales) 

        ##
        ##lr = 0.001 epoch=2000
        # 定义优化器
        # optimizer = torch.optim.Adam([scales], lr=0.005)  # 使用 Adam 优化器，学习率为 0.1

        # 训练次数
        num_epochs = 500

        blocksize = 128
        columns = fc.weight.shape[1]
        for blocki, col_st in enumerate(range(0, columns, blocksize)):
            col_ed = min(col_st + blocksize, columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            
            weight_block = fc.weight[:, st:ed]
            # l1_norm_origin = torch.sum(torch.abs(weight_block)).detach()
            scales_block = scales[:, st:ed]
            
            best_loss = float('inf')
            
            target = [float('inf')]
            loss_list = []
            sum_all = []
            
            # a = torch.mean(weight_block * scales_block, dim=1, keepdim=True).detach()  # 每行的均值 a (shape: [num_rows, 1])
            # b = torch.mean(torch.abs(weight_block * scales_block - a), dim=1, keepdim=True).detach() # 每行的绝对差均值 b (shape: [num_rows, 1])

            # # 目标值
            # target1 = b + a  # 每行的目标值 b + a (shape: [num_rows, 1])
            # target2 = -b + a  # 每行的目标值 -b + a (shape: [num_rows, 1])
            
            scales_block = torch.tensor(scales_block, requires_grad=True)  # 确保 scales_block 可训练
            # original_sum = torch.sum(torch.abs(scales_block)).detach()
            
            # 定义优化器
            optimizer = torch.optim.SGD([scales_block], lr=0.01)
                # 动态计算每一行的 a 和 b

            # 开始训练
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # 清零梯度
                
                
                scaled_matrix = weight_block * scales_block
                # sum = torch.sum(torch.abs(scales_block))
                # sum_all.append(sum.detach())
                # scaled_matrix = weight_block 
                # regularization_loss = 0.1 * torch.sum(scales_block ** 2) 
                
                loss, targetabs = proximity_loss(scaled_matrix)
                # target.append(targetabs.detach())
                if targetabs > target[-1]:
                # if torch.logical_and(targetabs < target[-1], abs(targetabs - target[-1]) > 1e-3):
                # if targetabs < target[-1] & abs(targetabs - target[-1]) > 1e-3:
                    print("finished!")
                    break
                else:
                    target.append(targetabs.detach())
                # loss_list.append(loss)
                
                
                    # 计算 scales_block 的约束损失
                # new_sum = torch.sum(torch.abs(scales_block))  # 新的 scales_block 的绝对值和
                # constraint_loss = (new_sum - original_sum) ** 2  # 差的平方
                
                # loss = loss + constraint_loss

                # # 计算矩阵乘以 scale 后的结果
                # scaled_matrix = torch.abs(weight_block) * scales_block

                # row_variances = torch.var(scaled_matrix, dim=1)

                # # 计算所有行方差的总和（目标函数）
                # loss = torch.sum(row_variances)

                # 反向传播并更新参数
                loss.backward()
                optimizer.step()

                # 打印训练过程
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
                
                if loss < best_loss:   
                    best_loss = loss
                    scales_new[:, st:ed] = scales_block.detach()
                    
            # target = [target1.cpu().detach().numpy() for target1 in target]
            # loss_list = [loss.cpu().detach().numpy() for loss in loss_list]
            # # plt.plot(loss_list, marker='o', linestyle='-', color='b', label='loss')

            # # 绘制第二个列表
            # plt.plot(target,  linestyle='--', color='r', label='target')

            # # 添加标题和标签
            # plt.title("Line Chart with Two Lists")
            # plt.xlabel("Index")
            # plt.ylabel("Value")

            # # 显示图例
            # plt.legend()
            # plt.savefig("loss_target.png", dpi=300)
        final_scales_list.append(scales_new)
    # 将所有 scales_new 拼接成一个形状为 (3, columns) 的张量
    final_scales_matrix = torch.cat(final_scales_list, dim=0)

    # 对每一列选择绝对值最大的值，并保留原始符号
    _, indices = torch.max(torch.abs(final_scales_matrix), dim=0)  # 获取绝对值最大的索引
    final_scales = final_scales_matrix[indices, torch.arange(final_scales_matrix.size(1))]
    scales = final_scales.to(dtype)


    # # ln.weight.div_(scales.detach())
    # ln.weight = torch.nn.Parameter(ln.weight.div(scales))
    # for fc in fcs:
    #     # fc.weight.mul_(scales.detach().view(1, -1)) 
    #     fc.weight = torch.nn.Parameter(fc.weight.mul(scales.view(1, -1)) )  
    #     print("done")
        
    # print("!!!!!!!!!!!!!") 
    
    ln.weight = torch.nn.Parameter(ln.weight.div(scales))
    ln.bias = torch.nn.Parameter(ln.bias.div(scales))

    for fc in fcs:
        fc.weight = torch.nn.Parameter(fc.weight.mul(scales.view(1, -1)) )   
        print("done")  
            
@torch.no_grad()
def smooth_ln_fcs_llama_like_new(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    


    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    
    # groups = weight_scales.view(-1, 128)

    # # 找到每一组中最大的两个值的索引
    # _, max_indices = torch.topk(groups, k=8, dim=1)

    # # 将组内索引转换为原始 tensor 的索引
    # original_indices = max_indices + (torch.arange(groups.size(0), device=weight_scales.device).view(-1, 1) * 128)

    # original_indices_flat1 = original_indices.flatten()


    # # index = set(index)
    # original_indices_flat1 = set(original_indices_flat1)
    # # original_indices_flat2 = set(original_indices_flat2)
    # # index = list(index1 | index2 | index)
    # index = list(original_indices_flat1)
    
    # scales[index] *= 2
    # scales *= 2
    
    # selected_columns = set()
    # for fc in fcs:
    # # 1. 计算每列的最小值
    #     col_min_values, col_min_indices = fc.weight.abs().min(dim=0)
        
    #     # 2. 取前 n 小的列索引
    #     _, smallest_indices = torch.topk(col_min_values, k=512, largest=False)

        
    #     # 3. 将这些列索引加入到并集中
    #     selected_columns.update(smallest_indices.tolist())

    # # 将选出的列索引转为一个排序的列表
    # selected_columns = sorted(selected_columns)
    # # scales = act_scales.to(device=device, dtype=dtype)
    # # scales = torch.ones_like(scales) * 2
    
    # scales[selected_columns] *= 2.2

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
        print(1)
        

@torch.no_grad()
def smooth_ln_fcs_llama_like_vo(v, o):

    device, dtype = v.weight.device, v.weight.dtype
    # act_scales = act_scales.to(device=device, dtype=dtype)
    # weight_scales = torch.cat(
    #     [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    # )
    # weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    # scales = (
    #     (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
    #     .clamp(min=1e-5)
    #     .to(device)
    #     .to(dtype)
    # )
    v_scales = torch.cat(
        [v.weight.abs().max(dim=0, keepdim=True)[0]], dim=0
    )
    v_scales = v_scales.max(dim=0)[0].clamp(min=1e-5)
    
    
    o_scales = torch.cat(
        [o.weight.abs().max(dim=0, keepdim=True)[0]], dim=0
    )
    o_scales = o_scales.max(dim=0)[0].clamp(min=1e-5)
    
    scales = torch.ones_like(o_scales).to(device=device, dtype=dtype)
    
    # top_values1, top_indices1 = torch.topk(v_scales, k=20)
    top_values2, top_indices2 = torch.topk(o_scales, k=20)
    # index1 = set(top_indices1)
    # index2 = set(top_indices2)
    # index = list(index2 )
    index = top_indices2
    
    scales[index] = 2

    v.weight.div_(scales.view(-1,1))
    o.weight.mul_(scales.view(1, -1))


# @torch.no_grad()
def smooth_lm_layer(layer, scales,index, i, a,  alpha=0.5):
    # a = 22
    # a = a/10
    # # a = 1.8
    # alpha = 0.85
    
    ratio = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse.pt")

    alpha = ratio[i]

    # ratio1 = torch.load("/home/zijian/projects/BiLLM/ratio_search_mse_ratio1.pt")

    # alpha1 = ratio1[i]

    # ratio2 = torch.load("/home/zijian/projects/BiLLM/ratio_search_mse_ratio1.pt")

    # alpha2 = ratio2[i]

    a_all = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse_alpha.pt")
    a = a_all[i]
    # a = 2

            
    attn_ln = layer.input_layernorm  # attention forward norm
    qkv = [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ]
    
    q_index = set(index[str(i)  + ".self_attn.q_proj"])
    # print(q_index)
    k_index = set(index[str(i)  + ".self_attn.k_proj"])
    v_index = set(index[str(i)  + ".self_attn.v_proj"])
    # intersect = list(q_index&k_index&v_index)
    union = list(q_index | k_index | v_index )

    # qkv_input_scales = scales[name + ".self_attn.q_proj"]
    qkv_input_scales = scales["model.layers."+ str(i) + ".self_attn.q_proj"]
    scale1 = smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales,union, a,  alpha)

    ffn_ln = layer.post_attention_layernorm  # feed forward norm
    fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
    # fcs_input_scales = scales[name + ".mlp.gate_proj"]
    fcs_input_scales = scales["model.layers."+ str(i) + ".mlp.gate_proj"]
    
    gate_index = set(index[str(i) + ".mlp.gate_proj"])
    up_index = set(index[str(i) + ".mlp.up_proj"])
    # intersect = list(gate_index&up_index)
    union = list(gate_index | up_index)

    scale2 = smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, union, a, alpha)

    return scale1

def smooth_mamba_layer(layer, scales,index, i, a,  alpha=0.5):
    # a = 22
    # a = a/10
    
    # ratio = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse.pt")

    # alpha = ratio[i]

    # a_all = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse_alpha.pt")
    # a = a_all[i]

            
    in_norm = layer.norm
    in_proj = layer.mixer.in_proj
    # print(name)

    input_scales = scales["backbone.layers."+ str(i) +  ".mixer.in_proj"]
    input_scales = input_scales
    inproj_index = index[str(i) + ".mixer.in_proj"]
    smooth_ln_fcs_mamba_like(in_norm, in_proj, input_scales,inproj_index,a, alpha)

# @torch.no_grad()
def smooth_lm_layer_two_ratio(layer, scales,index, i, a,  alpha1, alpha2):
    # a = 22
    # a = a/10
    a = 2
    # alpha = 0.85
    
    # ratio = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse.pt")

    # alpha = ratio[i]

    # a_all = torch.load("/home/zijian/projects/BiLLM/ratio_search_new_mse_alpha.pt")
    # a = a_all[i]

            
    attn_ln = layer.input_layernorm  # attention forward norm
    qkv = [
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ]
    
    q_index = set(index[str(i)  + ".self_attn.q_proj"])
    # print(q_index)
    k_index = set(index[str(i)  + ".self_attn.k_proj"])
    v_index = set(index[str(i)  + ".self_attn.v_proj"])
    # intersect = list(q_index&k_index&v_index)
    union = list(q_index | k_index | v_index )

    # qkv_input_scales = scales[name + ".self_attn.q_proj"]
    qkv_input_scales = scales["model.layers."+ str(i) + ".self_attn.q_proj"]
    scale1 = smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales,union, a,  alpha1)

    ffn_ln = layer.post_attention_layernorm  # feed forward norm
    fcs = [layer.mlp.gate_proj, layer.mlp.up_proj]
    # fcs_input_scales = scales[name + ".mlp.gate_proj"]
    fcs_input_scales = scales["model.layers."+ str(i) + ".mlp.gate_proj"]
    
    gate_index = set(index[str(i) + ".mlp.gate_proj"])
    up_index = set(index[str(i) + ".mlp.up_proj"])
    # intersect = list(gate_index&up_index)
    union = list(gate_index | up_index)

    scale2 = smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, union, a, alpha2)

    return scale1