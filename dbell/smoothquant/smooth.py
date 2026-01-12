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
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
# from mamba_ssm.modules.block import Block
# from mamba_ssm.ops.triton.layer_norm import RMSNorm
# from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
# from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
# from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer, GemmaRMSNorm
# from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
import numpy as np
# import matplotlib.pyplot as plt
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
    print(1)


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

    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)


    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )
    


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


    
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    
    scales[index] *= a
    

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))




# @torch.no_grad()
def smooth_lm_new(model, scales, alpha, device, epochs):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            # module = module.to(device)
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
            # smooth_ln_fcs_llama_like_train(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
            # del module
            # smooth_ln_fcs_llama_like_train(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)

        # elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer, GemmaDecoderLayer)):
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
           
            
                alpha = 0.85
                
                attn_ln = module.input_layernorm  # attention forward norm
                qkv = [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ]

                qkv_input_scales = scales[name + ".self_attn.q_proj"]

                smooth_ln_fcs_llama_like_train(attn_ln, qkv, qkv_input_scales, epochs, alpha)

                ffn_ln = module.post_attention_layernorm  # feed forward nor
                fcs = [module.mlp.gate_proj, module.mlp.up_proj]
                fcs_input_scales = scales[name + ".mlp.gate_proj"]


                smooth_ln_fcs_llama_like_train(ffn_ln, fcs, fcs_input_scales, epochs,  alpha)




#  定义损失函数
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
    (torch.abs(weights_scaled - target1)) ** 2 / (torch.abs(target1)  ) ** 2,
    (torch.abs(weights_scaled - target2) ) ** 2/ (torch.abs(target2)  ) ** 2
)

    # # 返回总损失（所有行的损失均值），以及每行的 a 和 b
    targetabs = 100 * torch.mean(torch.abs(target1))+torch.mean(torch.abs(target2))
    # targetabs = torch.mean(torch.abs(target1))
    
    loss_def = torch.mean(torch.min(torch.abs(weights_scaled - target1), torch.abs(weights_scaled - target2)))
    return 100 *torch.mean(loss), loss_def
    # return loss_def, loss_def


    
def smooth_ln_fcs_llama_like_train(ln, fcs, act_scales, epochs, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    # assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm))
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    final_scales_list = []

    
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

        scales_new = torch.zeros_like(scales) 



        # 训练次数
        num_epochs = epochs

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
            

            
            scales_block = torch.tensor(scales_block, requires_grad=True)  # 确保 scales_block 可训练
            # original_sum = torch.sum(torch.abs(scales_block)).detach()
            
            # 定义优化器
            optimizer = torch.optim.SGD([scales_block], lr=0.01)
                # 动态计算每一行的 a 和 b

            # 开始训练
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # 清零梯度
                
                
                scaled_matrix = weight_block * scales_block

                
                loss, targetabs = proximity_loss(scaled_matrix)
                
                if targetabs > target[-1]:

                    print("finished!")
                    break
                else:
                    target.append(targetabs.detach())
                # target.append(targetabs.cpu().detach().numpy())


                # 反向传播并更新参数
                loss.backward()
                optimizer.step()

                # 打印训练过程
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
                
                if loss < best_loss:   
                    best_loss = loss
                    scales_new[:, st:ed] = scales_block.detach()
                    

        final_scales_list.append(scales_new)
    # 将所有 scales_new 拼接成一个形状为 (3, columns) 的张量
    final_scales_matrix = torch.cat(final_scales_list, dim=0)

    # 对每一列选择绝对值最大的值，并保留原始符号
    _, indices = torch.max(torch.abs(final_scales_matrix), dim=0)  # 获取绝对值最大的索引
    final_scales = final_scales_matrix[indices, torch.arange(final_scales_matrix.size(1))]
    scales = final_scales.to(dtype)


    # ln.weight.div_(scales.detach())
    ln.weight = torch.nn.Parameter(ln.weight.div(scales))
    for fc in fcs:
        # fc.weight.mul_(scales.detach().view(1, -1)) 
        fc.weight = torch.nn.Parameter(fc.weight.mul(scales.view(1, -1)) )  
        print("done")
        

            
@torch.no_grad()
def smooth_ln_fcs_llama_like_new(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, GemmaRMSNorm))
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
    

    scales = torch.clamp(scales, min=0.5,max=1.5)


    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
        # print(1)
        

@torch.no_grad()
def smooth_ln_fcs_llama_like_vo(v, o):

    device, dtype = v.weight.device, v.weight.dtype

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




def smooth_mamba_layer(layer, scales,index, i, a,  alpha=0.5):

            
    in_norm = layer.norm
    in_proj = layer.mixer.in_proj
    # print(name)

    input_scales = scales["backbone.layers."+ str(i) +  ".mixer.in_proj"]
    input_scales = input_scales
    inproj_index = index[str(i) + ".mixer.in_proj"]
    smooth_ln_fcs_mamba_like(in_norm, in_proj, input_scales,inproj_index,a, alpha)

