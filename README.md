# DBellQuant


## DBellQuant: Breaking the Bell with Double-Bell Transformation for LLM Post Training Binarization

[arXiv](https://arxiv.org/abs/2507.01027)

DBellQuant Framework Overview

![DBellQuant Framework](imgs/framework.png)


Authors: Zijian Ye*, Wei Huang*, Yifei Yu, Tianhe Ren, Zhongrui Wang, Xiaojuan Qi



This repository contains the implementation of the paper:

## Dependencies

```
pip install -r requirements.txt
```

The act_scales pt files can be downloaded in [modelscope](https://modelscope.cn/models/zijian0007/DBellQuant_act_scales).

## Runing

DBellQuant for Llama-2 Families

```
python3 run_smooth.py meta-llama/Llama-2-7b-hf c4 braq --blocksize 128 --salient_metric hessian
```

DBellQuant for Llama-1 Families

```
python3 run_smooth.py huggyllama/llama-7b c4 braq --blocksize 128 --salient_metric hessian
```

DBellQuant for OPT Families

```
python3 run_smooth.py facebook/opt-6.7b c4 braq --blocksize 128 --salient_metric hessian
```

DBellQuant for Qwen Families

```
python3 run_qwen.py Qwen/Qwen2.5-7B c4 braq --blocksize 128 --salient_metric hessian
```
Attention: Qwen requires newer transformers version.

## Results

QA results:

In terms of average accuracy on QA datasets, DBellQuant significantly surpasses previous methods and increase the average accuracy up to 42.48%

![DBellQuant](imgs/qa_result.png)

Perplexity results

The results demonstrate that DBellQuant significantly
outperforms the state-of-the-art ARB-LLMX when only quantizing weights, achieving up to a 42.18%
reduction in perplexity. Moreover, when activations are quantized to lower bit-widths like 6-bit,
DBellQuant achieves up to a 76.66% reduction in perplexity for the LLaMA family compared
to BiLLM. 

![DBellQuant](imgs/perplexity.png)

## Related Work

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://github.com/IST-DASLab/gptq)

[BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://github.com/Aaronhuang-778/BiLLM)

[ARB-LLM: Alternating Refined Binarizations for Large Language Models](https://github.com/ZHITENGLI/ARB-LLM)

## Citation

If you find DBellQuant is useful and helpful to your work, please kindly cite this paper:

```
@article{ye2025dbellquant,
  title={DBellQuant: Breaking the Bell with Double-Bell Transformation for LLMs Post Training Binarization},
  author={Ye, Zijian and Huang, Wei and Yu, Yifei and Ren, Tianhe and Wang, Zhongrui and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2507.01027},
  year={2025}
}
```
