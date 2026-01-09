import time

import torch
import torch.nn as nn
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None



@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")
    print("111")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    print(nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache
    
    return ppl

@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    print({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache
    
    

@torch.no_grad()
def mamba_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // 2048

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.backbone.layers

    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.backbone.embeddings = model.backbone.embeddings.to(dev)
    # model.backbone.embedding = model.backbone.embedding.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # class Catcher(nn.Module):
    #     def __init__(self, module):
    #         super().__init__()
    #         self.module = module

    #     def forward(self, inp, **kwargs):
    #         inps[cache["i"]] = inp
    #         cache["i"] += 1
    #         # cache["attention_mask"] = kwargs["attention_mask"]
    #         raise ValueError

    # layers[0] = Catcher(layers[0])
    # for i in range(nsamples):
    #     batch = testenc[:, (i * 2048) : ((i + 1) * 2048)].to(dev)
    #     try:
    #         model(batch)
    #     except ValueError:
    #         pass
    for i in range(nsamples):
        batch = testenc[:, (i * 2048) : ((i + 1) * 2048)].to(dev)
        inps[cache["i"]] = model.backbone.embeddings(batch[0].to(dev))
        cache["i"] += 1
    # layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.backbone.embeddings = model.backbone.embeddings.cpu()
    # model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs[j] = layer(inps[j].unsqueeze(0))[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.backbone.norm_f is not None:
        # model.model.norm = model.model.norm.to(dev)
        model.backbone.norm_f = model.backbone.norm_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.backbone.norm_f is not None:
            hidden_states = model.backbone.norm_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache



@torch.no_grad()
def mamba2_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // 2048

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.backbone.layers

    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.backbone.embeddings = model.backbone.embeddings.to(dev)
    model.backbone.embedding = model.backbone.embedding.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, 2048, 2560), dtype=dtype, device=dev
    )
    # inps = torch.zeros(
    #     (nsamples, 2048, 2048), dtype=dtype, device=dev
    # )
    cache = {"i": 0}

    # class Catcher(nn.Module):
    #     def __init__(self, module):
    #         super().__init__()
    #         self.module = module

    #     def forward(self, inp, **kwargs):
    #         inps[cache["i"]] = inp
    #         cache["i"] += 1
    #         # cache["attention_mask"] = kwargs["attention_mask"]
    #         raise ValueError

    # layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * 2048) : ((i + 1) * 2048)].to(dev)
        inps[cache["i"]] = model.backbone.embedding(batch[0].to(dev))
        cache["i"] += 1
    # layers[0] = layers[0].module


    layers[0] = layers[0].cpu()
    # model.backbone.embeddings = model.backbone.embeddings.cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    # attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        if i == 0:
            rs = [None] * nsamples

        for j in range(nsamples):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            outs[j], rs[j]  = layer(inps[j].unsqueeze(0), rs[j])
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

        

    if model.backbone.norm_f is not None:
        # model.model.norm = model.model.norm.to(dev)
        model.backbone.norm_f = model.backbone.norm_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.backbone.norm_f is not None:
            hidden_states = layer_norm_fn(
                hidden_states,
                model.backbone.norm_f.weight,
                model.backbone.norm_f.bias,
                eps=model.backbone.norm_f.eps,
                residual=rs[i],
                prenorm=False,
                residual_in_fp32=True,
                is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm)
            )
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache
    
    return ppl


@torch.no_grad()
def mamba2_eval_new(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // 2048

    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    layers = model.backbone.layers

    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    # model.backbone.embeddings = model.backbone.embeddings.to(dev)
    model.backbone.embedding = model.backbone.embedding.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = model.backbone.embedding(testenc.to(dev))
    outs = torch.zeros_like(inps)


    # model.backbone.embeddings = model.backbone.embeddings.cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()

    # attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        if i == 0:
            rs = None

        outs, rs  = layer(inps, rs)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.backbone.norm_f is not None:
        # model.model.norm = model.model.norm.to(dev)
        model.backbone.norm_f = model.backbone.norm_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    # nlls = []
    # for i in range(nsamples):
    #     hidden_states = inps[i].unsqueeze(0)
    #     if model.backbone.norm_f is not None:
    #         hidden_states = model.backbone.norm_f(hidden_states)
    #     lm_logits = model.lm_head(hidden_states)
    #     shift_logits = lm_logits[:, :-1, :].contiguous()
    #     shift_labels = testenc[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(
    #         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    #     )
    #     neg_log_likelihood = loss.float() * 2048
    #     nlls.append(neg_log_likelihood)
    # ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
    # print(f"Perplexity: {ppl.item():3f}")

    # model.config.use_cache = use_cache
    
    hidden_states = inps
    if model.backbone.norm_f is not None:
        hidden_states = model.backbone.norm_f(hidden_states)
    lm_logits = model.lm_head(hidden_states)
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = testenc[:, 1:]
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    neg_log_likelihood = loss.float() * testenc.size(1)
    ppl = torch.exp(neg_log_likelihood / testenc.size(1))
    print(f"Perplexity: {ppl.item():3f}")
        

    
@torch.no_grad()
def llama_eval_train(model, dataloader,testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")
    print("111")
    
    trainenc = torch.cat([batch[0] for batch in dataloader], dim=1)

    # testenc = testenc.input_ids
    testenc = trainenc
    nsamples = testenc.numel() // model.seqlen
    print(nsamples)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache
    
    return ppl