import math
import time
from exceptiongroup import catch
import torch
import torch.nn as nn
import transformers
from utils.structure import structural_guassian_distribution
from utils.structure import structural_guassian_distribution_new
from utils.structure import structural_guassian_distribution_weight
from utils.structure import structural_guassian_distribution_weight_2
from smoothquant.fake_quant import W8A8Linear

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

'''
BRAGPTQ is the meaning of GPTQ used Binary Residual Approximation in paper to realize 1-bit quantization
BRAGPTQ uses structural mask to distinguish outliers and other data, and takes advantage of part of GPTQ to lower error
'''

class BRAGPTQ:
    def __init__(
        self, layer,braq_quantizer, salient_metric, disable_gptq=False
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        )or isinstance(
            self.layer, W8A8Linear
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01, 
                    partition=3, 
                    orders=(1,1,2),
                    ):
        

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(partition, dim=0)
            mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
            # mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 30)
            mask[0] = mask1
            mask[1] = mask2
            mask[2] = mask3

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                
                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                q_part_groups = []

                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i]))

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2
                    # breakpoint()

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        del mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
        
class BRAGPTQloss:
    def __init__(
        self, layer, braq_quantizer,scale,name,origin_weight,salient_metric, disable_gptq=False
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq
        self.scale = scale
        self.name = name
        self.origin_weight = origin_weight

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        )or isinstance(
            self.layer, W8A8Linear
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01, 
                    partition=3, 
                    orders=(1,1,1),
                    ):
        name = self.layer
        scale = self.scale
        # scale = scale["0.q_proj"]
        scale = scale.view(1, -1)
        W2 = self.origin_weight
        W = self.layer.weight.data.clone()
        W_origin = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        index = []

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(partition, dim=0)
            # mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
            mask1, mask2 = structural_guassian_distribution_weight_2(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 30)
            mask[0] = mask1
            mask[1] = mask2
            # mask[2] = mask3
            
            true_indices = torch.nonzero(mask3[0], as_tuple=True)[0]
            true_indices += col_st

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                
                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                q_part_groups = []

                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i]))

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q
                    # Losses1[:, i] = (w - q) ** 2 / d**2
                    Losses1[:, i] = (w - q) ** 2 
                    # print(d)
                    # breakpoint()

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))
            index.append(true_indices)
        index_all = torch.cat(index)
        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())
        
        loss = (W2 - W) ** 2
        loss = torch.sum(loss)
        # print(loss)

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        del mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        # return {"error": torch.sum(Losses).item()}
        # return {"error": loss.item()}, index_all
        return index_all

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
        
        


class BRAGPTQnew:
    def __init__(
        self, layer, braq_quantizer,salient_metric, disable_gptq=False
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01, 
                    partition=3, 
                    orders=(1,1,2),
                    ):
        W = self.layer.weight.data.clone()
        blocksize = 64
        W1 = W[:1280, :]
        W2 = W[1280:2560, :]
        W3 = W[2560:3840, :]
        W4 = W[3840:5120, :]
        W5 = W[5120:, :]
        
        # W1 = W[:2560, :]    
        # W2 = W[2560:5120, :] 
        # W3 = W[5120:7680, :]  
        # W4 = W[7680:, :] 
        input = [ W1,W2, W3, W4, W5]
        for a in range(5):
            W = input[a]
        
            if isinstance(self.layer, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(self.layer, transformers.Conv1D):
                W = W.t()
            W = W.float()
            tick = time.time()

            H = self.H
            # del self.H
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            Losses = torch.zeros(5120, device=self.dev)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H
            if a != 10:
                for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
                    col_ed = min(col_st + blocksize, self.columns)
                    n_cols = col_ed - col_st

                    st = col_st
                    ed = col_ed
                    mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(partition, dim=0)
                    # mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
                    mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
                    mask[0] = mask1
                    mask[1] = mask2
                    mask[2] = mask3

                    assert self.braq_quantizer.groupsize % blocksize == 0

                    if self.disable_gptq:
                        # RTN
                        # print("RTN")
                        w = W[:, col_st:col_ed]
                        
                        # from low to high group
                        q_part_groups = []
                        for i in range(mask.shape[0]):
                            q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))

                        q = torch.zeros_like(w)
                        for j in range(mask.shape[0]):
                            q += q_part_groups[j][:] * mask[j, :]
                        W[:, col_st:col_ed] = q
                    else:
                        # shape of W1: [oc, n_cols]
                        W1 = W[:, col_st:col_ed].clone()
                        Q1 = torch.zeros_like(W1)
                        Err1 = torch.zeros_like(W1)
                        Losses1 = torch.zeros_like(W1)
                        Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                        q_part_groups = []

                        for i in range(mask.shape[0]):
                            q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i]))

                        for i in range(n_cols):
                            # shape of w: [oc, 1]
                            w = W1[:, i]
                            d = Hinv1[i, i]

                            q = torch.zeros_like(w)
                            for j in range(mask.shape[0]):
                                q += q_part_groups[j][:, i] * mask[j, :, i]

                            Q1[:, i] = q
                            # Losses1[:, i] = (w - q) ** 2 / d**2
                            Losses1[:, i] = (w - q) ** 2 / d**2
                            # breakpoint()

                            err1 = (w - q) / d
                            Err1[:, i] = err1

                        W[:, col_st:col_ed] = Q1
                        # Losses += torch.sum(Losses1, 1) / 2

                        W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                        if DEBUG:
                            if i == 0 :
                                self.layer.weight[:5120, :].data[:, :col_ed] = W[:, :col_ed]
                                self.layer.weight[:5120, :].data[:, col_ed:] = W[:, col_ed:]
                                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                                print(torch.sum(Losses))
                            else :
                                self.layer.weight[5120:, :].data[:, :col_ed] = W[:, :col_ed]
                                self.layer.weight[5120:, :].data[:, col_ed:] = W[:, col_ed:]
                                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                                print(torch.sum(Losses))
            else:
                for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
                    col_ed = min(col_st + blocksize, self.columns)
                    n_cols = col_ed - col_st

                    st = col_st
                    ed = col_ed
                    mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(1, dim=0)
                    # mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
                    # mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 30)
                    mask1 = torch.ones_like(W[:, st:ed], dtype=torch.bool)
                    mask[0] = mask1
                    # mask[1] = mask2
                    # mask[2] = mask3

                    assert self.braq_quantizer.groupsize % blocksize == 0

                    if self.disable_gptq:
                        # RTN
                        # print("RTN")
                        w = W[:, col_st:col_ed]
                        
                        # from low to high group
                        q_part_groups = []
                        for i in range(mask.shape[0]):
                            q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=2))

                        q = torch.zeros_like(w)
                        for j in range(mask.shape[0]):
                            q += q_part_groups[j][:] * mask[j, :]
                        W[:, col_st:col_ed] = q
                    else:
                        # shape of W1: [oc, n_cols]
                        W1 = W[:, col_st:col_ed].clone()
                        Q1 = torch.zeros_like(W1)
                        Err1 = torch.zeros_like(W1)
                        Losses1 = torch.zeros_like(W1)
                        Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                        q_part_groups = []

                        for i in range(mask.shape[0]):
                            q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=2))

                        for i in range(n_cols):
                            # shape of w: [oc, 1]
                            w = W1[:, i]
                            d = Hinv1[i, i]

                            q = torch.zeros_like(w)
                            for j in range(mask.shape[0]):
                                q += q_part_groups[j][:, i] * mask[j, :, i]

                            Q1[:, i] = q
                            # Losses1[:, i] = (w - q) ** 2 / d**2
                            Losses1[:, i] = (w - q) ** 2 / d**2
                            # breakpoint()

                            err1 = (w - q) / d
                            Err1[:, i] = err1

                        W[:, col_st:col_ed] = Q1
                        # Losses += torch.sum(Losses1, 1) / 2

                        W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
                        
            

                        if DEBUG:
                            if i == 0 :
                                self.layer.weight[:5120, :].data[:, :col_ed] = W[:, :col_ed]
                                self.layer.weight[:5120, :].data[:, col_ed:] = W[:, col_ed:]
                                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                                print(torch.sum(Losses))
                            else :
                                self.layer.weight[5120:, :].data[:, :col_ed] = W[:, :col_ed]
                                self.layer.weight[5120:, :].data[:, col_ed:] = W[:, col_ed:]
                                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                                print(torch.sum(Losses))
                
            torch.cuda.synchronize()
            print("time %.2f" % (time.time() - tick))
            # print("error", torch.sum(Losses).item())

            if isinstance(self.layer, transformers.Conv1D):
                W = W.t()
            if a == 0:
                self.layer.weight.data[:1280, :] = W.reshape(self.layer.weight[:1280, :].shape).to(
                    self.layer.weight.data.dtype
                )
            elif a == 1:
                self.layer.weight.data[1280:2560, :]  = W.reshape(self.layer.weight[1280:2560, :].shape).to(
                    self.layer.weight.data.dtype
                )
            elif a == 2:
                self.layer.weight.data[2560:3840, :]  = W.reshape(self.layer.weight[2560:3840, :].shape).to(
                    self.layer.weight.data.dtype
                )
            elif a == 3:
                    self.layer.weight.data[3840:5120, :]  = W.reshape(self.layer.weight[3840:5120, :].shape).to(
                    self.layer.weight.data.dtype
                )
            # elif a == 2:
            #     self.layer.weight.data[5120:7680, :]  = W.reshape(self.layer.weight[5120:7680, :].shape).to(
            #         self.layer.weight.data.dtype
            #     )
            else:
                self.layer.weight.data[5120:, :] = W.reshape(self.layer.weight[5120:, :].shape).to(
                    self.layer.weight[5120:, :].data.dtype
                )
                
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        del mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        # return {"error": torch.sum(Losses).item()}
        return 1

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
        
        
        
        
class BRAGPTQTRAIN:
    def __init__(
        self, layer,weight,  braq_quantizer,salient_metric, disable_gptq=False
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = weight
        self.weight = weight
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        )or isinstance(
            self.layer, W8A8Linear
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01, 
                    partition=3, 
                    orders=(1,1,2),
                    ):
        # W = self.layer.weight.data.clone()
        W = self.weight
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

           # 修改 H 矩阵，确保计算图保留
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H = H.clone()  # 确保 H 是可修改的
        H[dead, dead] = 1
        W = W.clone()  # 克隆 W，避免修改原始 weight
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(partition, dim=0)
            # mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
            mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 30)
            mask[0] = mask1
            mask[1] = mask2
            mask[2] = mask3

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                
                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))
                # print(q_part_groups.requires_grad)
                # q = torch.zeros_like(w)
                # for j in range(mask.shape[0]):
                #     q += q_part_groups[j][:] * mask[j, :]
                # W[:, col_st:col_ed] = q
                # 替换直接赋值操作
                q = sum(q_part_groups[j] * mask[j] for j in range(mask.shape[0]))
                W = torch.cat([W[:, :col_st], q, W[:, col_ed:]], dim=1)  # 保留计算图
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                q_part_groups = []

                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i]))

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2
                    # breakpoint()

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                # W[:, col_st:col_ed] = Q1
                W = torch.cat([W[:, :col_st], Q1, W[:, col_ed:]], dim=1)  # 替换赋值
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                # if DEBUG:
                #     self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                #     self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                #     print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        # self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
        #     self.layer.weight.data.dtype
        # )
        output = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        del mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}, output

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
        
class BRAGPTQ64:
    def __init__(
        self, layer, braq_quantizer,salient_metric, disable_gptq=False
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        )or isinstance(
            self.layer, W8A8Linear
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self,
                    blocksize=64, 
                    percdamp=0.01, 
                    partition=3, 
                    orders=(1,1,2),
                    ):
        W = self.layer.weight.data.clone()
        W_origin = self.layer.weight.data.clone()
        blocksize = 64
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        index = []

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(partition, dim=0)
            # mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
            mask1, mask2, mask3 = structural_guassian_distribution(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 30)
            mask[0] = mask1
            mask[1] = mask2
            mask[2] = mask3
            
            true_indices = torch.nonzero(mask3[0], as_tuple=True)[0]
            true_indices += col_st

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                
                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                q_part_groups = []

                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i]))

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q
                    # Losses1[:, i] = (w - q) ** 2 / d**2
                    Losses1[:, i] = (w - q) ** 2 
                    # print(d)
                    # breakpoint()

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))
            index.append(true_indices)
        index_all = torch.cat(index)
        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())
        
        loss = (W_origin - W) ** 2
        loss = torch.sum(loss)
        # print(loss)

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        del mask1, mask2, mask3
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        # return {"error": torch.sum(Losses).item()}
        # return {"error": loss.item()}, index_all
        return index_all

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
        
        

class BRAGPTQ_newloss:
    def __init__(
        self, name,layer, braq_quantizer,salient_metric, disable_gptq=False
    ):
        self.name = name
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.braq_quantizer = braq_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq
        self.inp1 = []
        self.out1 = []

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        # self.inp1.append(inp)
        # self.out1.append(out)
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        )or isinstance(
            self.layer, W8A8Linear
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # self.inp1 = inp
        # self.out1 = out
        # breakpoint()

    def fasterquant(self,
                    blocksize=128, 
                    percdamp=0.01, 
                    partition=2, 
                    orders=(1,1,2),
                    ):
        
        # inp1 = self.inp1
        # name = self.name
        # out1 = self.out1

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            st = col_st
            ed = col_ed
            mask = torch.zeros_like(W[:, st:ed], dtype=torch.bool).unsqueeze(0).repeat_interleave(partition, dim=0)
            # mask1, mask2, mask3 = structural_guassian_distribution_new(W[:, st:ed], inp1,name, blocki, H[st:ed, st:ed], self.salient_metric, 50)
            mask1, mask2 = structural_guassian_distribution_weight_2(W[:, st:ed], H[st:ed, st:ed], self.salient_metric, 50)
            mask[0] = mask1
            mask[1] = mask2
            # mask[2] = mask3

            assert self.braq_quantizer.groupsize % blocksize == 0

            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                
                # from low to high group
                q_part_groups = []
                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(w, mask[i], order=orders[i]))

                q = torch.zeros_like(w)
                for j in range(mask.shape[0]):
                    q += q_part_groups[j][:] * mask[j, :]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                q_part_groups = []

                for i in range(mask.shape[0]):
                    q_part_groups.append(self.braq_quantizer.quantize(W1, mask[i], order=orders[i]))

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = torch.zeros_like(w)
                    for j in range(mask.shape[0]):
                        q += q_part_groups[j][:, i] * mask[j, :, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2
                    # breakpoint()

                    err1 = (w - q) / d
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        del mask
        del mask1, mask2
        if not self.disable_gptq:
            del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()