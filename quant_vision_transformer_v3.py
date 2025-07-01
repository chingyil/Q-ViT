import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from quant_vision_transformer import Q_PatchEmbed #, Q_Attention, Q_Mlp
from Quant import round_pass, ActQ, grad_scale, LinearQ
from _quan_base import Qmodes, _ActQ, _LinearQ
from timm.models.layers.weight_init import trunc_normal_
import numpy as np
from timm.models.layers.drop import DropPath
import math
import torch.nn.functional as F
from timm.models.layers.helpers import to_2tuple

def arr2s(x, nbit=16, start=2, prefix=""):
    return "\n".join([(prefix + hex(int(p) % (2 ** nbit))[start:]) for p in x.tolist()])

def arr2single(x, nbit=16, start=2, prefix=""):
    # return " ".join("%d" % xx for xx in x)
    if nbit == 1:
        b = "".join([format(int(xx), "01b") for xx in reversed(x)])
    elif nbit == 3:
        b = "".join([format(int(xx), "03b") for xx in reversed(x)])
        # bpad = "0" * ((4 - 1) - ((len(b) + 1) % 4)) + b
        assert len(b) % 4 == 0, print("b = ", b)
    elif nbit == 16:
        b = "".join([format(int(xx) % (2 ** 16), "016b") for xx in reversed(x)])
    elif nbit == 32:
        b = "".join([format(int(xx), "032b") for xx in reversed(x)])
        # import pdb; pdb.set_trace()
    # idxs = [i for i in range(0, len(b), 4)]
    tokens = [format(int(b[i:i+4], 2), "1x") for i in range(0, len(b), 4)]
    return "".join(tokens)

def arr2d2s(x, nbit=16, start=2, prefix=""):
    assert len(x.shape) == 2
    return "\n".join([arr2single(xx, nbit=nbit) for xx in x])
    # return "\n".join([(prefix + hex(int(p) % (2 ** nbit))[start:]) for p in x.tolist()])

class ActQ_v2(_ActQ):
    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
        super(ActQ_v2, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
        # print(self.alpha.shape, self.zero_point.shape)
    def forward(self, x0):
        assert self.alpha is not None
        # assert self.signed == 1
        # Qn = -2 ** (self.nbits - 1)
        # Qp = 2 ** (self.nbits - 1) - 1
        if self.signed == 1:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1


        # Method1:
        # zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        # # alpha = grad_scale(self.alpha, g)
        # # zero_point = grad_scale(zero_point, g)
        # # x0 = round_pass((x0 / alpha).clamp(Qn, Qp)) * alpha
        # if len(x0.shape)==2:
        #     zero_point = zero_point.unsqueeze(0)
        # elif len(x0.shape)==4:
        #     zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        alpha = self.alpha.mean()
        x1 = round_pass((x0 / alpha).clamp(Qn, Qp))
        return x1

class LinearQ_v2(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, step_in1=None, step_in2=None, step_out=None, **kwargs):
        super(LinearQ_v2, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        self.act = ActQ_v2(in_features=in_features, nbits_a=nbits_w)

    def forward(self, x):
        # assert not torch.is_floating_point(x)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        alpha = self.alpha.unsqueeze(1)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp))
        return F.linear(x, w_q.int())

class Q_Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, nbits, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = LinearQ_v2(in_features, hidden_features, nbits_w=nbits, mode=Qmodes.kernel_wise)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = LinearQ_v2(hidden_features, out_features, nbits_w=nbits, mode=Qmodes.kernel_wise)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x0):
        x0_2b = self.fc1.act(x0).int()
        x1 = self.fc1(x0_2b) * self.fc1.alpha * self.fc1.act.alpha.mean() + self.fc1.bias
        # x1 = self.fc1(x0)
        # import pdb; pdb.set_trace()
        # print(torch.max(x), torch.min(x))
        x1 = self.act(x1)
        
        x1_clipped = torch.clip(x1, -10., 10.)
        # print(torch.clip(x, -10., 10.))
        x1_clipped_dropped = self.drop1(x1_clipped)
        # x2 = self.fc2(x1_clipped_dropped) * self.fc2.act.alpha.mean()
        x1_2b = self.fc2.act(x1_clipped_dropped).int()
        x2 = self.fc2(x1_2b) * self.fc2.alpha * self.fc2.act.alpha.mean() + self.fc2.bias
        x2_dropped = self.drop2(x2)
        return x2_dropped

def mean_var(x, scale=1024):
    means = torch.zeros_like(x).int()
    varns = torch.zeros_like(x).int()
    for i in range(x.shape[3]):
        if i > 1:
            # means[:,:,:,i] = means[:,:,:,i - 1] + (x[:,:,:,i] - means[:,:,:,i - 1]) / (i + 1)
            means[:,:,:,i] = means[:,:,:,i - 1] + (((x[:,:,:,i] - means[:,:,:,i - 1]) * int(scale / (i + 1)))) // scale
            varns[:,:,:,i] = varns[:,:,:,i - 1] + (x[:,:,:,i] - means[:,:,:,i - 1]) * (x[:,:,:,i] - means[:,:,:,i])
        elif i > 0:
            # means[:,:,:,i] = means[:,:,:,i - 1] + (x[:,:,:,i] - means[:,:,:,i - 1]) / (i + 1)
            means[:,:,:,i] = means[:,:,:,i - 1] + (((x[:,:,:,i] - means[:,:,:,i - 1]) * int(scale / (i + 1)))) // scale
            varns[:,:,:,i] = varns[:,:,:,i - 1] + (x[:,:,:,i] - means[:,:,:,i - 1]) * (x[:,:,:,i] - means[:,:,:,i])
        else:
            means[:,:,:,i] = x[:,:,:,i]
            varns[:,:,:,i] = varns[:,:,:,i - 1] + (x[:,:,:,i] - 0) * (x[:,:,:,i] - means[:,:,:,i])

    # print(varns[:,:,:,-1][0,0,0], (x[0,0,0].std() ** 2) * x.shape[3])
    # import pdb; pdb.set_trace()
    mu = means[:,:,:,-1].unsqueeze(3)
    varn = varns[:,:,:,-1].unsqueeze(3) / x.shape[3]
    return mu, varn

class Q_Attention(nn.Module):
     
    def __init__(self, nbits, dim, num_heads=8, quantize_attn=True, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.quantize_attn = quantize_attn
        
        self.norm_q = nn.LayerNorm(head_dim)
        self.norm_k = nn.LayerNorm(head_dim)

        self.qkv = LinearQ_v2(dim, dim * 3, bias=qkv_bias, nbits_w=nbits, mode=Qmodes.kernel_wise)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = LinearQ_v2(dim, dim, nbits_w=nbits, mode=Qmodes.kernel_wise)
        self.q_act = ActQ_v2(nbits_a=nbits, in_features=self.num_heads)
        self.k_act = ActQ_v2(nbits_a=nbits, in_features=self.num_heads)
        self.v_act = ActQ_v2(nbits_a=nbits, in_features=self.num_heads)
        self.attn_act = ActQ_v2(nbits_a=nbits, in_features=self.num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def q_act_scaled(self, x):
        scale = self.norm_q.weight
        assert self.q_act.signed == True
        Qn = -2 ** (self.q_act.nbits - 1)
        Qp = 2 ** (self.q_act.nbits - 1) - 1
        alpha = self.q_act.alpha.mean()
        # zero_point = self.q_act.zero_point.round()
        # if len(x.shape)==2:
        #     zero_point = zero_point.unsqueeze(0)
        # elif len(x.shape)==4:
        #     zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # x1 = round_pass((x * scale / alpha + zero_point).clamp(Qn, Qp)) - zero_point
        x1 = round_pass((x * scale / alpha).clamp(Qn, Qp))
        return x1.int()

    def scaled_norm_qact_q(self, x, prescale=1):
        scale = self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[0] * prescale
        mean, varn = mean_var((x * scale).int())
        q1 = ((x * scale) - mean) / (torch.sqrt(varn) + 1e-5)
        # q1 = ((x * scale) - torch.mean(x * scale, (3,), keepdim=True)) / (torch.std(x * scale, (3,), keepdim=True) + 1e-5)
        q2_2b = self.q_act_scaled((q1 + self.norm_q.bias / self.norm_q.weight)).int()
        # import pdb; pdb.set_trace()
        return q2_2b


    def scaled_norm_qact_k(self, x, prescale=1):
        scale = self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[1] * prescale

        mean, varn = mean_var((x * scale).int())
        # k1 = self.norm_k(x)
        # k1 = ((x * scale) - torch.mean(x * scale, (3,), keepdim=True)) / (torch.std(x * scale, (3,), keepdim=True) + 1e-5)
        k1 = ((x * scale) - mean) / (torch.sqrt(varn) + 1e-5)
        # k2 = self.k_act(k1)
        # k2_2b = np.round(k2 / self.k_act.alpha.mean()).int()
        k2_2b = self.k_act((k1 + self.norm_k.bias / self.norm_k.weight) * self.norm_k.weight).int()
        # import pdb; pdb.set_trace()
        return k2_2b

    def scaled_softmax_qact(self, x, div=128):
        scale = self.scale * self.q_act.alpha.mean() * self.k_act.alpha.mean()
        exp = (1 / np.log(2) * scale * x)
        # attn_exp = (2 ** exp.floor()) * (2 ** (exp - exp.floor()))
        attn_exp = (2 ** exp.floor()) * (1 + (exp - exp.floor()))
        attn_exp_dropped = self.attn_drop(attn_exp)
        attn_2b = self.attn_act(attn_exp_dropped * (1 / attn_exp.sum((-1,)).unsqueeze(-1))).int()
                #   self.attn_act((2 ** (exp / div)) * (1 / (2 ** (exp / div)).sum((-1,)).unsqueeze(-1))).int()
        # import pdb; pdb.set_trace()
        return attn_2b

    def qkv_bias(self, x):
        bias = self.qkv.bias / self.qkv.act.alpha.mean() / self.qkv.alpha
        return x + bias.int()
    
    def v_act_scaled(self, x):
        scale = self.qkv.act.alpha.mean() * self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[2]
        assert self.v_act.signed == True
        Qn = -2 ** (self.v_act.nbits - 1)
        Qp = 2 ** (self.v_act.nbits - 1) - 1
        alpha = self.v_act.alpha.mean()
        # zero_point = self.v_act.zero_point.round()
        # if len(x.shape)==2:
        #     zero_point = zero_point.unsqueeze(0)
        # elif len(x.shape)==4:
        #     zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # x1 = round_pass((x * scale / alpha + zero_point).clamp(Qn, Qp)) - zero_point
        x1 = round_pass((x * scale / alpha).clamp(Qn, Qp))
        return x1.int()

    def proj_act_scaled(self, x):
        scale = self.attn_act.alpha.mean() * self.v_act.alpha.mean()
        assert self.proj.act.signed == True
        Qn = -2 ** (self.proj.act.nbits - 1)
        Qp = 2 ** (self.proj.act.nbits - 1) - 1
        alpha = self.proj.act.alpha.mean()
        # zero_point = self.proj.act.zero_point.round()
        # if len(x.shape)==2:
        #     zero_point = zero_point.unsqueeze(0)
        # elif len(x.shape)==4:
        #     zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x1 = round_pass((x * scale / alpha).clamp(Qn, Qp))
        return x1.int()

    def forward(self, x0):
        assert not np.isnan(x0).any()
        B, N, C = x0.shape
        x0_2b = self.qkv.act(x0).int()
        # open("inputfp.txt", "w").write("\n".join([("%d" % x) for x in x0[0].reshape(-1) * (2 ** 10)]))
        # open("input2b.txt", "w").write("\n".join([hex(x % 8)[2:] for x in x0_2b[0].reshape(-1)]))
        # open("input2b.txt", "w").write("\n".join([("%d" % x) for x in ((x0_2b[0].reshape(-1) + 4) % 8 - 4)]))
        # open("input2b2d.txt", "w").write(arr2d2s(x0_2b[0] % 8, nbit=3))
        # f = open("input2b64bbus.txt", "w")
        # for x in arr2d2s(x0_2b[0] % 8, nbit=3).split("\n"):
        #     assert len(x) % 16 == 0
        #     buses = [x[i - 16:i] for i in range(len(arr2d2s(x0_2b[0] % 8, nbit=3).split("\n")[0]), 0, -16)]
        #     f.write("\n".join(reversed(buses)) + "\n")
        # f.close()
        # import pdb; pdb.set_trace()
        # self.qkv.act.alpha.mean() * (2 ** 10)
        qkv = self.qkv(x0_2b)
        # qkv weight output: open("qkv_weightq.mem", "w").write("\n".join([str(p % 8) for p in round_pass((self.qkv.weight / self.qkv.alpha.unsqueeze(1)).clamp(-4, 3)).reshape(3, self.num_heads,1,-1,self.qkv.weight.shape[1])[0,0,0].permute(1,0).reshape(-1).round().int().tolist()]));
        # qkv weight output: open("qkv_weightqr.mem", "w").write("\n".join([str(p % 8) for p in np.array(round_pass((self.qkv.weight / self.qkv.alpha.unsqueeze(1)).clamp(-4, 3)).reshape(3, self.num_heads,1,-1,self.qkv.weight.shape[1])[0,0,0].permute(1,0).round().int())[:,::-1].reshape(-1).tolist()]));
        # qkv weight output: open("qkv_weightk.mem", "w").write("\n".join([str(p % 8) for p in round_pass((self.qkv.weight / self.qkv.alpha.unsqueeze(1)).clamp(-4, 3)).reshape(3, self.num_heads,1,-1,self.qkv.weight.shape[1])[1,0,0].permute(1,0).reshape(-1).round().int().tolist()]));
        # qkv weight output: open("qkv_weightkr.mem", "w").write("\n".join([str(p % 8) for p in np.array(round_pass((self.qkv.weight / self.qkv.alpha.unsqueeze(1)).clamp(-4, 3)).reshape(3, self.num_heads,1,-1,self.qkv.weight.shape[1])[1,0,0].permute(1,0).round().int())[:,::-1].reshape(-1).tolist()]));
        # qkv weight output: open("qkv_weightvr.mem", "w").write("\n".join([str(p % 8) for p in np.array(round_pass((self.qkv.weight / self.qkv.alpha.unsqueeze(1)).clamp(-4, 3)).reshape(3, self.num_heads,1,-1,self.qkv.weight.shape[1])[2,0,0].permute(1,0).round().int())[:,::-1].reshape(-1).tolist()]));
        qkv_biased = self.qkv_bias(qkv)
        # qkv bias output: open("qkv_biasq.mem",  "w").write("\n".join(([hex(p % (2 ** 16))[2:] for p in (self.qkv.bias / self.qkv.act.alpha.mean() / self.qkv.alpha).reshape(3, self.num_heads,1,-1)[0,0,0].int().tolist()])))
        # qkv bias output: open("qkv_biasqr.mem", "w").write("\n".join(([hex(p % (2 ** 16))[2:] for p in ((self.qkv.bias / self.qkv.act.alpha.mean() / self.qkv.alpha).reshape(3, self.num_heads,1,-1)[0,0,0].int())[::-1].tolist()])))
        # qkv bias output: open("qkv_biask.mem", "w").write("\n".join(([hex(p % (2 ** 16))[2:] for p in (self.qkv.bias / self.qkv.act.alpha.mean() / self.qkv.alpha).reshape(3, self.num_heads,1,-1)[1,0,0].int().tolist()])))
        # qkv bias output: open("qkv_biaskr.mem", "w").write("\n".join(([hex(p % (2 ** 16))[2:] for p in np.array((self.qkv.bias / self.qkv.act.alpha.mean() / self.qkv.alpha).reshape(3, self.num_heads,1,-1)[1,0,0].int())[::-1].tolist()])))
        # qkv bias output: open("qkv_biasvr.mem", "w").write("\n".join(([hex(p % (2 ** 16))[2:] for p in np.array((self.qkv.bias / self.qkv.act.alpha.mean() / self.qkv.alpha).reshape(3, self.num_heads,1,-1)[2,0,0].int())[::-1].tolist()])))
        qkv_reshaped = (qkv_biased).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv_reshaped = 1
        # alpha_reshaped = self.qkv.alpha.reshape(3, self.num_heads, 1, C // self.num_heads)
        # alpha_reshaped = 1
        
        # def array2d2s(arr, prefix=""):
        #     assert len(arr.shape) == 2
        #     s = ""
        #     for arr1d in arr:
        #         s += " ".join([(prefix + "%d" % int(p)) for p in arr1d.tolist()]) + "\n"
        #     return s

        q0, k0, v0 = qkv_reshaped.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q2_2b = self.scaled_norm_qact_q(q0) #, qkv_scale[0])
        # snqq_scale: open("snqq_scale.mem", "w").write("\n".join([hex(p % (2 ** 32))[2:] for p in (self.qkv.alpha.reshape(3, self.num_heads,1,-1) * 1024).round().int()[0,0,0].tolist()]))
        # snqq_scale: open("snqq_scaler.mem", "w").write("\n".join([hex(p % (2 ** 32))[2:] for p in reversed((self.qkv.alpha.reshape(3, self.num_heads,1,-1) * 1024).round().int()[0,0,0].tolist())]))
        # snqq_bias:  open("snqq_bias_by256.mem", "w").write("\n".join([hex(-p % (2 ** 32))[2:] for p in (self.norm_q.bias / self.norm_q.weight * 256).round().int().tolist()]))
        # snqq_bias:  open("snqq_biasr_by256.mem", "w").write("\n".join([hex(-p % (2 ** 32))[2:] for p in reversed((self.norm_q.bias / self.norm_q.weight * 256).round().int().tolist())]))
        # qact_alpha: self.q_act.alpha.mean() * 256 
        # snqq_step: open("snqq_step_by256.mem", "w").write("\n".join([hex(p % (2 ** 32))[2:] for p in (self.q_act.alpha.mean() * 256 / self.norm_q.weight).round().int().tolist()]))
        # snqq_step: open("snqq_stepr_by256.mem", "w").write("\n".join([hex(p % (2 ** 32))[2:] for p in reversed((self.q_act.alpha.mean() * 256 / self.norm_q.weight).round().int().tolist())]))
        k2_2b = self.scaled_norm_qact_k(k0) #, qkv_scale[1])
        # snqq_scale: open("snqk_scale.mem", "w").write("\n".join([hex(p % (2 ** 32))[2:] for p in (self.qkv.alpha.reshape(3, self.num_heads,1,-1) * 1024).round().int()[1,0,0].tolist()]))
        # snqq_scale: open("snqk_scaler.mem", "w").write("\n".join(reversed([hex(p % (2 ** 32))[2:] for p in (self.qkv.alpha.reshape(3, self.num_heads,1,-1) * 1024).round().int()[1,0,0].tolist()])))
        # snqq_bias:  open("snqk_bias_by256.mem", "w").write("\n".join([hex(-p % (2 ** 32))[2:] for p in (self.norm_k.bias / self.norm_k.weight * 256).round().int().tolist()]))
        # snqq_bias:  open("snqk_biasr_by256.mem", "w").write("\n".join(reversed([hex(-p % (2 ** 32))[2:] for p in (self.norm_k.bias / self.norm_k.weight * 256).round().int().tolist()])))        
        # open("snqk_step_by256.mem", "w").write("\n".join([hex(p % (2 ** 16))[2:] for p in (self.k_act.alpha.mean() * 256 / self.norm_k.weight).round().int().tolist()]))
        # open("snqk_stepr_by256.mem", "w").write("\n".join(reversed([hex(p % (2 ** 16))[2:] for p in (self.k_act.alpha.mean() * 256 / self.norm_k.weight).round().int().tolist()])))
        # open("k2_2b.mem", "w").write(" ".join(["%d" % p for p in k2_2b[0,0].reshape(-1)]))
        # import pdb; pdb.set_trace()
        v2_2b = self.v_act_scaled(v0) # * alpha_reshaped[2])
        # open("v2rev_2b.mem", "w").write(" ".join(["%d" % p for p in np.array(v2_2b)[0,0,:,::-1].reshape(-1)]))
        # open("vstep.mem", "w").write(" ".join([hex(p)[2:] for p in (self.v_act.alpha.mean() / (self.qkv.act.alpha.mean() * self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[2,0,0])).round().int().tolist()]))
        # open("vhstep.mem", "w").write(" ".join([hex(p)[2:] for p in (.5 * self.v_act.alpha.mean() / (self.qkv.act.alpha.mean() * self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[2,0,0])).round().int().tolist()]))
        # open("vrhstep.mem", "w").write(" ".join([hex(p)[2:] for p in reversed(.5 * self.v_act.alpha.mean() / (self.qkv.act.alpha.mean() * self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[2,0,0])).round().int().tolist()]))
        # import pdb; pdb.set_trace()

        qk_prod = (q2_2b @ k2_2b.transpose(-2, -1))
        attn_2b = self.scaled_softmax_qact(qk_prod)
        # scale * 1 / np.log(2) * (2 ** 17)

        x1_prod = (attn_2b @ v2_2b).transpose(1, 2).reshape(B, N, C).int()
        # attnstep = self.proj.act.alpha.mean() / (self.attn_act.alpha.mean() * self.v_act.alpha.mean()) * 16
        x1_2b = self.proj_act_scaled(x1_prod)
        x2 = self.proj(x1_2b) * self.proj.alpha * self.proj.act.alpha.mean() + self.proj.bias
        x3 = self.proj_drop(x2)

        # for idx_head in range(6):
        #     q0_scaled_by16 = (q0 * self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[0])[0, idx_head] * 16
        #     open("groundtruths/outq_gt%d.txt" % idx_head, "w").write(array2d2s(q0_scaled_by16.round().int()))
        #     q0_prescaled = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0, 0, idx_head]
        #     open("groundtruths/outqpre_gt%d.txt" % idx_head, "w").write(array2d2s(q0_prescaled))
        #     k0_scaled_by16 = (k0 * self.qkv.alpha.reshape(3, self.num_heads, 1, -1)[1])[0, idx_head] * 16
        #     open("groundtruths/outk_gt%d.txt" % idx_head, "w").write(array2d2s(k0_scaled_by16.round().int()))
        #     k0_prescaled = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[1, 0, idx_head]
        #     open("groundtruths/outkpre_gt%d.txt" % idx_head, "w").write(array2d2s(k0_prescaled))
        #     open("groundtruths/outv_gt%d.txt" % idx_head, "w").write(array2d2s(v0[0, idx_head]))
        #     open("groundtruths/outqk_gt%d.txt" % idx_head, "w").write(array2d2s(qk_prod[0, idx_head]))        
        #     open("groundtruths/out_attn_gt%d.txt" % idx_head, "w").write(array2d2s((attn_2b @ v2_2b)[0, idx_head]))
        #     open("groundtruths/out_qkvp_gt%d.txt" % idx_head, "w").write(array2d2s(self.proj(x1_2b * torch.Tensor([0,] * idx_head * 64 + [1,] * 64 + [0,] * (5 - idx_head) * 64).int())[0]))        
        # import pdb; pdb.set_trace()
        return x3

class Q_Block(nn.Module):

    def __init__(self, nbits, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Q_Attention(nbits, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        print("drop_path = ", drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Q_Mlp(nbits=nbits, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x0):
        x1 = x0 + self.drop_path(self.attn(self.norm1(x0)))
        x2 = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        # print(np.percentile(x0, (0, 5, 95, 100)), x0.std())
        # import pdb; pdb.set_trace()
        return x2

class lowbit_VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, nbits, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=Q_PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            nbits: nbits
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # import pdb; pdb.set_trace()
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            nbits=nbits, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Q_Block(
                nbits=nbits, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = LinearQ(self.num_features, num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
        # nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = LinearQ(self.embed_dim, self.num_classes, nbits_w=8) if num_classes > 0 else nn.Identity()
        # import pdb; pdb.set_trace()

    def forward_features(self, x0):
        x1 = self.patch_embed(x0)
        assert not np.isnan(x1).any()
        cls_token = self.cls_token.expand(x1.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x2 = torch.cat((cls_token, self.dist_token.expand(x1.shape[0], -1, -1), x1), dim=1)
        assert not np.isnan(x2).any()
        x3 = self.pos_drop(x2 + self.pos_embed)
        assert not np.isnan(x3).any()
        x4 = self.blocks(x3)
        x5 = self.norm(x4)
        # import pdb; pdb.set_trace()
        return x5[:, 0], x5[:, 1]

    def forward(self, x0):
        x1 = self.forward_features(x0)
        x, x_dist = self.head(x1[0]), self.head_dist(x1[1])  # x must be a tuple
        # import pdb; pdb.set_trace()
        return (x + x_dist) / 2
