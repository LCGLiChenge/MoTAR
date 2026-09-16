"""Ordinary dense 16x16 BERT MaskGIT conditioned by fixed proxy token anchors.

The only transformer sequence is [class, grid0, ..., grid255]. Continuous
features are consumed outside this model by a frozen converter, never appended
to its sequence. Optional output_indices only avoids evaluating unused LM heads;
it does not change the encoder sequence or attention.
"""
import math
import os
os.environ.setdefault('USE_TF','0')
import torch
from torch import nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from bert2d.model import BertSparse2D, load_titok_backbone
from bert2d.runtime import smoothed_objective
from h20.base_model import sample_arccos_mask


class GridMaskGIT(nn.Module):
    grid_tokens=256
    mask_token=16384
    vocabulary=16384

    def __init__(self,dim=768,depth=24,heads=16,dropout=.1):
        super().__init__()
        self.dim=dim
        self.config=dict(dim=dim,depth=depth,heads=heads,dropout=dropout)
        cfg=BertConfig(hidden_size=dim,num_hidden_layers=depth,num_attention_heads=heads,
            intermediate_size=4*dim,hidden_act='gelu',hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,layer_norm_eps=1e-12,is_decoder=False)
        cfg._attn_implementation='sdpa'
        self.encoder=BertEncoder(cfg)
        self.input_norm=nn.LayerNorm(dim,eps=1e-12)
        self.input_dropout=nn.Dropout(dropout)
        self.class_embedding=nn.Embedding(1001,dim)
        self.class_position=nn.Parameter(torch.zeros(1,1,dim))
        self.token_type=nn.Parameter(torch.zeros(1,1,dim))
        self.token_embedding=nn.Embedding(self.vocabulary+1,dim)
        self.spatial_embedding=nn.Embedding(self.grid_tokens,dim)
        self.head=nn.Linear(dim,self.vocabulary)
        self.apply(BertSparse2D._initialize)

    enable_recompute=BertSparse2D.enable_recompute

    def forward(self,tokens,labels,force_drop_ids=None,output_indices=None):
        if tokens.ndim!=2 or tokens.shape[1]!=256 or tokens.dtype!=torch.long:
            raise ValueError('ordinary MaskGIT requires full [B,256] token grid')
        if labels.shape!=(len(tokens),):raise ValueError('labels must be [B]')
        if bool(((tokens<0)|(tokens>self.mask_token)).any()):raise ValueError('invalid grid token')
        if force_drop_ids is None:force_drop_ids=torch.zeros_like(labels,dtype=torch.bool)
        labels=torch.where(force_drop_ids.bool(),1000,labels)
        cls=self.class_embedding(labels)[:,None]+self.class_position
        grid=self.token_embedding(tokens)+self.spatial_embedding.weight[None]
        h=self.input_dropout(self.input_norm(torch.cat([cls,grid],1)+self.token_type))
        h=self.encoder(h,attention_mask=None,return_dict=True).last_hidden_state[:,1:]
        if output_indices is not None:
            if output_indices.ndim!=2 or len(output_indices)!=len(tokens):raise ValueError('bad output indices')
            h=h.gather(1,output_indices.clamp(0,255)[...,None].expand(-1,-1,self.dim))
        return self.head(h).float()


def construct_training_grid(proxy,gt_grid,selected,masked,mask_token=16384):
    if proxy.shape!=gt_grid.shape or proxy.shape!=selected.shape or proxy.shape!=masked.shape:
        raise ValueError('full-grid shapes differ')
    if selected.dtype!=torch.bool or masked.dtype!=torch.bool:raise ValueError('boolean masks required')
    if bool((masked&~selected).any()):raise ValueError('cannot mask fixed anchor positions')
    tokens=torch.where(selected,gt_grid,proxy)
    return tokens.masked_fill(masked,mask_token)


def selected_mask(indices,valid):
    if indices.shape!=valid.shape or valid.dtype!=torch.bool:raise ValueError('invalid routes')
    if bool(((indices[valid]<0)|(indices[valid]>=256)).any()):raise ValueError('invalid selected position')
    result=torch.zeros(len(indices),256,dtype=torch.bool,device=indices.device)
    batch=torch.arange(len(indices),device=indices.device)[:,None].expand_as(indices)[valid]
    result[batch,indices[valid]]=True
    if not torch.equal(result.sum(1),valid.sum(1)):raise ValueError('duplicate selected positions')
    return result


class GridObjective(nn.Module):
    def __init__(self,core):super().__init__();self.core=core
    def forward(self,proxy,gt_grid,labels,indices,valid):
        selected=selected_mask(indices,valid)
        masked,_=sample_arccos_mask(selected)
        inputs=construct_training_grid(proxy,gt_grid,selected,masked)
        drop=torch.rand(len(labels),device=labels.device)<.1
        # Only evaluate selected-position LM heads; all 257 tokens enter every
        # transformer layer. This equals selecting these logits after a full head.
        logits=self.core(inputs,labels,drop,indices)
        target=gt_grid.gather(1,indices.clamp_min(0))
        local_mask=masked.gather(1,indices.clamp_min(0))&valid
        loss,nll=smoothed_objective(logits,target,local_mask,valid)
        return loss,torch.stack([loss.detach(),nll,masked.sum()/selected.sum(),drop.float().mean()])


@torch.no_grad()
def sample_grid(model,proxy,selected,labels,steps=16,cfg=4.5,choice_temperature=1.,trace=None):
    """Standard categorical/confidence MaskGIT completion with a cosine schedule.

    Initial anchors AND accepted predictions remain fixed. Mask counts refer to
    the initially selected K, not all256. No Halton ordering, margins or blending.
    """
    if proxy.shape!=selected.shape or selected.dtype!=torch.bool:raise ValueError('invalid proxy/selection')
    if bool(((proxy<0)|(proxy>=model.vocabulary)).any()):raise ValueError('proxy must contain content IDs')
    if steps<1:raise ValueError('positive step count required')
    tokens=proxy.masked_fill(selected,model.mask_token)
    counts=selected.sum(1)
    for step in range(steps):
        unknown=tokens==model.mask_token
        if not bool(unknown.any()):break
        cond=model(tokens,labels)
        if cfg!=1:
            uncond=model(tokens,labels,torch.ones_like(labels,dtype=torch.bool))
            logits=uncond+cfg*(cond-uncond)
        else:logits=cond
        gumbel=-torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-6,1-1e-6)))
        sampled=(logits+gumbel).argmax(-1)
        sampled=torch.where(unknown,sampled,tokens)
        ratio=(step+1)/steps
        if step==steps-1:
            tokens=sampled
        else:
            confidence=logits.gather(-1,sampled[...,None]).squeeze(-1)-torch.logsumexp(logits,-1)
            noise=-torch.log(-torch.log(torch.rand_like(confidence).clamp_(1e-6,1-1e-6)))
            confidence=(confidence+choice_temperature*(1-ratio)*noise).masked_fill(~unknown,float('inf'))
            remain=torch.floor(counts.float()*math.cos(ratio*math.pi/2)).long()
            remain=torch.minimum(remain,(unknown.sum(1)-1).clamp_min(0))
            order=confidence.argsort(1)
            remask=torch.zeros_like(selected)
            ranks=torch.arange(256,device=tokens.device)[None]<remain[:,None]
            remask.scatter_(1,order,ranks)
            assert not bool((remask&~unknown).any())
            tokens=sampled.masked_fill(remask,model.mask_token)
        assert torch.equal(tokens[~selected],proxy[~selected])
        if trace is not None:trace.append(dict(step=step,remaining=int((tokens==model.mask_token).sum()),anchors_exact=True))
    if bool((tokens==model.mask_token).any()):raise RuntimeError('completion left masked positions')
    return tokens


def test():
    torch.set_num_threads(2);torch.manual_seed(33)
    model=GridMaskGIT(dim=32,depth=2,heads=4,dropout=0.)
    proxy=torch.randint(16384,(2,256));gt=torch.randint(16384,(2,256));labels=torch.tensor([3,7])
    selected=torch.zeros(2,256,dtype=torch.bool);selected[0,:64]=True;selected[1,128:]=True
    masked=selected & (torch.arange(256)[None]%2==0)
    tokens=construct_training_grid(proxy,gt,selected,masked)
    assert torch.equal(tokens[~selected],proxy[~selected])
    poisoned=gt.clone();poisoned[~selected]=(poisoned[~selected]+123)%16384
    assert torch.equal(tokens,construct_training_grid(proxy,poisoned,selected,masked))
    assert torch.equal(tokens[selected&~masked],gt[selected&~masked])
    assert (tokens[masked]==16384).all()
    lengths=[]
    hook=model.encoder.register_forward_pre_hook(lambda module,args:lengths.append(args[0].shape[1]))
    indices=torch.stack([torch.arange(128),torch.arange(128,256)])
    full=model(tokens,labels);sub=model(tokens,labels,output_indices=indices)
    torch.testing.assert_close(sub,full.gather(1,indices[...,None].expand(-1,-1,16384)))
    valid=torch.ones(2,128,dtype=torch.bool);valid[0,64:]=False
    loss,_=GridObjective(model)(proxy,gt,labels,indices,valid);loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    trace=[];model.eval()
    result=sample_grid(model,proxy,selected,labels,steps=4,cfg=1,trace=trace)
    assert torch.equal(result[~selected],proxy[~selected]) and (result<16384).all()
    assert all(x==257 for x in lengths)
    assert all(b['remaining']<=a['remaining'] for a,b in zip(trace,trace[1:]))
    empty=torch.zeros_like(selected)
    assert torch.equal(sample_grid(model,proxy,empty,labels),proxy)
    hook.remove()
    print('PASS full257 sequence, no GT anchor leakage, gathered-head equivalence, finite backward, fixed anchors, completed sampling')


if __name__=='__main__':test()
