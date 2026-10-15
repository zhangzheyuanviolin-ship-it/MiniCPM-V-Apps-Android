"""Small deterministic hybrid Qwen3.5 MoE; compares inline weights with streamed experts."""
import sys,pathlib,json
import numpy as np
import gguf
root=pathlib.Path(sys.argv[1]);(root/'experts').mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(7)
D,F,E,V,L=32,16,4,32,4
tensors={}
def tensor(name,shape,norm=False):
 a=np.ones(shape,dtype=np.float32) if norm else (rng.standard_normal(shape)*0.025).astype(np.float32)
 tensors[name]=a
 return a
tensor('token_embd.weight',(V,D));tensor('output_norm.weight',(D,),True);tensor('output.weight',(V,D))
entries=[]
for i in range(L):
 p=f'blk.{i}.'
 tensor(p+'attn_norm.weight',(D,),True);tensor(p+'post_attention_norm.weight',(D,),True)
 if i%2==1:
  for n,shape in [('attn_q',(64,D)),('attn_k',(16,D)),('attn_v',(16,D)),('attn_output',(D,D))]:tensor(p+n+'.weight',shape)
  tensor(p+'attn_q_norm.weight',(16,),True);tensor(p+'attn_k_norm.weight',(16,),True)
 else:
  tensor(p+'attn_qkv.weight',(64,D));tensor(p+'attn_gate.weight',(32,D))
  tensor(p+'ssm_conv1d.weight',(64,4));tensor(p+'ssm_dt.bias',(2,))
  tensors[p+'ssm_a']=np.array([-1.,-1.],dtype=np.float32)
  tensor(p+'ssm_beta.weight',(2,D));tensor(p+'ssm_alpha.weight',(2,D))
  tensor(p+'ssm_norm.weight',(16,),True);tensor(p+'ssm_out.weight',(D,32))
 tensor(p+'ffn_gate_inp.weight',(E,D))
 routed=[]
 for name,shape in [('ffn_gate_exps',(E,F,D)),('ffn_up_exps',(E,F,D)),('ffn_down_exps',(E,D,F))]:
  routed.append((name,tensor(p+name+'.weight',shape)))
 tensor(p+'ffn_gate_inp_shexp.weight',(D,))
 for n,shape in [('ffn_gate_shexp',(F,D)),('ffn_up_shexp',(F,D)),('ffn_down_shexp',(D,F))]:tensor(p+n+'.weight',shape)
 stride=sum(a[0].nbytes for _,a in routed);data=b''.join(a[e].tobytes() for e in range(E) for _,a in routed)
 file=f'layer_{i:03}.bin';(root/'experts'/file).write_bytes(data);offset=0
 for name,a in routed:
  entries.append({'layer':i,'tensor_family':name,'tensor_name':p+name+'.weight','quant_type':'F32','exact_byte_length':a.nbytes,'bytes_per_expert':a[0].nbytes,'expert_count':E,'repacked_file':file,'repacked_offset':offset,'expert_stride':stride})
  offset+=a[0].nbytes
manifest={'schema_version':1,'layout':'layer_major_expert','experts_kind':'omni_moe_gguf','model':{'arch':'qwen35moe','expert_count':E,'expert_used_count':2},'entries':entries}
(root/'experts/manifest.json').write_text(json.dumps(manifest))
for split in [False,True]:
 w=gguf.GGUFWriter(str(root/('split.gguf' if split else 'inline.gguf')),'qwen35moe')
 for k,v in {'context_length':128,'embedding_length':D,'block_count':L,'feed_forward_length':F,'attention.head_count':2,'attention.head_count_kv':1,'attention.key_length':16,'attention.value_length':16,'rope.dimension_count':16,'expert_count':E,'expert_used_count':2,'expert_feed_forward_length':F,'expert_shared_feed_forward_length':F,'ssm.conv_kernel':4,'ssm.inner_size':32,'ssm.state_size':16,'ssm.time_step_rank':2,'ssm.group_count':1,'full_attention_interval':2}.items():w.add_uint32('qwen35moe.'+k,v)
 w.add_float32('qwen35moe.attention.layer_norm_rms_epsilon',1e-6)
 w.add_array('qwen35moe.rope.dimension_sections',[4,4,0,0])
 w.add_string('tokenizer.ggml.model','llama')
 w.add_array('tokenizer.ggml.tokens',['<unk>','<s>','</s>']+[f'token{i}' for i in range(V-3)])
 w.add_array('tokenizer.ggml.scores',[0.]*V)
 w.add_array('tokenizer.ggml.token_type',[2,3,3]+[1]*(V-3))
 w.add_uint32('tokenizer.ggml.unknown_token_id',0);w.add_uint32('tokenizer.ggml.bos_token_id',1);w.add_uint32('tokenizer.ggml.eos_token_id',2)
 for name,a in tensors.items():
  if split and any(f in name for f in ['ffn_gate_exps','ffn_up_exps','ffn_down_exps']):continue
  w.add_tensor(name,a)
 w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
print('Synthetic model files created; no published model weights downloaded')
