# Di-R
Diffusion with RWKV

## Replace Transformer with RWKV-v6 block

This project is to replace the transformer in the original Di-R with RWKV-v6 block.

### RWKV-v6 block
RWKV-v6 block is a RNN-like block with a large number of parameters. It can be used to replace the transformer in the original Di-R.

### Model architecture

The original DiT model architecture is as follows:

![DiT](https://www.wpeebles.com/images/DiT/block.png)

We just replace the DiT Block to DiR Block.

![DiR](DIRWKVBlock.png)