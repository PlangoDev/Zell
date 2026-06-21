"""Hot-path kernels, isolated so they can be tuned without touching model logic.

  fused_act       codes-bound input @ projection - bias, in one cuBLAS call
  approx_topk     two-level top-k (group -> candidates -> final): ~3-5x cheaper
                  than a full top-k over n_gran, with negligible quality loss
                  because kWTA already discards near-threshold granules
  scatter_add_2d  flat 1D scatter-add for the (granule, word) readout update,
                  replacing 2D index_put_(accumulate=True) and its atomic
                  contention on the dominant Ww payload
"""
import torch
import torch.nn.functional as F


def fused_act(x, proj_t, neg_gbias):
    """x [B, IN] @ proj [IN, G] - gbias, computed as F.linear(x, proj_t, neg_gbias)
    = x @ proj_t.T + neg_gbias in a single fused matmul+bias kernel.

    proj_t is proj.T (shape [G, IN]); neg_gbias is -gbias (shape [G])."""
    return F.linear(x, proj_t, neg_gbias)


def approx_topk(a, groups, per_group, K):
    """Approximate top-(K+1) over the granule axis of `a` [B, G].

    Split G into `groups` contiguous blocks of size G//groups, take the top
    `per_group` within each block (cheap, local), gather those candidates, then
    run an exact top-(K+1) over the candidate set. Returns (idx [B,K], margin
    [B,K] fp32) with the kWTA relu margin already applied.

    Requires G % groups == 0 and groups*per_group >= K+1 (asserted by the caller
    through config; here we fall back to an exact top-k if the shape is off so the
    smoke/CPU path can never silently break).
    """
    B, G = a.shape
    gs = G // groups
    if gs * groups != G or groups * per_group < K + 1:
        # safety fallback: exact path (used only on odd shapes)
        vals, idx = a.topk(K + 1, dim=1)
        margin = (vals[:, :K] - vals[:, K:K + 1]).clamp(min=0)
        return idx[:, :K], margin.float()

    ag = a.view(B, groups, gs)                                  # [B, groups, gs]
    loc_vals, loc_idx = ag.topk(per_group, dim=2)              # [B, groups, pg]
    offsets = (torch.arange(groups, device=a.device) * gs).view(1, groups, 1)
    glob_idx = (loc_idx + offsets).reshape(B, groups * per_group)   # [B, C]
    cand_vals = loc_vals.reshape(B, groups * per_group)             # [B, C]
    top_vals, top_pos = cand_vals.topk(K + 1, dim=1)          # [B, K+1]
    top_idx = glob_idx.gather(1, top_pos)                      # [B, K+1] global ids
    margin = (top_vals[:, :K] - top_vals[:, K:K + 1]).clamp(min=0)
    return top_idx[:, :K], margin.float()


def scatter_add_2d(W, rows, cols, vals):
    """In-place W[rows, cols] += vals via a flat 1D scatter_add_ on W.view(-1).

    The 1D scatter_add_ kernel is materially faster than index_put_ with
    accumulate=True on a 2D index, which is the v13 word-head bottleneck. rows,
    cols, vals are 1D and equal length; W is 2D and contiguous.
    """
    ncols = W.size(1)
    flat_idx = rows * ncols + cols
    W.view(-1).scatter_add_(0, flat_idx, vals)
