#!/usr/bin/env python3
"""Export ZECBTC book_ticker (+ optional depth) as a flat f32 binary stream.

Output format: raw little-endian f32 triples (mid, depth_imb, flow_imb), no header.
The Rust replay reader infers record count from file size.

Defaults match the rich_collector layout on this box.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--book',
                    default='/steam/rotko/ctm/ctm-agent/data/orderbook/rich/ZECBTC_book_ticker.parquet')
    ap.add_argument('--depth',
                    default='/steam/rotko/ctm/ctm-agent/data/orderbook/rich/ZECBTC_depth.parquet',
                    help='depth parquet for imbalance feature; pass "" to skip')
    ap.add_argument('--out', default='/tmp/zec_ticks.bin')
    ap.add_argument('--limit', type=int, default=0,
                    help='clip to last N rows (0 = all)')
    args = ap.parse_args()

    bk = pd.read_parquet(args.book)
    bk = bk.dropna(subset=['bid', 'ask', 'bid_sz', 'ask_sz'])
    bk = bk[(bk.bid > 0) & (bk.ask > 0) & (bk.ask >= bk.bid)]
    bk = bk.sort_values('timestamp').reset_index(drop=True)

    mid = 0.5 * (bk.bid + bk.ask)
    # Top-of-book imbalance as fallback when depth unavailable
    tob_imb = ((bk.bid_sz - bk.ask_sz) / (bk.bid_sz + bk.ask_sz)).clip(-1, 1).fillna(0)

    if args.depth:
        try:
            dp = pd.read_parquet(args.depth)
            dp = dp[['timestamp', 'imbalance_10']].dropna()
            dp = dp[dp.imbalance_10.between(-1, 1)]
            merged = pd.merge_asof(
                bk[['timestamp']].assign(_ord=range(len(bk))),
                dp.sort_values('timestamp'),
                on='timestamp', direction='backward', tolerance=2000)
            depth_imb = merged.imbalance_10.fillna(tob_imb).to_numpy(dtype=np.float32)
        except Exception as e:
            print(f'depth merge failed ({e!r}); using top-of-book imbalance only',
                  file=sys.stderr)
            depth_imb = tob_imb.to_numpy(dtype=np.float32)
    else:
        depth_imb = tob_imb.to_numpy(dtype=np.float32)

    # Flow proxy: 1-tick mid return, sign-clipped. Real flow needs aggTrades;
    # this is a coarse stand-in until live wsfeed feeds raw trades into the brain.
    mid_arr = mid.to_numpy(dtype=np.float32)
    ret = np.diff(mid_arr, prepend=mid_arr[0]) / np.maximum(mid_arr, 1e-12)
    flow_proxy = np.tanh(ret * 1e4).astype(np.float32)  # ~[-1, 1]

    if args.limit > 0:
        mid_arr = mid_arr[-args.limit:]
        depth_imb = depth_imb[-args.limit:]
        flow_proxy = flow_proxy[-args.limit:]

    n = len(mid_arr)
    out = np.empty(n * 3, dtype=np.float32)
    out[0::3] = mid_arr
    out[1::3] = depth_imb
    out[2::3] = flow_proxy

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        f.write(out.tobytes())

    span = (f'{pd.to_datetime(bk.timestamp.iloc[-n], unit="ms")} -> '
            f'{pd.to_datetime(bk.timestamp.iloc[-1], unit="ms")}')
    print(f'wrote {args.out}: {n:,} records ({n*12/1e6:.1f} MB) {span}')
    print(f'mid:        [{mid_arr.min():.6f}, {mid_arr.max():.6f}], mean={mid_arr.mean():.6f}')
    print(f'depth_imb:  [{depth_imb.min():.3f}, {depth_imb.max():.3f}], mean={depth_imb.mean():+.3f}')
    print(f'flow_proxy: [{flow_proxy.min():.3f}, {flow_proxy.max():.3f}], '
          f'p99={np.quantile(np.abs(flow_proxy), 0.99):.3f}')


if __name__ == '__main__':
    main()
