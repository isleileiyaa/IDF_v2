import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm


def list_parquet_files(path):
    return sorted(str(p) for p in Path(path).glob("*.parquet"))


def list_column_to_2d_numpy(column, dtype=np.float32):
    arr = column.combine_chunks()
    n = len(arr)

    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
        values = arr.values.to_numpy(zero_copy_only=False)
        dim = len(values) // n
        return np.asarray(values, dtype=dtype).reshape(n, dim)

    # fallback，慢一些，但更兼容
    return np.asarray(arr.to_pylist(), dtype=dtype)


def normalize_np_rows(x, method="zscore", eps=1e-8):
    x = np.asarray(x, dtype=np.float32)

    if method == "zscore":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        x = (x - mean) / std
    elif method == "minmax":
        mn = x.min(axis=1, keepdims=True)
        mx = x.max(axis=1, keepdims=True)
        span = np.where((mx - mn) < eps, 1.0, mx - mn)
        x = (x - mn) / span
    else:
        raise ValueError(f"Unsupported norm method: {method}")

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm < eps, 1.0, norm)
    return x / norm


def normalize_torch_rows(x, method="zscore", eps=1e-8):
    x = x.float()

    if method == "zscore":
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        std = torch.where(std < eps, torch.ones_like(std), std)
        x = (x - mean) / std
    elif method == "minmax":
        mn = x.amin(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        span = torch.where((mx - mn) < eps, torch.ones_like(mx), mx - mn)
        x = (x - mn) / span
    else:
        raise ValueError(f"Unsupported norm method: {method}")

    norm = torch.linalg.norm(x, dim=1, keepdim=True)
    norm = torch.where(norm < eps, torch.ones_like(norm), norm)
    return x / norm


def load_database_x(db_path, context_length, norm_method):
    print("Loading retrieval database x ...", flush=True)
    table = pq.read_table(db_path, columns=["x"])
    db_x = list_column_to_2d_numpy(table["x"])[:, :context_length]

    print(f"Loaded db_x shape={db_x.shape}, dtype={db_x.dtype}", flush=True)
    print("Normalizing db_x on CPU ...", flush=True)
    db_x = normalize_np_rows(db_x, method=norm_method).astype(np.float32, copy=False)
    print(f"Normalized db_x shape={db_x.shape}, memory={db_x.nbytes / 1024 ** 3:.2f} GB", flush=True)
    return db_x


def compute_topk_for_query_batch(
    query_targets,
    db_x_norm,
    context_length,
    top_k,
    norm_method,
    db_batch_size,
    device,
    use_fp16,
    first_log,
):
    q_np = np.asarray(query_targets, dtype=np.float32)[:, :context_length]
    q = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)
    q = normalize_torch_rows(q, method=norm_method)

    if use_fp16 and device.type == "cuda":
        q = q.half()

    bsz = q.shape[0]
    best_sims = torch.full((bsz, top_k), -float("inf"), device=device, dtype=torch.float32)
    best_indices = torch.full((bsz, top_k), -1, device=device, dtype=torch.long)

    n_db = db_x_norm.shape[0]

    for start in range(0, n_db, db_batch_size):
        end = min(start + db_batch_size, n_db)
        db_chunk_np = db_x_norm[start:end]

        if first_log["db"]:
            print(f"First db batch shape={db_chunk_np.shape}, moving to {device}", flush=True)
            print("Starting first matmul...", flush=True)

        db = torch.from_numpy(db_chunk_np).to(device=device, dtype=torch.float32)
        if use_fp16 and device.type == "cuda":
            db = db.half()

        sims = q @ db.T
        sims = sims.float()

        if first_log["db"]:
            print("First matmul done", flush=True)
            first_log["db"] = False

        cand_idx = torch.arange(start, end, device=device, dtype=torch.long)
        cand_idx = cand_idx.unsqueeze(0).expand(bsz, -1)

        merged_sims = torch.cat([best_sims, sims], dim=1)
        merged_indices = torch.cat([best_indices, cand_idx], dim=1)

        best_sims, order = torch.topk(merged_sims, k=top_k, dim=1)
        best_indices = torch.gather(merged_indices, 1, order)

        del db, sims, cand_idx, merged_sims, merged_indices

    distances = 1.0 - best_sims
    return best_indices.cpu().numpy(), distances.cpu().numpy()


def write_output(output_path, targets, starts, indices, distances):
    table = pa.table({
        "target": targets,
        "indices": [row.tolist() for row in indices],
        "distances": [row.tolist() for row in distances],
        "start": starts,
    })

    tmp_path = output_path + ".tmp"
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, output_path)


def build(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"device={device}, use_fp16={args.use_fp16}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        print(torch.cuda.get_device_name(0), flush=True)

    if args.use_fp16 and device.type != "cuda":
        raise ValueError("--use_fp16 requires --device cuda")

    input_files = list_parquet_files(args.input_dir)
    if not input_files:
        raise ValueError(f"No parquet files found in {args.input_dir}")

    db_x_norm = load_database_x(
        args.retrieval_database_path,
        context_length=args.context_length,
        norm_method=args.rawx_norm,
    )

    processed = 0
    first_log = {"db": True}

    for input_file in input_files:
        if args.max_rows is not None and processed >= args.max_rows:
            break

        output_path = os.path.join(args.output_dir, os.path.basename(input_file))
        if os.path.exists(output_path):
            print(f"Skipping existing shard: {output_path}", flush=True)
            continue

        print(f"Reading query shard: {input_file}", flush=True)
        table = pq.read_table(input_file, columns=["target", "start"])

        targets_np = list_column_to_2d_numpy(table["target"])
        starts = table["start"].to_pylist()

        if args.max_rows is not None:
            remain = args.max_rows - processed
            targets_np = targets_np[:remain]
            starts = starts[:remain]

        n = len(targets_np)
        if n == 0:
            continue

        print(f"Processing shard {input_file}, rows={n}", flush=True)

        all_indices = []
        all_distances = []

        pbar = tqdm(total=n, desc=f"Queries {os.path.basename(input_file)}", unit="query")

        for begin in range(0, n, args.query_batch_size):
            end = min(begin + args.query_batch_size, n)

            idx, dist = compute_topk_for_query_batch(
                query_targets=targets_np[begin:end],
                db_x_norm=db_x_norm,
                context_length=args.context_length,
                top_k=args.top_k_buffer,
                norm_method=args.rawx_norm,
                db_batch_size=args.db_batch_size,
                device=device,
                use_fp16=args.use_fp16,
                first_log=first_log,
            )

            all_indices.append(idx)
            all_distances.append(dist)
            pbar.update(end - begin)

        pbar.close()

        indices = np.concatenate(all_indices, axis=0)
        distances = np.concatenate(all_distances, axis=0)

        write_output(
            output_path=output_path,
            targets=[row.tolist() for row in targets_np],
            starts=starts,
            indices=indices,
            distances=distances,
        )

        processed += n
        print(f"Saved: {output_path}", flush=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--retrieval_database_path", required=True)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--top_k_buffer", type=int, default=20)
    parser.add_argument("--rawx_norm", choices=["zscore", "minmax"], default="zscore")
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--db_batch_size", type=int, default=4096)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use_fp16", action="store_true")
    args = parser.parse_args()

    build(args)


if __name__ == "__main__":
    main()