"""
Streaming parallel WebDataset converter. Writes shards as episodes complete.
Low memory usage - doesn't hold all samples in RAM.
"""
import argparse, json, os, io, sys
import cv2
import h5py
import numpy as np
import torch
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def convert_one_episode(args_tuple):
    episode_id, hdf5_path, video_paths, target_size = args_tuple
    samples = []
    try:
        with h5py.File(hdf5_path, 'r') as f:
            actions = f['action'][:]
            qpos = f['observations/qpos'][:]

        caps = [cv2.VideoCapture(vp) for vp in video_paths]
        if not all(c.isOpened() for c in caps):
            return episode_id, [], 0

        num_timesteps = len(actions)
        for t in range(num_timesteps):
            sample_key = f'episode_{episode_id:04d}_{t:04d}'
            parts = {}
            for ci, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
                tensor = torch.from_numpy(frame)
                buf = io.BytesIO()
                torch.save(tensor, buf)
                parts[f'cam{ci+1}.pth'] = buf.getvalue()
            if len(parts) < 2:
                break
            qpos_t = torch.from_numpy(qpos[t].astype(np.float16))
            buf = io.BytesIO()
            torch.save(qpos_t, buf)
            parts['qpos.pth'] = buf.getvalue()
            actions_future = torch.from_numpy(actions[t:].astype(np.float16))
            buf = io.BytesIO()
            torch.save(actions_future, buf)
            parts['actions.pth'] = buf.getvalue()
            parts['key'] = sample_key
            samples.append(parts)
        for c in caps:
            c.release()
        return episode_id, samples, len(samples)
    except Exception as e:
        return episode_id, [], 0

def write_tar(samples, tar_path):
    with tarfile.open(tar_path, 'w') as tar:
        for sample in samples:
            key = sample['key']
            for fname, data in sample.items():
                if fname == 'key':
                    continue
                info = tarfile.TarInfo(name=f'{key}.{fname}')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--shard-size', type=int, default=500)
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    data_dir = args.data_dir
    webd_dir = os.path.join(data_dir, 'webdataset')
    if os.path.exists(webd_dir):
        import shutil
        shutil.rmtree(webd_dir)
    os.makedirs(webd_dir)

    for mf in ['dataset_metadata.json', 'metadata.json']:
        mp = os.path.join(data_dir, mf)
        if os.path.exists(mp):
            with open(mp) as f:
                metadata = json.load(f)
            break

    episodes = metadata.get('episodes', [])
    target_size = tuple(args.target_size)

    tasks = []
    for ep in episodes:
        ep_id = ep['episode_id']
        h5_path = os.path.join(data_dir, ep['file_name'])
        vids = [os.path.join(data_dir, vf) for vf in ep.get('video_files', [])[:2]]
        if len(vids) < 2 or not all(os.path.exists(v) for v in vids):
            continue
        tasks.append((ep_id, h5_path, vids, target_size))

    print(f'Processing {len(tasks)} episodes with {args.workers} workers (streaming)...')
    sys.stdout.flush()
    t0 = time.time()

    # Streaming: write shards as episodes complete
    current_shard = []
    shard_idx = 0
    total_samples = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one_episode, t): t[0] for t in tasks}
        for future in as_completed(futures):
            ep_id, samples, count = future.result()
            done += 1

            for s in samples:
                current_shard.append(s)
                if len(current_shard) >= args.shard_size:
                    tar_path = os.path.join(webd_dir, f'train-{shard_idx:05d}.tar')
                    write_tar(current_shard, tar_path)
                    shard_idx += 1
                    total_samples += len(current_shard)
                    current_shard = []

            if done % 20 == 0:
                elapsed = time.time() - t0
                rate = total_samples / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / (done / elapsed) if done > 0 else 0
                print(f'  [{done}/{len(tasks)}] {total_samples} samples, {shard_idx} shards, {rate:.0f} samples/s, ETA {eta:.0f}s')
                sys.stdout.flush()

    if current_shard:
        tar_path = os.path.join(webd_dir, f'train-{shard_idx:05d}.tar')
        write_tar(current_shard, tar_path)
        total_samples += len(current_shard)
        shard_idx += 1

    elapsed = time.time() - t0
    print(f'\nDone! {total_samples} samples in {shard_idx} shards')
    print(f'Time: {elapsed:.0f}s ({total_samples/elapsed:.0f} samples/s)')
    sys.stdout.flush()

    info = {'total_samples': total_samples, 'num_shards': shard_idx, 'samples_per_shard': args.shard_size}
    with open(os.path.join(webd_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == '__main__':
    main()
