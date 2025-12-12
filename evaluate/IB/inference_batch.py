#!/usr/bin/env python3
"""
ImageBind Audio-Video Consistency Batch Inference Script

Specially designed for testing audio and video consistency/matching.

Supports two modes:
1. Pairing mode: audio and video files are paired one-to-one (same filename or specified pairing file)
2. Retrieval mode: computes similarity matrix for all audio-video pairs

Usage examples:
1. Pairing mode (one-to-one files):
   python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs

2. Retrieval mode (compute all pairs):
   python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs --retrieval_mode

3. Read from file list:
   python batch_inference.py --audio_file audio_list.txt --video_file video_list.txt --output_dir ./outputs
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def load_path_list(path_file: str) -> List[str]:
    """Read path list from file"""
    with open(path_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def get_files_from_dir(directory: str, extensions: List[str]) -> List[str]:
    """Get all files with specified extensions from directory"""
    directory = Path(directory)
    files = []
    for ext in extensions:
        files.extend(list(directory.glob(f"**/*{ext}")))
    return sorted([str(f) for f in files])


def match_audio_video_files(audio_paths: List[str], video_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Match audio and video files by filename

    Args:
        audio_paths: list of audio file paths
        video_paths: list of video file paths

    Returns:
        List of matched (audio_path, video_path) pairs
    """
    pairs = []
    audio_dict = {Path(p).stem: p for p in audio_paths}
    video_dict = {Path(p).stem: p for p in video_paths}

    # Find common filenames (without extension)
    common_stems = set(audio_dict.keys()) & set(video_dict.keys())

    for stem in sorted(common_stems):
        pairs.append((audio_dict[stem], video_dict[stem]))

    print(f"Found {len(pairs)} matched audio-video file pairs")
    if len(pairs) < len(audio_paths) or len(pairs) < len(video_paths):
        print(f"Warning: {len(audio_paths) - len(pairs)} audio files not matched")
        print(f"Warning: {len(video_paths) - len(pairs)} video files not matched")

    return pairs


def compute_consistency_metrics(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    pairs: Optional[List[Tuple[str, str]]] = None
) -> Dict:
    """
    Compute audio-video consistency metrics

    Args:
        audio_embeddings: audio embedding vectors (N, 1024)
        video_embeddings: video embedding vectors (M, 1024)
        pairs: pairing list, if None computes all pairs

    Returns:
        Dictionary containing various metrics
    """
    metrics = {}

    # Normalize embeddings (for cosine similarity)
    audio_norm = torch.nn.functional.normalize(audio_embeddings, p=2, dim=1)
    video_norm = torch.nn.functional.normalize(video_embeddings, p=2, dim=1)

    # Compute similarity matrix
    similarity_matrix = audio_norm @ video_norm.T  # (N, M)

    if pairs is not None and len(audio_embeddings) == len(video_embeddings):
        # Pairing mode: compute paired consistency
        # In pairing mode, audio and video order is aligned, use diagonal elements
        paired_similarities = []
        n_pairs = min(len(audio_embeddings), len(video_embeddings))
        for i in range(n_pairs):
            paired_similarities.append(similarity_matrix[i, i].item())

        if paired_similarities:
            metrics['paired_similarity_mean'] = np.mean(paired_similarities)
            metrics['paired_similarity_std'] = np.std(paired_similarities)
            metrics['paired_similarity_min'] = np.min(paired_similarities)
            metrics['paired_similarity_max'] = np.max(paired_similarities)
            metrics['paired_similarities'] = paired_similarities

    # Retrieval mode: compute retrieval metrics
    # For each audio, find the rank of the most similar video
    # In pairing mode, compute retrieval performance (rank needed to find correct pair)
    if pairs is not None and len(audio_embeddings) == len(video_embeddings):
        retrieval_ranks = []
        for i in range(len(audio_embeddings)):
            similarities = similarity_matrix[i].cpu().numpy()
            # In pairing mode, correct pair should be the i-th video
            correct_idx = i
            sorted_indices = np.argsort(similarities)[::-1]  # descending order
            rank = np.where(sorted_indices == correct_idx)[0][0] + 1  # rank starts from 1
            retrieval_ranks.append(rank)

        if retrieval_ranks:
            metrics['mean_rank'] = np.mean(retrieval_ranks)
            metrics['median_rank'] = np.median(retrieval_ranks)
            metrics['recall_at_1'] = np.mean([1 if r == 1 else 0 for r in retrieval_ranks])
            metrics['recall_at_5'] = np.mean([1 if r <= 5 else 0 for r in retrieval_ranks])
            metrics['recall_at_10'] = np.mean([1 if r <= 10 else 0 for r in retrieval_ranks])

    # Overall statistics
    metrics['similarity_matrix_mean'] = similarity_matrix.mean().item()
    metrics['similarity_matrix_std'] = similarity_matrix.std().item()
    metrics['similarity_matrix'] = similarity_matrix.cpu().numpy()

    return metrics


def save_results(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    audio_names: List[str],
    video_names: List[str],
    metrics: Dict,
    output_dir: str,
    pairs: Optional[List[Tuple[str, str]]] = None
):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings
    audio_np = audio_embeddings.cpu().numpy()
    video_np = video_embeddings.cpu().numpy()

    np.save(os.path.join(output_dir, "audio_embeddings.npy"), audio_np)
    np.save(os.path.join(output_dir, "video_embeddings.npy"), video_np)
    print(f"\nEmbeddings saved:")
    print(f"  Audio: {audio_np.shape}")
    print(f"  Video: {video_np.shape}")

    # Save file name lists
    with open(os.path.join(output_dir, "audio_names.txt"), 'w', encoding='utf-8') as f:
        for name in audio_names:
            f.write(f"{name}\n")

    with open(os.path.join(output_dir, "video_names.txt"), 'w', encoding='utf-8') as f:
        for name in video_names:
            f.write(f"{name}\n")

    # Save pairing info
    if pairs:
        with open(os.path.join(output_dir, "pairs.txt"), 'w', encoding='utf-8') as f:
            for audio_path, video_path in pairs:
                f.write(f"{audio_path}\t{video_path}\n")

    # Save similarity matrix
    similarity_matrix = metrics['similarity_matrix']
    np.save(os.path.join(output_dir, "similarity_matrix.npy"), similarity_matrix)

    # Save metrics
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'similarity_matrix'}
    with open(os.path.join(output_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

    # Save readable report
    with open(os.path.join(output_dir, "consistency_report.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Audio-Video Consistency Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of audio files: {len(audio_names)}\n")
        f.write(f"Number of video files: {len(video_names)}\n")
        if pairs:
            f.write(f"Number of pairs: {len(pairs)}\n")
        f.write("\n")

        f.write("Similarity statistics:\n")
        f.write(f"  Mean similarity: {metrics['similarity_matrix_mean']:.4f}\n")
        f.write(f"  Std: {metrics['similarity_matrix_std']:.4f}\n")
        f.write("\n")

        if 'paired_similarity_mean' in metrics:
            f.write("Paired consistency:\n")
            f.write(f"  Mean paired similarity: {metrics['paired_similarity_mean']:.4f}\n")
            f.write(f"  Std: {metrics['paired_similarity_std']:.4f}\n")
            f.write(f"  Min: {metrics['paired_similarity_min']:.4f}\n")
            f.write(f"  Max: {metrics['paired_similarity_max']:.4f}\n")
            f.write("\n")

        if 'mean_rank' in metrics:
            f.write("Retrieval performance:\n")
            f.write(f"  Mean rank: {metrics['mean_rank']:.2f}\n")
            f.write(f"  Median rank: {metrics['median_rank']:.2f}\n")
            f.write(f"  Recall@1: {metrics['recall_at_1']:.4f}\n")
            f.write(f"  Recall@5: {metrics['recall_at_5']:.4f}\n")
            f.write(f"  Recall@10: {metrics['recall_at_10']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Similarity Matrix (first 10x10):\n")
        f.write("=" * 80 + "\n")
        matrix_preview = similarity_matrix[:10, :10]
        np.savetxt(f, matrix_preview, fmt='%.4f', delimiter='\t')

    print(f"\nResults saved to: {output_dir}")
    print(f"  - Embeddings: audio_embeddings.npy, video_embeddings.npy")
    print(f"  - Similarity matrix: similarity_matrix.npy")
    print(f"  - Metrics: metrics.json")
    print(f"  - Consistency report: consistency_report.txt")


def batch_inference_av_consistency(
    audio_paths: List[str],
    video_paths: List[str],
    device: str = "cuda:0",
    batch_size: int = 4,
    output_dir: Optional[str] = None,
    retrieval_mode: bool = False,
):
    """
    Audio-video consistency batch inference

    Args:
        audio_paths: list of audio file paths
        video_paths: list of video file paths
        device: compute device
        batch_size: batch size (video processing needs smaller batch)
        output_dir: output directory
        retrieval_mode: whether retrieval mode (False=pairing mode, True=retrieval mode)
    """
    # Check device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"

    # Load model
    print(f"Loading ImageBind model (device: {device})...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("Model loaded!")

    # Match audio-video files
    pairs = None
    if not retrieval_mode:
        pairs = match_audio_video_files(audio_paths, video_paths)
        if not pairs:
            print("Warning: No matched audio-video pairs found, switching to retrieval mode")
            retrieval_mode = True
        else:
            # Use paired files
            audio_paths = [p[0] for p in pairs]
            video_paths = [p[1] for p in pairs]

    # Process audio
    print(f"\nProcessing {len(audio_paths)} audio files...")
    valid_audio_paths = [p for p in audio_paths if os.path.exists(p)]
    if len(valid_audio_paths) != len(audio_paths):
        print(f"Warning: {len(audio_paths) - len(valid_audio_paths)} audio files do not exist")

    audio_embeddings_all = []
    audio_names = []
    for i in tqdm(range(0, len(valid_audio_paths), batch_size), desc="Audio batch"):
        batch_paths = valid_audio_paths[i:i+batch_size]
        try:
            batch_inputs = data.load_and_transform_audio_data(batch_paths, device)

            with torch.no_grad():
                batch_outputs = model({ModalityType.AUDIO: batch_inputs})
                audio_embeddings_all.append(batch_outputs[ModalityType.AUDIO])

            audio_names.extend([Path(p).stem for p in batch_paths])
        except Exception as e:
            print(f"Error processing audio batch {i//batch_size + 1}: {e}")
            continue

    if not audio_embeddings_all:
        print("Error: No audio files processed successfully")
        return

    audio_embeddings = torch.cat(audio_embeddings_all, dim=0)
    print(f"Audio embedding shape: {audio_embeddings.shape}")

    # Process video
    print(f"\nProcessing {len(video_paths)} video files...")
    valid_video_paths = [p for p in video_paths if os.path.exists(p)]
    if len(valid_video_paths) != len(video_paths):
        print(f"Warning: {len(video_paths) - len(valid_video_paths)} video files do not exist")

    video_embeddings_all = []
    video_names = []
    for i in tqdm(range(0, len(valid_video_paths), batch_size), desc="Video batch"):
        batch_paths = valid_video_paths[i:i+batch_size]
        try:
            batch_inputs = data.load_and_transform_video_data(batch_paths, device)

            with torch.no_grad():
                batch_outputs = model({ModalityType.VISION: batch_inputs})
                video_embeddings_all.append(batch_outputs[ModalityType.VISION])

            video_names.extend([Path(p).stem for p in batch_paths])
        except Exception as e:
            print(f"Error processing video batch {i//batch_size + 1}: {e}")
            continue

    if not video_embeddings_all:
        print("Error: No video files processed successfully")
        return

    video_embeddings = torch.cat(video_embeddings_all, dim=0)
    print(f"Video embedding shape: {video_embeddings.shape}")

    # Compute consistency metrics
    print("\nComputing audio-video consistency metrics...")
    metrics = compute_consistency_metrics(
        audio_embeddings, video_embeddings, pairs if not retrieval_mode else None
    )

    # Print key metrics
    print("\n" + "=" * 80)
    print("Consistency Evaluation Results:")
    print("=" * 80)
    print(f"Mean similarity: {metrics['similarity_matrix_mean']:.4f} ± {metrics['similarity_matrix_std']:.4f}")

    if 'paired_similarity_mean' in metrics:
        print(f"\nPaired consistency:")
        print(f"  Mean paired similarity: {metrics['paired_similarity_mean']:.4f} ± {metrics['paired_similarity_std']:.4f}")
        print(f"  Range: [{metrics['paired_similarity_min']:.4f}, {metrics['paired_similarity_max']:.4f}]")

    if 'mean_rank' in metrics:
        print(f"\nRetrieval performance:")
        print(f"  Mean rank: {metrics['mean_rank']:.2f}")
        print(f"  Median rank: {metrics['median_rank']:.2f}")
        print(f"  Recall@1: {metrics['recall_at_1']:.4f}")
        print(f"  Recall@5: {metrics['recall_at_5']:.4f}")
        print(f"  Recall@10: {metrics['recall_at_10']:.4f}")

    # Save results
    if output_dir:
        save_results(
            audio_embeddings, video_embeddings,
            audio_names, video_names,
            metrics, output_dir, pairs if not retrieval_mode else None
        )

    return {
        'audio_embeddings': audio_embeddings,
        'video_embeddings': video_embeddings,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="ImageBind Audio-Video Consistency Batch Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Pairing mode (one-to-one files)
  python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs

  # Retrieval mode (compute all pairs)
  python batch_inference.py --audio_dir ./audios --video_dir ./videos --output_dir ./outputs --retrieval_mode

  # Read from file list
  python batch_inference.py --audio_file audio_list.txt --video_file video_list.txt --output_dir ./outputs
        """
    )

    # Input options
    parser.add_argument("--audio_dir", type=str, help="Audio directory path")
    parser.add_argument("--audio_file", type=str, help="Audio path list file, one path per line")
    parser.add_argument("--audio_paths", type=str, nargs="+", help="List of audio paths")
    parser.add_argument("--video_dir", type=str, help="Video directory path")
    parser.add_argument("--video_file", type=str, help="Video path list file, one path per line")
    parser.add_argument("--video_paths", type=str, nargs="+", help="List of video paths")

    # Processing options
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (default: cuda:0)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4, video processing recommends small batch)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--retrieval_mode", action="store_true", help="Retrieval mode: compute similarity for all audio-video pairs")

    args = parser.parse_args()

    # Parse audio input
    audio_paths = None
    if args.audio_paths:
        audio_paths = args.audio_paths
    elif args.audio_file:
        audio_paths = load_path_list(args.audio_file)
    elif args.audio_dir:
        audio_paths = get_files_from_dir(args.audio_dir, ['.wav', '.mp3', '.flac', '.m4a', '.ogg'])

    # Parse video input
    video_paths = None
    if args.video_paths:
        video_paths = args.video_paths
    elif args.video_file:
        video_paths = load_path_list(args.video_file)
    elif args.video_dir:
        video_paths = get_files_from_dir(args.video_dir, ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'])

    # Check input
    if not audio_paths:
        parser.print_help()
        print("\nError: Please provide audio input!")
        return

    if not video_paths:
        parser.print_help()
        print("\nError: Please provide video input!")
        return

    # Run batch inference
    results = batch_inference_av_consistency(
        audio_paths=audio_paths,
        video_paths=video_paths,
        device="cuda:2",
        batch_size=4,
        output_dir="./outputs",
    )

    if results:
        print("\nAudio-Video Consistency Evaluation Completed!")


if __name__ == "__main__":
    main()
