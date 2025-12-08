#!/usr/bin/env python3
"""
Main entry point for team classification system.
Provides command-line interface compatible with original script.
"""

import os
import sys
import cv2
import time
import argparse
import logging
from pathlib import Path

from config import TeamClassifierConfig
from video_processor import VideoProcessor, VideoWriter


def setup_logging(debug_mode: bool = False, quiet: bool = False):
    """
    Setup logging configuration.
    
    Args:
        debug_mode: Enable debug logging
        quiet: Suppress most output
    """
    if quiet:
        level = logging.ERROR
    elif debug_mode:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def process_video(video_path: str,
                 output_path: str = None,
                 display: bool = False,
                 save_video: bool = True,
                 sample_rate: int = 1,
                 max_frames: int = None,
                 verbose: bool = True,
                 debug_mode: bool = False) -> dict:
    """
    Process video with team classification.
    
    Args:
        video_path: Input video file path
        output_path: Output video file path
        display: Display results in real-time
        save_video: Save output video
        sample_rate: Frame sampling rate (1=every frame, 2=every 2nd frame)
        max_frames: Maximum frames to process
        verbose: Enable verbose output
        debug_mode: Enable debug mode (logging, block visualization)
        
    Returns:
        Processing statistics dictionary
    """
    # Validate input
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return {}
    
    # Print header
    if verbose:
        print("=" * 80)
        mode_str = "DEBUG MODE" if debug_mode else "RELEASE MODE"
        print(f"TEAM CLASSIFICATION SYSTEM - {mode_str}")
        print("=" * 80)
        print(f"Input Video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return {}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.2f} seconds")
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
        if verbose:
            print(f"   Processing: {total_frames} frames (limited)")
    
    if verbose:
        print(f"   Sample Rate: 1/{sample_rate}")
    
    # Setup output
    video_writer = None
    if save_video:
        if output_path is None:
            video_name = Path(video_path).stem
            output_dir = Path("video_clustering_results")
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"{video_name}_clustered.mp4")
        
        video_writer = VideoWriter(
            output_path, 
            fps // sample_rate, 
            width, 
            height,
            enable_logging=debug_mode
        )
        
        if verbose:
            print(f"Output Video: {output_path}")
    
    # Create configuration
    config = TeamClassifierConfig(
        team_colors={
            'home': (201, 201, 199),
            'away': (111, 85, 82),
            'home_gk': (80, 60, 200),
            'away_gk': (185, 185, 130)
        },
        group_limits={
            'home': 11,
            'away': 11,
            'home_gk': 1,
            'away_gk': 1
        },
        green_rgb_threshold=90,
        green_cosine_similarity_min=0.75,
        gk_max_distance=80,
        debug_mode=debug_mode,
        verbose=verbose
    )
    
    # Initialize processor
    if verbose:
        print(f"\nInitializing Video Processor...")
        print(f"   Team Colors: Home={config.team_colors['home']}, Away={config.team_colors['away']}")
        print(f"   GK Colors: Home={config.team_colors['home_gk']}, Away={config.team_colors['away_gk']}")
    
    processor = VideoProcessor(config=config)
    
    if verbose:
        print(f"\nStarting video processing...")
        print("=" * 80)
    
    # Processing loop
    start_time = time.time()
    frame_count = 0
    processed_count = 0
    last_print_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if max_frames and frame_count > max_frames:
                break
            
            if frame_count % sample_rate != 0:
                continue
            
            processed_count += 1
            
            # Progress output
            current_time = time.time()
            time_since_last_print = current_time - last_print_time
            print_interval = 2.0 if verbose else 5.0
            
            should_print = debug_mode and (
                processed_count == 1 or
                time_since_last_print >= print_interval or
                frame_count >= total_frames
            )
            
            if should_print:
                elapsed = current_time - start_time
                fps_current = processed_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                
                if verbose:
                    print(f"Progress: {progress:5.1f}% | Frame {frame_count}/{total_frames} | "
                          f"Processed: {processed_count} | FPS: {fps_current:.1f}")
                else:
                    bar_length = 30
                    filled = int(bar_length * progress / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r[{bar}] {progress:5.1f}% | FPS: {fps_current:.1f}", end='', flush=True)
                
                last_print_time = current_time
            
            # Process frame
            result = processor.process_frame(frame)
            processor.update_stats(result)
            
            # Handle result
            if result and result.get('success', False):
                visualization = result['visualization']
                
                # Save block filtering visualization in debug mode
                if debug_mode and result.get('block_filtering_vis') is not None:
                    block_vis_dir = Path(output_path).parent / "block_filtering_frames" \
                                   if output_path else Path("video_clustering_results/block_filtering_frames")
                    block_vis_dir.mkdir(parents=True, exist_ok=True)
                    block_vis_path = block_vis_dir / f"frame_{frame_count:05d}_blocks.jpg"
                    cv2.imwrite(str(block_vis_path), result['block_filtering_vis'])
                
                # Write to video
                if video_writer:
                    video_writer.write_frame(visualization, result, frame_count)
                
                # Display
                if display:
                    cv2.imshow('Team Classification', visualization)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠️  User interrupted processing")
                        break
            else:
                # Failed frame
                if video_writer:
                    video_writer.write_frame(frame)
                
                if display:
                    cv2.imshow('Team Classification', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser interrupted processing")
                        break
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
    
    # Print results
    elapsed_time = time.time() - start_time
    stats = processor.get_stats()
    team_stats = processor.get_team_statistics()
    
    if not debug_mode and not verbose:
        print()  # New line after progress bar
    
    print("\n" + "=" * 80)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 80)
    
    if verbose:
        print(f"Processing Statistics:")
        print(f"   Total Frames Read: {frame_count}")
        print(f"   Frames Processed: {processed_count}")
        print(f"   Successful: {stats['successful_frames']}")
        print(f"   Failed: {stats['failed_frames']}")
        if processed_count > 0:
            print(f"   Success Rate: {stats['successful_frames']/processed_count*100:.1f}%")
    else:
        if processed_count > 0:
            print(f"Processed: {processed_count} frames | "
                  f"Success Rate: {stats['successful_frames']/processed_count*100:.1f}%")
    
    print(f"Total Time: {elapsed_time:.2f} seconds | "
          f"Average FPS: {processed_count/elapsed_time:.2f}")
    
    if verbose and stats['successful_frames'] > 0:
        print(f"\nTeam Detection Summary:")
        for team, total in stats['team_totals'].items():
            avg_per_frame = total / stats['successful_frames']
            team_name = team.replace('_', ' ').upper()
            print(f"   {team_name}: {total} total detections | Avg {avg_per_frame:.1f} per frame")
    
    if save_video:
        print(f"Output saved to: {output_path}")
        if debug_mode:
            log_path = str(Path(output_path).with_suffix('.log'))
            print(f"Block count log saved to: {log_path}")
    
    print("=" * 80)
    
    return {
        'total_frames': frame_count,
        'processed_frames': processed_count,
        'successful_frames': stats['successful_frames'],
        'failed_frames': stats['failed_frames'],
        'elapsed_time': elapsed_time,
        'output_path': output_path if save_video else None,
        'team_totals': stats['team_totals']
    }


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Team Classification System - Refactored Modular Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python main.py video.mp4
  
  # Process with debug mode (logging + block visualization)
  python main.py video.mp4 --debug
  
  # Process every 2nd frame for faster processing
  python main.py video.mp4 --sample-rate 2
  
  # Process first 100 frames only
  python main.py video.mp4 --max-frames 100
  
  # Display real-time results
  python main.py video.mp4 --display
        """
    )
    
    parser.add_argument('video', type=str, help='Input video file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path (default: auto-generated)')
    parser.add_argument('--display', '-d', action='store_true',
                       help='Display results in real-time')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    parser.add_argument('--sample-rate', '-s', type=int, default=1,
                       help='Frame sampling rate (1=every frame, 2=every 2nd frame)')
    parser.add_argument('--max-frames', '-m', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - minimal output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - enables logging and block visualization')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(debug_mode=args.debug, quiet=args.quiet)
    
    # Process video
    results = process_video(
        video_path=args.video,
        output_path=args.output,
        display=args.display,
        save_video=not args.no_save,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        verbose=not args.quiet,
        debug_mode=args.debug
    )
    
    if not results:
        print("Video processing failed!")
        sys.exit(1)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
