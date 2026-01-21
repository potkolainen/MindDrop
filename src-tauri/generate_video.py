#!/usr/bin/env python3
"""
Local video generation using diffusers library.
Supports Stable Video Diffusion (SVD) img2vid models.
"""
import sys
import json
import os
from pathlib import Path

def generate_video(args):
    """Generate video from image using Stable Video Diffusion"""
    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video
        import torch
        from datetime import datetime
        
        # Parse arguments
        reference_image_path = args.get('reference_image')
        width = args.get('width', 1024)
        height = args.get('height', 576)
        num_frames = args.get('num_frames', 14)  # SVD generates 14 or 25 frames
        fps = args.get('fps', 7)
        motion_bucket_id = args.get('motion_bucket_id', 127)  # Controls motion intensity (1-255)
        noise_aug_strength = args.get('noise_aug_strength', 0.02)  # Image conditioning noise
        seed = args.get('seed', -1)
        model_id = args.get('model_id')
        
        # Clamp num_frames to valid values
        if num_frames > 25:
            num_frames = 25
        elif num_frames < 14:
            num_frames = 14
        elif num_frames > 14 and num_frames < 25:
            num_frames = 14  # Default to 14 if between valid values
        
        # Round dimensions to multiples of 64 for better memory usage
        width = round(width / 64) * 64
        height = round(height / 64) * 64
        # Clamp to reasonable range
        width = max(256, min(1024, width))
        height = max(256, min(1024, height))
        
        if not reference_image_path:
            return {
                'success': False,
                'error': 'Reference image is required for video generation.\n\nUpload an image to animate it into a video.'
            }
        
        # Use default SVD model if none specified
        if not model_id or model_id == 'null':
            model_id = 'stabilityai/stable-video-diffusion-img2vid-xt'
        
        # Validate model looks like a video diffusion model
        model_lower = model_id.lower()
        if not any(keyword in model_lower for keyword in ['video', 'svd', 'img2vid']):
            return {
                'success': False,
                'error': f'Model "{model_id}" does not appear to be a video generation model.\n\n' +
                         'Recommended models:\n' +
                         '  - stabilityai/stable-video-diffusion-img2vid-xt (high quality, 25 frames)\n' +
                         '  - stabilityai/stable-video-diffusion-img2vid (standard, 14 frames)'
            }
        
        # Set random seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)
        
        # Load model from local cache only
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        # Helper function to find model in HF cache structure
        def find_model_snapshot(base_path):
            try:
                if not base_path or not base_path.exists():
                    return None
                
                if (base_path / 'model_index.json').exists():
                    return base_path
                
                models_dir = base_path / f"models--{model_id.replace('/', '--')}"
                if models_dir.exists():
                    snapshots_dir = models_dir / 'snapshots'
                    if snapshots_dir.exists():
                        try:
                            snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                            if snapshots:
                                snapshot = snapshots[0]
                                if (snapshot / 'model_index.json').exists():
                                    return snapshot
                        except Exception:
                            pass
                
                return None
            except Exception:
                return None
        
        cache_locations = [
            Path(__file__).parent / 'Downloads' / model_id.replace('/', '--'),
            Path.home() / '.cache' / 'huggingface' / 'hub' / f"models--{model_id.replace('/', '--')}",
            Path.home() / '.cache' / 'ai-workspace' / 'video-models' / model_id.replace('/', '--'),
        ]
        
        model_path = None
        for location in cache_locations:
            found = find_model_snapshot(location)
            if found:
                model_path = found
                break
        
        if not model_path:
            available_locations = '\n  - '.join(str(loc) for loc in cache_locations)
            return {
                'success': False,
                'error': f'Model "{model_id}" not found locally.\n\n' +
                         f'Searched in:\n  - {available_locations}\n\n' +
                         f'To download this model:\n' +
                         f'1. Go to the Library tab\n' +
                         f'2. Search for "{model_id}"\n' +
                         f'3. Click Download\n\n' +
                         f'Recommended Video Diffusion models:\n' +
                         f'  - stabilityai/stable-video-diffusion-img2vid-xt\n' +
                         f'  - stabilityai/stable-video-diffusion-img2vid'
            }
        
        # Load the pipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True
        )
        
        # Memory optimizations - CRITICAL for large models
        if device == 'cuda':
            # Enable sequential CPU offloading - moves layers to CPU when not in use
            pipe.enable_model_cpu_offload()
            
            # Enable VAE tiling for lower memory usage
            if hasattr(pipe, 'enable_vae_tiling'):
                pipe.enable_vae_tiling()
            
            # Enable VAE slicing
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
            
            # Set environment variable for better memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Clear cache before generation
            torch.cuda.empty_cache()
        else:
            pipe = pipe.to(device)
        
        # Load and preprocess reference image
        image = load_image(reference_image_path)
        # Resize to user-specified dimensions
        image = image.resize((width, height))
        
        # For very low memory, reduce decode chunk size
        decode_chunk = 4 if num_frames == 25 else 8
        
        # Generate video frames
        frames = pipe(
            image=image,
            width=width,
            height=height,
            num_frames=num_frames,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            generator=generator,
            decode_chunk_size=decode_chunk  # Smaller chunks = less memory
        ).frames[0]
        
        # Save video to generated_content/videos/ folder
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'generated_content' / 'videos'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: YYYY-MM-DD_HH-MM-SS.mp4
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_path = output_dir / f'{timestamp}.mp4'
        
        export_to_video(frames, str(output_path), fps=fps)
        
        # Clean up
        del pipe
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'video_path': str(output_path),
            'seed': seed,
            'num_frames': len(frames),
            'fps': fps
        }
        
    except ImportError as e:
        return {
            'success': False,
            'error': f'Missing required library: {str(e)}\n\nInstall with: pip install diffusers transformers accelerate torch opencv-python'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    # Read JSON arguments from stdin
    input_data = sys.stdin.read()
    args = json.loads(input_data)
    
    # Generate video
    result = generate_video(args)
    
    # Output JSON result
    print(json.dumps(result))
