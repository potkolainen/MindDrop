#!/usr/bin/env python3
"""
Local image generation using diffusers library.
Supports Stable Diffusion models from Hugging Face.
"""
import sys
import json
import os
from pathlib import Path

def generate_image(args):
    """Generate image using local Stable Diffusion model"""
    try:
        from diffusers import (
            StableDiffusionPipeline, 
            StableDiffusionXLPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionXLImg2ImgPipeline,
            DPMSolverMultistepScheduler
        )
        import torch
        import json
        from datetime import datetime
        from PIL import Image
        
        # Parse arguments
        prompt = args.get('prompt', '')
        negative_prompt = args.get('negative_prompt')
        width = args.get('width', 512)
        height = args.get('height', 512)
        steps = args.get('steps', 20)
        cfg_scale = args.get('cfg_scale', 7.5)
        seed = args.get('seed', -1)
        model_id = args.get('model_id')
        reference_image_path = args.get('reference_image')
        
        # Auto-round dimensions to nearest multiple of 8 (SD requirement)
        # This allows users to enter any size in the UI
        width = round(width / 8) * 8
        height = round(height / 8) * 8
        
        # Use a good default SD model if none specified
        if not model_id or model_id == 'null':
            model_id = 'runwayml/stable-diffusion-v1-5'
        
        # Validate model_id looks like a text-to-image model
        # Common SD models: stable-diffusion, sdxl, sd-v1, sd-v2
        model_lower = model_id.lower()
        if not any(keyword in model_lower for keyword in ['stable-diffusion', 'sdxl', 'sd-v1', 'sd-v2', 'sd_', 'sd-']):
            return {
                'success': False,
                'error': f'Model "{model_id}" does not appear to be a Stable Diffusion text-to-image model.\\n\\n' +
                         'Recommended models:\\n' +
                         '  - runwayml/stable-diffusion-v1-5 (classic, well-tested)\\n' +
                         '  - stabilityai/stable-diffusion-2-1-base (improved SD2)\\n' +
                         '  - stabilityai/sdxl-turbo (fast SDXL variant)\\n\\n' +
                         'Visit Models tab to download a Stable Diffusion model.'
            }
        
        # Set random seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)
        
        # Load model from local cache only (no automatic downloads)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        # Check HuggingFace cache locations
        # Helper function to find the actual model directory in HF cache structure
        def find_model_snapshot(base_path):
            """Find the snapshot directory containing model files in HF cache structure"""
            try:
                if not base_path or not base_path.exists():
                    return None
                
                # Check if model files are directly in base_path
                if (base_path / 'model_index.json').exists():
                    return base_path
                
                # Check for nested HuggingFace cache structure: models--org--model/snapshots/hash/
                models_dir = base_path / f"models--{model_id.replace('/', '--')}"
                if models_dir.exists():
                    snapshots_dir = models_dir / 'snapshots'
                    if snapshots_dir.exists():
                        # Get the first (and usually only) snapshot
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
            # AI Workspace downloads folder
            Path(__file__).parent / 'Downloads' / model_id.replace('/', '--'),
            # HuggingFace default cache
            Path.home() / '.cache' / 'huggingface' / 'hub' / f"models--{model_id.replace('/', '--')}",
            # Custom cache location
            Path.home() / '.cache' / 'ai-workspace' / 'image-models' / model_id.replace('/', '--'),
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
                         f'Recommended Stable Diffusion models:\n' +
                         f'  - runwayml/stable-diffusion-v1-5\n' +
                         f'  - stabilityai/stable-diffusion-2-1-base\n' +
                         f'  - stabilityai/sdxl-turbo'
            }
        
        # Auto-detect pipeline type from model_index.json
        # Use img2img pipeline if reference image is provided
        model_index_path = model_path / 'model_index.json'
        is_xl_model = False
        
        if model_index_path.exists():
            try:
                model_index = json.loads(model_index_path.read_text())
                class_name = model_index.get('_class_name', '')
                if 'XL' in class_name or 'xl' in class_name.lower():
                    is_xl_model = True
            except Exception:
                pass  # Use default if we can't read the file
        
        # Select appropriate pipeline class based on model type and whether we have a reference image
        if reference_image_path:
            pipeline_class = StableDiffusionXLImg2ImgPipeline if is_xl_model else StableDiffusionImg2ImgPipeline
        else:
            pipeline_class = StableDiffusionXLPipeline if is_xl_model else StableDiffusionPipeline
        
        pipe = pipeline_class.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True
        )
        
        # Memory optimizations for large images and VRAM constraints
        if device == 'cuda':
            # Don't move to device yet - let CPU offload handle it
            # Enable sequential CPU offloading - moves model layers to CPU when not in use
            # This dramatically reduces VRAM usage at the cost of some speed
            pipe.enable_model_cpu_offload()
            
            # Enable attention slicing - reduces memory usage significantly
            pipe.enable_attention_slicing(slice_size='auto')
            
            # Enable VAE slicing - helps with large resolution images
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
            
            # Set PyTorch memory allocator for better fragmentation handling
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Clear CUDA cache before generation
            torch.cuda.empty_cache()
        else:
            pipe = pipe.to(device)
        
        # Use DPM++ scheduler (similar to euler_a)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Generate image
        if reference_image_path:
            # Load and preprocess reference image for img2img
            init_image = Image.open(reference_image_path).convert('RGB')
            # Resize to match target dimensions
            init_image = init_image.resize((width, height), Image.LANCZOS)
            
            # For img2img, we use strength parameter (0.0-1.0)
            # Lower strength = more faithful to original, higher = more creative
            # Default to 0.75 for a good balance
            strength = args.get('strength', 0.75)
            
            image = pipe(
                prompt=prompt,
                image=init_image,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                strength=strength,
                generator=generator
            ).images[0]
        else:
            # Text-to-image generation
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            ).images[0]
        
        # Save image to generated_content/images/ folder
        # Use workspace root directory (parent of src-tauri)
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'generated_content' / 'images'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filename: YYYY-MM-DD_HH-MM-SS.png
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_path = output_dir / f'{timestamp}.png'
        image.save(str(output_path))
        
        # Clean up to free VRAM for next generation
        del pipe
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'image_path': str(output_path),
            'seed': seed
        }
        
    except ImportError as e:
        return {
            'success': False,
            'error': f'Missing required library: {str(e)}\n\nInstall with: pip install diffusers transformers accelerate torch'
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
    
    # Generate image
    result = generate_image(args)
    
    # Output JSON result
    print(json.dumps(result))
