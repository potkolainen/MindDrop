#!/usr/bin/env python3
"""
Example Text Transformer Plugin

This plugin demonstrates the basic plugin contract:
- Reads JSON from stdin
- Processes the input
- Outputs JSON to stdout
"""

import sys
import json


def transform_text(text, mode):
    """Transform text based on the specified mode."""
    if mode == "uppercase":
        return text.upper()
    elif mode == "lowercase":
        return text.lower()
    elif mode == "titlecase":
        return text.title()
    elif mode == "reverse":
        return text[::-1]
    else:
        return text


def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract parameters
        text = input_data.get("text", "")
        mode = input_data.get("mode", "uppercase")
        
        # Validate input
        if not text:
            raise ValueError("No text provided")
        
        # Process the text
        result = transform_text(text, mode)
        
        # Prepare output
        output = {
            "result": result,
            "length": len(result),
            "mode_used": mode,
            "success": True
        }
        
        # Output as JSON
        print(json.dumps(output))
        
    except Exception as e:
        # Error handling
        error_output = {
            "error": str(e),
            "success": False
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
