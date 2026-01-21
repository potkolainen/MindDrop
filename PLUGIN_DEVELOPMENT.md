# Plugin Development Guide

## Overview

The AI Workspace Tauri plugin system is designed to be **adapter-based**, meaning plugins don't need to be custom-built for this application. Most existing CLI tools and scripts can work as plugins with minimal or no modification.

## Core Principles

1. **Language Agnostic**: Plugins can be written in any language (Python, Node.js, Rust, Go, etc.)
2. **Out-of-Process**: Plugins run in separate processes for security and stability
3. **JSON Communication**: All input/output uses JSON via stdin/stdout
4. **Minimal Contract**: Only a simple `plugin.json` manifest is required
5. **Existing Tools**: Designed to wrap existing tools without rewrites

## Plugin Structure

### Directory Layout

```
Plugins/
├── my-plugin/
│   ├── plugin.json        # Required: Plugin manifest
│   ├── main.py            # Entry point (example: Python)
│   ├── requirements.txt   # Optional: Dependencies
│   └── README.md          # Optional: Documentation
└── another-plugin/
    ├── plugin.json
    ├── index.js           # Entry point (example: Node.js)
    └── package.json
```

### Plugin Manifest (plugin.json)

The `plugin.json` file defines your plugin's metadata and contract:

```json
{
  "name": "My Plugin",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "A brief description of what this plugin does",
  "type": "python",
  "entry": "main.py",
  "inputs": ["text", "config"],
  "outputs": ["result", "metadata"],
  "permissions": ["network", "filesystem"]
}
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ Yes | Display name of the plugin |
| `version` | string | ❌ No | Semantic version (e.g., "1.0.0") |
| `author` | string | ❌ No | Plugin author |
| `description` | string | ❌ No | Brief description of functionality |
| `type` | string | ✅ Yes | Plugin type: `python`, `node`, `http`, or `executable` |
| `entry` | string | ✅ Yes | Entry point file (relative to plugin folder) |
| `inputs` | array | ❌ No | List of expected input fields |
| `outputs` | array | ❌ No | List of output fields |
| `permissions` | array | ❌ No | Requested permissions (currently informational) |

## Plugin Types & Adapters

### 1. Python CLI Plugin

**Type**: `python`

Python plugins receive JSON on stdin and output JSON on stdout.

**Example** (`main.py`):
```python
#!/usr/bin/env python3
import sys
import json

def main():
    # Read input from stdin
    input_data = json.load(sys.stdin)
    
    # Process the input
    text = input_data.get("text", "")
    result = text.upper()  # Example transformation
    
    # Output result as JSON
    output = {
        "result": result,
        "length": len(result)
    }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
```

**Manifest** (`plugin.json`):
```json
{
  "name": "Text Transformer",
  "version": "1.0.0",
  "description": "Transforms text to uppercase",
  "type": "python",
  "entry": "main.py",
  "inputs": ["text"],
  "outputs": ["result", "length"]
}
```

### 2. Node.js CLI Plugin

**Type**: `node`

Node.js plugins work similarly to Python plugins.

**Example** (`index.js`):
```javascript
#!/usr/bin/env node

const readline = require('readline');

// Read stdin
let inputData = '';
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  inputData += line;
});

rl.on('close', () => {
  try {
    const input = JSON.parse(inputData);
    
    // Process the input
    const result = {
      reversed: input.text.split('').reverse().join(''),
      original_length: input.text.length
    };
    
    // Output as JSON
    console.log(JSON.stringify(result));
  } catch (error) {
    console.error(JSON.stringify({ error: error.message }));
    process.exit(1);
  }
});
```

**Manifest** (`plugin.json`):
```json
{
  "name": "Text Reverser",
  "version": "1.0.0",
  "description": "Reverses input text",
  "type": "node",
  "entry": "index.js",
  "inputs": ["text"],
  "outputs": ["reversed", "original_length"]
}
```

### 3. Binary/Executable Plugin

**Type**: `executable`

Any compiled binary or executable script that accepts JSON on stdin and outputs JSON on stdout.

**Example** (Rust binary):
```rust
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Deserialize)]
struct Input {
    numbers: Vec<i32>,
}

#[derive(Serialize)]
struct Output {
    sum: i32,
    average: f64,
    count: usize,
}

fn main() -> io::Result<()> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    
    let input: Input = serde_json::from_str(&buffer)?;
    
    let sum: i32 = input.numbers.iter().sum();
    let count = input.numbers.len();
    let average = sum as f64 / count as f64;
    
    let output = Output { sum, average, count };
    println!("{}", serde_json::to_string(&output)?);
    
    Ok(())
}
```

**Manifest** (`plugin.json`):
```json
{
  "name": "Number Statistics",
  "version": "1.0.0",
  "description": "Calculates sum and average of numbers",
  "type": "executable",
  "entry": "number_stats",
  "inputs": ["numbers"],
  "outputs": ["sum", "average", "count"]
}
```

### 4. HTTP/REST Plugin

**Type**: `http`

*(Future implementation - not yet supported)*

For plugins that expose HTTP endpoints.

## Wrapping Existing Tools

One of the key design goals is to wrap existing third-party tools without modification.

### Example: Wrapping `jq` (JSON processor)

**Wrapper** (`jq_wrapper.py`):
```python
#!/usr/bin/env python3
import sys
import json
import subprocess

def main():
    input_data = json.load(sys.stdin)
    
    # Extract the filter and data
    filter_expr = input_data.get("filter", ".")
    data = input_data.get("data", {})
    
    # Run jq as subprocess
    process = subprocess.Popen(
        ["jq", filter_expr],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    stdout, stderr = process.communicate(input=json.dumps(data).encode())
    
    if process.returncode == 0:
        result = json.loads(stdout.decode())
        print(json.dumps({"result": result}))
    else:
        print(json.dumps({"error": stderr.decode()}))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Manifest** (`plugin.json`):
```json
{
  "name": "jq JSON Processor",
  "version": "1.0.0",
  "description": "Wrapper for the jq command-line JSON processor",
  "type": "python",
  "entry": "jq_wrapper.py",
  "inputs": ["filter", "data"],
  "outputs": ["result"]
}
```

## Input/Output Contract

### Input Format

Plugins receive JSON via stdin in this format:

```json
{
  "text": "some text",
  "config": {
    "option1": true,
    "option2": "value"
  }
}
```

### Output Format

Plugins should output JSON to stdout:

**Success**:
```json
{
  "result": "processed output",
  "metadata": {
    "processing_time": 0.123
  }
}
```

**Error**:
```json
{
  "error": "Error message describing what went wrong"
}
```

## Error Handling

Plugins should:

1. **Exit with non-zero code on errors**
2. **Output error messages to stderr OR as JSON with `error` field**
3. **Validate input before processing**

Example:
```python
try:
    input_data = json.load(sys.stdin)
    if "required_field" not in input_data:
        raise ValueError("Missing required_field")
    # ... process ...
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
```

## Security Considerations

### Current Limitations

The current plugin system has **minimal security sandboxing**. Plugins run with the same permissions as the main application.

### Best Practices

1. **Only install trusted plugins**
2. **Review plugin code before enabling**
3. **Disable plugins you don't use**
4. **Use virtual environments for Python plugins**
5. **Limit network access when possible**

### Future Improvements

Planned security features:
- Process isolation
- Permission system (network, filesystem, execution)
- Resource limits (CPU, memory, time)
- Signature verification

## Testing Your Plugin

### Manual Testing

1. **Create test input** (`test_input.json`):
```json
{
  "text": "hello world"
}
```

2. **Test from command line**:
```bash
# Python plugin
cat test_input.json | python3 main.py

# Node.js plugin
cat test_input.json | node index.js

# Binary plugin
cat test_input.json | ./my_binary
```

3. **Verify output is valid JSON**:
```bash
cat test_input.json | python3 main.py | jq .
```

### Integration Testing

1. Place plugin in `Plugins/` directory
2. Open the Plugins tab in the application
3. Click "Refresh" to discover the plugin
4. Toggle the plugin to enable it
5. Test execution through the application UI

## Example Plugins

### 1. Text Summarizer (Python + Transformers)

```python
#!/usr/bin/env python3
import sys
import json
from transformers import pipeline

# Initialize summarization pipeline (cached after first run)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def main():
    input_data = json.load(sys.stdin)
    text = input_data.get("text", "")
    
    if not text:
        print(json.dumps({"error": "No text provided"}))
        sys.exit(1)
    
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    
    output = {
        "summary": summary[0]["summary_text"],
        "original_length": len(text),
        "summary_length": len(summary[0]["summary_text"])
    }
    
    print(json.dumps(output))

if __name__ == "__main__":
    main()
```

### 2. Code Formatter (Node.js + Prettier)

```javascript
#!/usr/bin/env node
const prettier = require('prettier');
const readline = require('readline');

let inputData = '';
const rl = readline.createInterface({
  input: process.stdin,
  terminal: false
});

rl.on('line', (line) => { inputData += line; });

rl.on('close', async () => {
  try {
    const input = JSON.parse(inputData);
    const code = input.code || '';
    const language = input.language || 'javascript';
    
    const formatted = await prettier.format(code, {
      parser: language,
      semi: true,
      singleQuote: true,
    });
    
    console.log(JSON.stringify({
      formatted_code: formatted,
      changed: formatted !== code
    }));
  } catch (error) {
    console.error(JSON.stringify({ error: error.message }));
    process.exit(1);
  }
});
```

### 3. Image Metadata Reader (Python + PIL)

```python
#!/usr/bin/env python3
import sys
import json
from PIL import Image

def main():
    input_data = json.load(sys.stdin)
    image_path = input_data.get("image_path", "")
    
    if not image_path:
        print(json.dumps({"error": "No image_path provided"}))
        sys.exit(1)
    
    try:
        img = Image.open(image_path)
        
        output = {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": img.tell()
        }
        
        # Extract EXIF data if available
        if hasattr(img, '_getexif') and img._getexif():
            output["has_exif"] = True
        
        print(json.dumps(output))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Plugin Not Discovered

- Ensure `plugin.json` is in the plugin folder
- Check JSON syntax is valid
- Click "Refresh" in the Plugins tab

### Plugin Won't Execute

- Verify the entry file exists and is executable
- Check file permissions (`chmod +x main.py`)
- Test manually from command line
- Check error messages in application logs

### Invalid Output

- Ensure output is valid JSON
- Use `json.dumps()` (Python) or `JSON.stringify()` (Node.js)
- Don't print debug messages to stdout (use stderr instead)

### Dependencies Not Found

- For Python: Install in virtual environment or system-wide
- For Node.js: Run `npm install` in plugin directory
- For binaries: Ensure all shared libraries are available

## Contributing

Have ideas for improving the plugin system? Open an issue or submit a PR!

### Roadmap

- [ ] HTTP/REST adapter
- [ ] WebSocket adapter for real-time plugins
- [ ] Plugin marketplace/registry
- [ ] Automatic dependency installation
- [ ] Plugin update system
- [ ] Enhanced security sandboxing
- [ ] Plugin composition (chaining plugins)

## License

Plugins are independent of the main application and can use any license. Please include a LICENSE file in your plugin folder.

---

**Questions?** Open an issue on GitHub or check the main README for support information.
