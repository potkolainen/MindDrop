# Text Transformer Plugin

An example plugin demonstrating the MindDrop plugin system.

## Features

- Transforms text to uppercase
- Transforms text to lowercase
- Transforms text to title case
- Reverses text

## Usage

### From Command Line

```bash
echo '{"text": "hello world", "mode": "uppercase"}' | python3 transform.py
```

Output:
```json
{
  "result": "HELLO WORLD",
  "length": 11,
  "mode_used": "uppercase",
  "success": true
}
```

### Modes

- `uppercase`: Converts text to UPPERCASE
- `lowercase`: Converts text to lowercase
- `titlecase`: Converts Text To Title Case
- `reverse`: Reverses the text

### Input Format

```json
{
  "text": "your text here",
  "mode": "uppercase"
}
```

### Output Format

```json
{
  "result": "TRANSFORMED TEXT",
  "length": 16,
  "mode_used": "uppercase",
  "success": true
}
```

## Testing

Test the plugin manually:

```bash
# Uppercase
echo '{"text": "hello", "mode": "uppercase"}' | python3 transform.py

# Lowercase
echo '{"text": "HELLO", "mode": "lowercase"}' | python3 transform.py

# Title case
echo '{"text": "hello world", "mode": "titlecase"}' | python3 transform.py

# Reverse
echo '{"text": "hello", "mode": "reverse"}' | python3 transform.py
```

## License

MIT
