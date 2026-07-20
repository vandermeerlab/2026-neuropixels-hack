import os
import re

def parse_expkeys(directory):
    """
    Find and parse an ExpKeys file from a directory.
    
    Args:
        directory (str): Path to the directory containing the ExpKeys file
        
    Returns:
        dict: Dictionary containing the parsed ExpKeys values
        
    Raises:
        ValueError: If no ExpKeys file is found or if multiple ExpKeys files are found
    """
    # Find all files ending with 'keys.m'
    keys_files = [f for f in os.listdir(directory) if f.endswith('keys.m')]
    
    # Check number of files found
    if len(keys_files) == 0:
        raise ValueError(f"No ExpKeys exist in {directory}")
    elif len(keys_files) > 1:
        raise ValueError(f"Multiple ExpKeys exist in {directory}")
        
    # Process the single found file
    filename = os.path.join(directory, keys_files[0])
    expkeys = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Skip commented lines (only if the comment starts at the beginning of the line)
        if line.strip().startswith('%'):
            continue
            
        # Look for lines containing 'ExpKeys.'
        if 'ExpKeys.' in line:
            # Split only on first occurrence of '='
            parts = line.split('=', 1)
            if len(parts) != 2:
                continue
                
            # Get key (remove 'ExpKeys.' and strip whitespace)
            key = parts[0].replace('ExpKeys.', '').strip()
            
            # Get value part and handle comments properly
            value_part = parts[1]
            
            # Use a more nuanced approach to handle comments
            # First, let's isolate any quoted strings
            def replace_quoted_strings(text):
                """Replace quoted strings with placeholders to protect their content"""
                placeholders = []
                
                # Function to replace a match with a placeholder
                def replace_match(match):
                    placeholder = f"PLACEHOLDER_{len(placeholders)}"
                    # Store the match with its quotes
                    placeholders.append(match.group(0))
                    return placeholder
                
                # Replace both single and double quoted strings
                pattern = r'(["\'])(.*?)(?<!\\)\1'
                modified_text = re.sub(pattern, replace_match, text, flags=re.DOTALL)
                
                return modified_text, placeholders
            
            # Replace quoted strings with placeholders
            modified_value, placeholders = replace_quoted_strings(value_part)
            
            # Now we can safely split on % for comments outside quotes
            if '%' in modified_value:
                modified_value = modified_value.split('%')[0]
            
            # Restore the original quoted strings
            for i, placeholder in enumerate(placeholders):
                modified_value = modified_value.replace(f"PLACEHOLDER_{i}", placeholder)
            
            value_part = modified_value
            
            # Remove trailing semicolon and whitespace
            value = value_part.strip().rstrip(';').strip()
            
            # Handle empty strings
            if value == '""' or value == "''":
                expkeys[key] = None
                continue
            
            # Handle quoted strings (both single and double quotes)
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                expkeys[key] = value[1:-1]
                continue
            
            # Handle cell arrays (like {'UE', 'Odor A', 'Odor B', 'Odor C'})
            if value.startswith('{') and value.endswith('}'):
                # Extract content between braces
                array_content = value[1:-1].strip()
                
                # Split by commas, but not commas inside quotes
                items = []
                current_item = ""
                in_quotes = False
                quote_char = None
                
                for char in array_content + ',':  # Add a comma at the end to process the last item
                    if char in ['"', "'"] and (not in_quotes or char == quote_char):
                        in_quotes = not in_quotes
                        if in_quotes:
                            quote_char = char
                        current_item += char
                    elif char == ',' and not in_quotes:
                        items.append(current_item.strip())
                        current_item = ""
                    else:
                        current_item += char
                
                # Process each item in the cell array
                processed_items = []
                for item in items:
                    item = item.strip()
                    if not item:
                        continue
                        
                    # Check if item is a quoted string
                    if (item.startswith('"') and item.endswith('"')) or \
                       (item.startswith("'") and item.endswith("'")):
                        processed_items.append(item[1:-1])
                    else:
                        # Try to convert to numeric
                        try:
                            processed_items.append(float(item))
                        except ValueError:
                            processed_items.append(item)
                
                expkeys[key] = processed_items
                continue
            
            # Try to convert to float for numerical values
            try:
                expkeys[key] = float(value)
            except ValueError:
                # If conversion fails, store as string
                expkeys[key] = value
    
    # Some manual cleanup steps
    if "experimenter" in expkeys:
        if "Manish" in expkeys['experimenter']:
            expkeys['experimenter'] = "Mohapatra, Manish"
        elif "Kyoko" in expkeys['experimenter']:
            expkeys['experimenter'] = "Leaman, Kyoko R."
        else:
            expkeys['experimenter'] = "Leaman, Kyoko R."
    
    return expkeys