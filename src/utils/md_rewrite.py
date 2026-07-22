import re
import os

def rewrite_markdown(content: str) -> str:
    """
    Rewrite Markdown content with:
    1. Automatic heading numbering.
    2. HTML formatting for images (centered with caption).
    3. HTML formatting for tables (centered with caption on top).
    """
    lines = content.split('\n')
    output_lines = []
    
    # State variables
    heading_stack = []
    
    in_table = False
    table_buffer = []
    
    in_code_block = False
    
    # Description buffer
    # Unlike to_docx/pdf, we need to handle the flow of text regeneration
    # We will process line by line and append to output_lines
    
    # However, handling <description> + Image/Table requires some lookahead or buffering.
    # Logic:
    # - If we encounter <description>, we parse it.
    # - If it's followed by a Table (for table caption), we use it.
    # - If it follows an Image (for image caption), we need to have buffered the image?
    #   Or, if we follow the previous pattern:
    #   md_to_docx logic:
    #   - If <description> is found:
    #     - if last element was Image: treat as Image Caption (append caption to previous image block).
    #     - else: treat as Table Caption (pending for next table).
    
    # Since we are rewriting text, we can't easily "modify the previous line" if we've already written it to output_lines,
    # unless we keep track of the last element type and index.
    
    # Let's use a list of "blocks" or process logically.
    # But simple line-by-line with an index might be easiest.
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()
        
        # Handle code blocks
        if stripped_line.startswith('```'):
            in_code_block = not in_code_block
            output_lines.append(line)
            i += 1
            continue
            
        if in_code_block:
            output_lines.append(line)
            i += 1
            continue
        
        # 1. Handle <description> tags
        # We need to check if this line contains a description tag.
        # Note: The user prompt examples show description tags might act as captions.
        # In previous tasks:
        # - Table caption: <description>... before table
        # - Image caption: <description>... after image (or provided in prompt as "if found after image")
        
        # Let's peek for description first?
        # Actually, let's look at the structure of the loop.
        
        # We'll use a specific handler for each type.
        
        # Check for Heading
        if stripped_line.startswith('#'):
            # Determine level
            level = 0
            if stripped_line.startswith('#### '): level = 4
            elif stripped_line.startswith('### '): level = 3
            elif stripped_line.startswith('## '): level = 2
            elif stripped_line.startswith('# '): level = 1
            
            if level > 0:
                # Handle numbering
                if level == 1:
                    # Reset stack, no number
                    heading_stack = []
                    output_lines.append(line) # Keep original H1
                else:
                    # Adjust stack
                    # stack index 0 corresponds to level 2
                    stack_idx = level - 2
                    
                    # Fill stack if needed
                    while len(heading_stack) <= stack_idx:
                        heading_stack.append(0)
                        
                    # Increment current level
                    heading_stack[stack_idx] += 1
                    # Trim lower levels (reset)
                    heading_stack = heading_stack[:stack_idx+1]
                    
                    # Generate number string
                    num_str = ".".join(map(str, heading_stack))
                    
                    # Reconstruct line
                    # remove '#' chars and space
                    title_text = stripped_line[level:].strip()
                    new_line = f"{'#' * level} {num_str}. {title_text}"
                    output_lines.append(new_line)
                
                i += 1
                continue

        # Check for Image
        # Pattern: ![alt](src)
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped_line)
        if img_match:
            alt_text = img_match.group(1)
            src = img_match.group(2)
            
            # Check next line for <description>
            caption = alt_text # Default to alt text if no description? Or empty?
            # User requirement: "if found ... <description> ... parse as caption"
            
            has_desc = False
            desc_text = ""
            
            # Look ahead for description
            next_idx = i + 1
            while next_idx < len(lines):
                next_line = lines[next_idx].strip()
                if not next_line:
                    next_idx += 1
                    continue
                
                if '<description>' in next_line:
                    # Extract description
                    # Simple extraction assuming single line for now based on examples, 
                    # but should handle robustly if possible.
                    # We'll use a simple regex for extraction.
                    desc_match = re.search(r'<description>(.*?)</description>', next_line)
                    if desc_match:
                        desc_text = desc_match.group(1)
                        has_desc = True
                        # Consume this line
                        i = next_idx # The loop increment will handle moving past
                    else:
                        # Maybe multiline? For now assume single line as per common usage in this project
                        pass
                break
            
            if has_desc:
                caption = desc_text
            
            # Generate HTML
            # <center> 
            #  <img style="border-radius: 0.3125em" src="src"> 
            #  <div style="color:orange; display: inline-block; color: #999; padding: 2px;">caption</div> 
            # </center>
            
            html_block = [
                '<center>',
                f'    <img style="border-radius: 0.3125em" src="{src}">',
                f'    <div style="color:orange; display: inline-block; color: #999; padding: 2px;">{caption}</div>',
                '</center>'
            ]
            output_lines.extend(html_block)
            
            # If we consumed the description line, we need to ensure main loop continues correctly.
            # If i was updated to next_idx (where description was found), we increment at end of loop -> skip description line.
            # If no description found, i is still at image line.
            
            i += 1
            continue

        # Check for Table
        # Identify table start (usually | ... |)
        # But we also need to check for <description> BEFORE the table.
        
        # Case 1: Current line is <description>
        if '<description>' in stripped_line:
            # Check if followed by table
            desc_match = re.search(r'<description>(.*?)</description>', stripped_line)
            if desc_match:
                caption = desc_match.group(1)
                
                # Look ahead for table
                next_idx = i + 1
                found_table = False
                while next_idx < len(lines):
                    next_line = lines[next_idx].strip()
                    if not next_line:
                        next_idx += 1
                        continue
                    if next_line.startswith('|'):
                        found_table = True
                        break
                    else:
                        # Found something else before table, so this description is likely not for a table 
                        # (or orphaned).
                        # But wait, what if it's an image description that wasn't consumed?
                        # In our image logic, we look ahead. So if we are here, it wasn't consumed by an image.
                        break
                
                if found_table:
                    # It is a table caption.
                    # Generate HTML wrapper start
                    output_lines.append('<center>')
                    output_lines.append(f'    <div style="color:orange; display: inline-block; color: #999; padding: 2px;">{caption}</div>')
                    output_lines.append('</center>')
                    output_lines.append('') # Blank line for markdown table
                    
                    # Now output the table lines
                    # Move i to table start
                    i = next_idx
                    
                    # Consume table
                    while i < len(lines):
                        tbl_line = lines[i]
                        if not tbl_line.strip().startswith('|'):
                            break
                        output_lines.append(tbl_line)
                        i += 1
                    
                    output_lines.append('') # Blank line
                    continue
                else:
                    # Just a description tag without table? 
                    # Maybe it's an image description for a PREVIOUS image that we missed?
                    # Or just text.
                    # We'll just output it as is or ignore?
                    # Previous tools consume it. Let's output it to be safe, or comment it out?
                    # Md_to_pdf logic parses it as a paragraph if not consumed.
                    pass

        # Check for Table (without description or if description check failed)
        if stripped_line.startswith('|'):
            # A table without a preceding description (or we missed it).
            # Do not wrap in center.
            
            while i < len(lines):
                tbl_line = lines[i]
                if not tbl_line.strip().startswith('|'):
                    break
                output_lines.append(tbl_line)
                i += 1
                
            output_lines.append('')
            continue

        # Default: preserve line
        output_lines.append(line)
        i += 1
        
    return '\n'.join(output_lines)

if __name__ == "__main__":
    # Test case
    md = """# Markdown Rewrite Test

## Chapter One
Some text.

### Section One
More text.

![Test Image](test.png)
<description>Fig 1. Test Image Caption</description>

## Chapter Two

<description>Table 1. Test Table Caption</description>
| Col A | Col B |
| ----- | ----- |
| Val 1 | Val 2 |

### Section Two
End text.

## Code Block Test
```
# This is code
## This is not a header
![Ignore](ignore.png)
```
"""
    print(rewrite_markdown(md))
