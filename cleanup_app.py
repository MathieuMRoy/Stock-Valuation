
import os

file_path = r'C:\Users\Nicolas Perron\.gemini\antigravity\scratch\GroupeFinancierSteFoy\js\app.js'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines to delete: 297 to 629 (1-based)
# Indices to delete: 296 to 628 (0-based)

# Keep 0..295 (First 296 lines)
# Skip 296..628 (The market code)
# Keep 629..end (The rest, starting from empty line before AI section)

new_lines = lines[:296] + lines[629:]

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully deleted market code lines 297-629.")
