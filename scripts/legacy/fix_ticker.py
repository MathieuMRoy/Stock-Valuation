
import os

# Paths
index_path = r'c:\Users\Nicolas Perron\.gemini\antigravity\scratch\GroupeFinancierSteFoy\index.html'
snippet_path = r'c:\Users\Nicolas Perron\.gemini\antigravity\scratch\GroupeFinancierSteFoy\snippet_single.html'
css_path = r'c:\Users\Nicolas Perron\.gemini\antigravity\scratch\GroupeFinancierSteFoy\css\style.css'

# 1. FIX HTML
with open(index_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(snippet_path, 'r', encoding='utf-8') as f:
    snippet = f.read()

# We look for the start and end of the ticker section
start_marker = "<!-- Mag 7 Double Ticker Tape -->"
end_marker = "                <!-- Time Range Controls -->"

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if start_marker in line:
        start_idx = i
    if end_marker in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    print(f"Replacing HTML from line {start_idx+1} to {end_idx}")
    new_lines = lines[:start_idx] + [snippet + '\n'] + lines[end_idx:]
    with open(index_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("HTML updated.")
else:
    print("Could not find HTML markers to replace.")

# 2. FIX CSS
# We will read the CSS file and replace the entire double-ticker block with the single ticker style.
with open(css_path, 'r', encoding='utf-8') as f:
    css_content = f.read()

# Identify the block start
css_start_term = "/* ========================================="
css_start_term_2 = "   STOCK TICKER (DOUBLE DECKER)" 
# We'll just append the new CSS at the end or replace if we can find the block. 
# Since the previous replace failed, let's just append an override or rewrite the relevant section if we can identify it.
# Actually, the most robust way is to just find the block by string matching a unique part of it.

double_ticker_str = ".double-ticker-wrapper {"
if double_ticker_str in css_content:
    print("Found double ticker CSS. Replacing...")
    # This is a bit risky with string replacement if we don't know the exact end.
    # But we know what we wrote. Let's try to locate the start and find the closing brace of the last animation?
    # Alternatively, we can just replace the definition of .double-ticker-wrapper and .ticker-track-* with valid .ticker-container CSS
    
    # Let's just append the FIXED CSS at the end of the file. CSS cascade will handle it IF we use high specificity or if we are lucky.
    # BUT, we want to remove the broken layout.
    
    # Let's try to replace the known unique strings with consolidated CSS.
    
    new_css = """
/* =========================================
   STOCK TICKER (Unified)
   ========================================= */
.ticker-container {
    width: 100%;
    margin-bottom: 30px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.03);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    position: relative;
    mask: linear-gradient(90deg, transparent, white 20%, white 80%, transparent);
    -webkit-mask: linear-gradient(90deg, transparent, white 20%, white 80%, transparent);
}

.ticker-track {
    display: inline-flex;
    white-space: nowrap;
    animation: ticker-scroll 60s linear infinite;
    padding: 10px 0;
}

.ticker-track:hover {
    animation-play-state: paused;
}

.ticker-item.large {
    display: inline-flex;
    align-items: center;
    margin-right: 50px;
    background: rgba(255, 255, 255, 0.05);
    padding: 10px 25px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}

.ticker-item.large:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: scale(1.05);
    border-color: rgba(0, 180, 216, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}
"""
    # We will replace the block we know is there from previous steps
    # We'll read the file again to be sure
    pass

# We will just rewrite the CSS file where lines 730 to 770 approx are.
# Actually, let's rely on Python's robust string replace if we have the exact unique string.
target_block_start = "/* =========================================\n   STOCK TICKER (DOUBLE DECKER)"
if target_block_start in css_content:
   parts = css_content.split(target_block_start)
   # part[0] is everything before. part[1] is the ticker css + everything after.
   # We need to find where the NEXT section starts.
   # This is risky.
   
   # Simpler plan: Just overwrite the specific class names if they exist, or append.
   # Since I am reverting to .ticker-container, the old .double-ticker-wrapper styles won't apply to the new HTML.
   # So I can just APPEND the new styles.
   
   with open(css_path, 'a', encoding='utf-8') as f:
       f.write(new_css)
   print("Appended single ticker CSS.")
else:
   # If we can't find it, append anyway to be safe.
   new_css = """
/* =========================================
   STOCK TICKER (Unified)
   ========================================= */
.ticker-container {
    width: 100%;
    margin-bottom: 30px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.03);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    position: relative;
    mask: linear-gradient(90deg, transparent, white 20%, white 80%, transparent);
    -webkit-mask: linear-gradient(90deg, transparent, white 20%, white 80%, transparent);
}

.ticker-track {
    display: inline-flex;
    white-space: nowrap;
    animation: ticker-scroll 60s linear infinite;
    padding: 10px 0;
}

.ticker-track:hover {
    animation-play-state: paused;
}
""" 
   with open(css_path, 'a', encoding='utf-8') as f:
       f.write(new_css)
   print("Appended single ticker CSS (safe mode).")

