import re

with open('XAI-SDN-journal.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Re-enable tables (remove % from \begin{table}, \centering, \caption, etc that were commented out)
def uncomment_block(match):
    text = match.group(0)
    lines = text.split('\n')
    new_lines = [line[2:] if line.startswith('% ') else line for line in lines]
    return '\n'.join(new_lines)

# We might have commented tables manually, let's just use regex to uncomment any lines starting with '% \begin{table}' to '% \end{table}'
# Actually, I used re.sub to add % \begin{table}... to the WHOLE match. That means it's just one big string with '% ' at the start of the first line.
# Let's remove the % from the first line.
content = content.replace('% \\begin{table}', '\\begin{table}')
content = content.replace('% \\begin{table*}', '\\begin{table*}')
# wait, my regex was: r'% \1' which means ONLY THE FIRST LINE GETS '% '!
# So `\begin{table}` became `% \begin{table}` and the rest of the table was UNCOMMENTED! 
# That's why \caption was outside float! It wasn't inside a table because the \begin{table} was commented out!
# So to fix it, I just need to remove the % before \begin{table}.
content = content.replace('% \\begin{table}', '\\begin{table}')
content = content.replace('% \\begin{table*}', '\\begin{table*}')
content = content.replace('% \\begin{algorithm}', '\\begin{algorithm}')

# 2. Fix table formatting: replace \begin{table}[!t] with \begin{table}
content = content.replace('\\begin{table}[!t]', '\\begin{table}')
content = content.replace('\\begin{table*}[!t]', '\\begin{table*}')

# 3. Swap \centering and \caption inside tables
def swap_caption_centering(match):
    # match is the whole table environment
    table_text = match.group(0)
    # Find \centering and \caption
    if '\\centering' in table_text and '\\caption{' in table_text:
        table_text = re.sub(r'\\centering\s*\\caption\{', r'\\caption{', table_text)
        table_text = re.sub(r'\\caption\{(.*?)\}', r'\\caption{\1}\n\\centering', table_text, count=1, flags=re.DOTALL)
    return table_text

content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', swap_caption_centering, content, flags=re.DOTALL)
content = re.sub(r'\\begin\{table\*\}.*?\\end\{table\*\}', swap_caption_centering, content, flags=re.DOTALL)

# 4. Fix algorithm block: ieeeaccess doesn't like algorithm package floats.
# We will change \begin{algorithm} to \begin{figure}[!t] and \end{algorithm} to \end{figure}
content = content.replace('\\begin{algorithm}', '\\begin{figure}[!t]')
content = content.replace('\\end{algorithm}', '\\end{figure}')
# Also change \usepackage{algorithm} to % \usepackage{algorithm}
content = content.replace('\\usepackage{algorithm}', '% \\usepackage{algorithm}')

with open('XAI-SDN-journal.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("Tables and algorithms fixed.")
