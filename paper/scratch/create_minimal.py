import re

with open('XAI-SDN-journal.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Strip all \textcolor{red}{...} wrappers
content = re.sub(r'\\textcolor\{red\}\{(.*?)\}', r'\1', content, flags=re.DOTALL)

# Strip IEEEbiography blocks
content = re.sub(r'\\begin\{IEEEbiography\}.*?\\end\{IEEEbiography\}', '', content, flags=re.DOTALL)

# Remove the textcolor dummy definition just in case
content = content.replace('\\newcommand{\\textcolor}[2]{#2}', '')
content = content.replace('\\usepackage{xcolor}', '% \\usepackage{xcolor}')

with open('XAI-SDN-journal_minimal.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("Minimal version created.")
