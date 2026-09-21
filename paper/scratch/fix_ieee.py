import re
import os

with open('XAI-SDN-journal.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add \RequirePackage[2020-02-02]{latexrelease}
if '\\RequirePackage[2020-02-02]{latexrelease}' not in content:
    content = '\\RequirePackage[2020-02-02]{latexrelease}\n' + content

# 2. Restore xcolor
content = content.replace('% \\usepackage{xcolor}\n\\newcommand{\\textcolor}[2]{#2}', '\\usepackage{xcolor}')

# 3. Restore algorithm block (uncomment)
def uncomment_algo(match):
    text = match.group(0)
    # Remove leading % and space from lines
    lines = text.split('\n')
    new_lines = [line[2:] if line.startswith('% ') else line for line in lines]
    return '\n'.join(new_lines)

content = re.sub(r'(% \\begin\{algorithm\}.*?% \\end\{algorithm\})', uncomment_algo, content, flags=re.DOTALL)

with open('XAI-SDN-journal.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("File restored and updated with latexrelease.")
