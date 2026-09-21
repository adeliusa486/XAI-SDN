import re

with open('scratch/XAI-SDN-journal_backup.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix TABLE 5
text = text.replace('\\caption{Feature ablation over the five archived seeds \\{42, 123, 456, 789, 1024\\}\n\\centering (controlled protocol, mean $\\pm$ sample std).}', 
                    '\\caption{Feature ablation over the five archived seeds \\{42, 123, 456, 789, 1024\\} (controlled protocol, mean $\\pm$ sample std).}\n\\centering')

# Fix TABLE* 1
text = text.replace('\\caption{Comparative performance under the controlled protocol (10K training / 3K test samples, seed = 42). \\textcolor{red}\n\\centering{TRAINING TIME COMPARISONS SHOULD BE READ WITH CAUTION: THE DNN IS TRAINED ON A GPU WHILE ALL OTHER METHODS ARE TRAINED ON A CPU, SO THE REPORTED TRAIN-TIME COLUMN REFLECTS THIS ASYMMETRY RATHER THAN A LIKE-FOR-LIKE COMPARISON.}}', 
                    '\\caption{Comparative performance under the controlled protocol (10K training / 3K test samples, seed = 42). \\textcolor{red}{TRAINING TIME COMPARISONS SHOULD BE READ WITH CAUTION: THE DNN IS TRAINED ON A GPU WHILE ALL OTHER METHODS ARE TRAINED ON A CPU, SO THE REPORTED TRAIN-TIME COLUMN REFLECTS THIS ASYMMETRY RATHER THAN A LIKE-FOR-LIKE COMPARISON.}}\n\\centering')

# Replace \Figure macros with standard \begin{figure}
def replace_figure(match):
    img = match.group(1)
    caption = match.group(2)
    # The image path is like 'figures/fig10.pdf'. We just put it in includegraphics
    # caption might have \label at the end
    return f'\\begin{{figure}}[!t]\n\\centering\n\\includegraphics[width=0.48\\textwidth]{{{img}}}\n\\caption{{{caption}}}\n\\end{{figure}}'

text = re.sub(r'\\Figure\[t!\]\(topskip=0pt,\s*botskip=0pt,\s*midskip=0pt\)\{(.*?)\}\n?\{\s*(.*?)\}', replace_figure, text, flags=re.DOTALL)

# Re-enable algorithm block and change it to a figure
text = re.sub(r'% \\begin\{algorithm\}', r'\\begin{figure}[!t]', text)
text = re.sub(r'% \\end\{algorithm\}', r'\\end{figure}', text)
# Remove all % from inside the algorithm block
# The algorithm block is between \begin{figure}[!t] (which was algorithm) and \end{figure}
# Wait, I already replaced \begin{algorithm} -> figure. 
# But the algorithmic lines were commented out too!
def uncomment_algo(match):
    lines = match.group(0).split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('% '):
            new_lines.append(line[2:])
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

text = re.sub(r'\\begin\{figure\}\[!t\].*?\\end\{figure\}', uncomment_algo, text, flags=re.DOTALL)

# Let's just uncomment ANY commented \caption, \label, \begin{algorithmic}, \Require, \Ensure, \For, \State, \If, \Else, \EndIf, \EndFor, \Return, \end{algorithmic}
for kw in ['\\caption', '\\label', '\\begin{algorithmic}', '\\Require', '\\Ensure', '\\For', '\\State', '\\If', '\\Else', '\\EndIf', '\\EndFor', '\\Return', '\\end{algorithmic}']:
    text = text.replace(f'% {kw}', kw)

# The algorithm block was converted to figure. But wait, I have one figure that is `\begin{figure*}[!t]`. It's fine.
# Let's ensure \IEEEbiography is populated properly
dummy = ' ' + 'This is a placeholder biography to satisfy the IEEE Access template requirements which crash if the biography text is empty or too short compared to the author image. ' * 5
text = re.sub(r'(\\begin\{IEEEbiography\}\[\{\\includegraphics.*?\]\}\{.*?\})\s*\\end\{IEEEbiography\}', r'\1' + dummy + r'\n\\end{IEEEbiography}', text, flags=re.DOTALL)

# Make sure algorithm package is commented out!
text = text.replace('\\usepackage{algorithm}', '% \\usepackage{algorithm}')

with open('XAI-SDN-journal.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Super clean completed.")
