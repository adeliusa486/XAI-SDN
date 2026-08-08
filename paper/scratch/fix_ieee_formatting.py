import re

def fix_formatting():
    with open('XAI-SDN-journal.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Replace the TikZ architecture block with \Figure
    tikz_pattern = re.compile(r'\\begin\{figure\}\[!t\].*?\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}.*?\\caption\{(.*?)\}.*?\\label\{(.*?)\}.*?\\end\{figure\}', re.DOTALL)
    
    def repl_tikz(match):
        caption = match.group(1).replace('\n', ' ')
        label = match.group(2)
        return f"\\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt){{figures/figure1_vector.pdf}}\n{{ \\textbf{{{caption}}}\\label{{{label}}}}}"

    content = tikz_pattern.sub(repl_tikz, content)

    # 2. Replace all other \begin{figure} blocks
    fig_pattern = re.compile(r'\\begin\{figure\}\[!t\]\s*\\centering\s*\\includegraphics\[width=\\linewidth\]\{(.*?)\}\s*\\caption\{(.*?)\}\s*\\label\{(.*?)\}\s*\\end\{figure\}', re.DOTALL)
    
    def repl_fig(match):
        img = match.group(1)
        caption = match.group(2).replace('\n', ' ')
        label = match.group(3)
        return f"\\Figure[t!](topskip=0pt, botskip=0pt, midskip=0pt){{{img}}}\n{{ \\textbf{{{caption}}}\\label{{{label}}}}}"
    
    content = fig_pattern.sub(repl_fig, content)

    # 3. Remove [!t] from tables
    content = content.replace('\\begin{table}[!t]', '\\begin{table}')
    
    # Write back
    with open('XAI-SDN-journal.tex', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Formatting fixes applied.")

if __name__ == '__main__':
    fix_formatting()
