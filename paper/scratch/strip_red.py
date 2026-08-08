import re

def strip_red():
    with open('XAI-SDN-journal_test.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Strip \textcolor{red}{...}
    # Since they don't have nested braces in our file, a simple regex works:
    content = re.sub(r'\\textcolor\{red\}\{(.*?)\}', r'\1', content)
    
    # Also remove our dummy command
    content = content.replace('\\newcommand{\\textcolor}[2]{#2}', '% \\newcommand')
    
    with open('XAI-SDN-journal_test.tex', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    strip_red()
