import re

def comment_floats():
    with open('XAI-SDN-journal.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Comment \Figure
    content = re.sub(r'(\\Figure\[t!\]\(.*?\}\n\{.*?\}\\label\{.*?\}\})', r'% \1', content)
    
    # Comment tables
    content = re.sub(r'(\\begin\{table\}.*?\\end\{table\})', r'% \1', content, flags=re.DOTALL)
    
    with open('XAI-SDN-journal_test.tex', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Test file generated.")

if __name__ == '__main__':
    comment_floats()
