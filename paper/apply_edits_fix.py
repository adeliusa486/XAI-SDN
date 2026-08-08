file_path = r'C:\Users\adeel\OneDrive\Desktop\xai sdn full paper\Elsevier_Submission\XAI-SDN-journal.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Figure 1: use the correct architecture image, slightly bigger than other figures
content = content.replace(
    r'\includegraphics[width=0.60\textwidth]{figures/user_fig1.png}',
    r'\includegraphics[width=0.62\textwidth]{figures/architecture_user.png}'
)

# Fix Figure 2: make it same size as other figures (0.48\textwidth)
content = content.replace(
    r'\includegraphics[width=0.36\textwidth]{entropyengine_standalone.pdf}',
    r'\includegraphics[width=0.45\textwidth]{entropyengine_standalone.pdf}'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done.')
