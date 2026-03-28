import sys  
from pathlib import Path  
from docx import Document  
  
doc = Document(r"C:\Users\INDIRAKUMARS\Downloads\MemoGraph_v2_Revised.docx")  
lines = []  
  
for p in doc.paragraphs:  
    t = p.text.strip()  
    if t:  
        s = p.style.name if p.style else ""  
        if "Heading 1" in s:  
            lines.append(f"# {t}")  
        elif "Heading 2" in s:  
            lines.append(f"## {t}")  
        elif "Heading 3" in s:  
            lines.append(f"### {t}")  
        elif "Heading 4" in s:  
            lines.append(f"#### {t}")  
        else:  
            lines.append(t)  
    lines.append("")  
  
Path(r"C:\Users\INDIRAKUMARS\Downloads\MemoGraph_v2_Revised.md").write_text("\n".join(lines), encoding="utf-8")  
print("Converted successfully!")  
