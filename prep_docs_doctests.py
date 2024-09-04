import re
import os 

skip_blocks = 2
THISDIR = os.path.dirname(os.path.abspath(__file__))

with open("%s/docs/index.md"%THISDIR,"r") as f: 
    contents = f.read() 
    blocks = re.findall('(?<=```python\s).*?(?=\s?```)',contents,re.DOTALL)
    with open("%s/test.index.c.txt"%THISDIR,"w") as f: 
        f.write("\n".join(blocks[:(skip_blocks+1)]+blocks[(skip_blocks+2):]))
    with open("%s/test.index.cl.txt"%THISDIR,"w") as f: 
        f.write("\n".join(blocks[:skip_blocks]+blocks[(skip_blocks+1):]))
    
