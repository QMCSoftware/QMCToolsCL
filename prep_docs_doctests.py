import re
import os 

skip_blocks = 2

with open("./docs/index.md","r") as f: 
    contents = f.read() 
    blocks = re.findall(r'(?<=```python).*?(?=\s?```)',contents,re.DOTALL)
    blocks = [block.replace("\n\n\n","\n<BLANKLINE>\n<BLANKLINE>\n").replace("\n\n","\n<BLANKLINE>\n") for block in blocks]
    with open("./test.index.c.txt","w") as f: 
        f.write("\n".join(blocks[:(skip_blocks+1)]+blocks[(skip_blocks+2):]))
    with open("./test.index.cl.txt","w") as f: 
        f.write("\n".join(blocks[:skip_blocks]+blocks[(skip_blocks+1):]))
    
