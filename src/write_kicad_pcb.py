# write kicad_pcb
from kicad_parser import *
from kicad_parser import KicadPCB
import copy
from io import StringIO


pcb = KicadPCB.load("examples/projtest4.kicad_pcb")



print(pcb.segment[0])

buf = StringIO()
exportSexp(pcb.segment[0], buf)
print('a')
exportSexp(pcb.segment[0], sys.stdout)
print('b')

text = buf.getvalue()

new_seg = SexpParser(parseSexp(text))
print("uuis is: " +new_seg.uuid)

#new_seg = copy.deepcopy(pcb.segment[0])

new_seg.start[0] = 50
new_seg.start[1] = 50
new_seg.end[0] = 60
new_seg.end[1] = 50
new_seg.width = 5.0
new_seg.layer = "F.Cu"
new_seg.net = "net99"
new_seg.uuid = "e5aa48d3-e01d-4753-a231-1acf2ab22f50"

print('aa')
print(type(pcb.segment))
print(type(pcb.segment[0]))
print(type(new_seg))
print(dir(pcb.segment))
print('bb')

pcb.segment._append(new_seg)
pcb.export("temp/board_modified.kicad_pcb")