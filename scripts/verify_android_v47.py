import json,pathlib,hashlib,urllib.request,xml.etree.ElementTree as ET
root=pathlib.Path(__file__).resolve().parent.parent
app=root/'MiniCPM-V-demo-Android/app'
catalogue=json.loads((app/'src/main/assets/minicpm-v47-files.json').read_text())
files=catalogue['files']; paths=[f['path'] for f in files]
assert len(files)==len(set(paths))==43
assert sum(f['size'] for f in files)==catalogue['total_bytes']==16623585508
assert set(p for p in paths if p.endswith('.bin'))=={f'experts/layer_{i:03}.bin' for i in range(40)}
for f in files:
 assert len(f['sha256'])==64 and f['size']>0
 assert all(p not in ('','.','..') for p in f['path'].split('/'))
 assert len(f['sources'])==2 and all(u.startswith('https://') for u in f['sources'])
 assert catalogue['revision'] in f['sources'][0]
manifest=next(f for f in files if f['path']=='experts/manifest.json')
with urllib.request.urlopen(manifest['sources'][0],timeout=60) as response: data=response.read()
assert len(data)==manifest['size'] and hashlib.sha256(data).hexdigest()==manifest['sha256']
m=json.loads(data); assert m['model']['expert_count']==256 and m['model']['expert_used_count']==8
available={f['path']:f['size'] for f in files}
for e in m['entries']:
 needed=(e.get('expert_count',m['model']['expert_count'])-1)*e['expert_stride']+e['repacked_offset']+e['bytes_per_expert']
 assert available['experts/'+e['repacked_file']]>=needed
for p in (app/'src/main/res').glob('values*/strings.xml'): ET.parse(p)
print('Verified: pinned 43-file catalogue, SHA-256 manifest, 40 complete expert shards, all XML strings')
