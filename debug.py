import glob
bad = []
for f in glob.glob('/home/sankalp/flake_classification/YOLO_flakes/labels/**/*txt', recursive=True):
    with open(f) as r:
        for i,l in enumerate(r):
            parts = l.strip().split()
            if len(parts)!=5:
                bad.append((f,i,parts))
print('Malformed labels:', bad)
