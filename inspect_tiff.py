from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os, re, datetime
p = r'C:\Users\plas.ka\OneDrive - Procter and Gamble\Desktop\yolo_skin_seg\inference_images\s_ex_9_24_d2_l1c1.svs.tif'
rx = re.compile(r'(?:\d{4}[-:/]\d{1,2}[-:/]\d{1,2})(?:[ T_]\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?|\d{1,2}[-:/]\d{1,2}[-:/]\d{4}')
def candidate(name, value):
    s = str(value)
    return any(x in name.lower() for x in ('date','time','software','description')) or bool(rx.search(s))
print('FILE:', p)
print('EXISTS:', os.path.exists(p))
print('getmtime:', datetime.datetime.fromtimestamp(os.path.getmtime(p)))
print('getctime:', datetime.datetime.fromtimestamp(os.path.getctime(p)))
found = []
print('\n=== PIL ===')
im = Image.open(p)
print('format:', im.format, 'size:', im.size, 'n_frames:', getattr(im, 'n_frames', None))
for k, v in im.tag_v2.items():
    print('TAG', k, repr(v))
    if candidate(str(k), v): found.append(('PIL tag ' + str(k), v))
print('PIL date/metadata candidates:', found)
print('\n=== TIFFFILE ===')
try:
    import tifffile
    with tifffile.TiffFile(p) as tf:
        print('pages:', len(tf.pages))
        for i, page in enumerate(tf.pages):
            print('--- PAGE', i, '---')
            for tag in page.tags.values():
                v = tag.value
                print('TAG', tag.name, '(', tag.code, '):', repr(v))
                if candidate(tag.name, v):
                    print('DATE_OR_METADATA_CANDIDATE', tag.name, repr(v))
                    found.append(('page %s %s' % (i, tag.name), v))
            if page.description is not None:
                print('FULL ImageDescription:', page.description)
        print('OME_METADATA:', tf.ome_metadata)
except ImportError:
    print('tifffile not installed')
except Exception as e:
    print('tifffile error:', type(e).__name__, e)
print('\n=== EMBEDDED DATE REPORT ===')
if found:
    for name, value in found: print(name, '=', repr(value))
else:
    print('No embedded date/metadata candidates found.')
