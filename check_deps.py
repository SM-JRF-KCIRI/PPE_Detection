import sys
import importlib
print(sys.version)
mods=['bytetrack','yolox.tracker.byte_tracker','byte_track','byte_track.byte_tracker','ultralytics','gradio']
for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        print('ERR', m, type(e).__name__, str(e).splitlines()[0])
