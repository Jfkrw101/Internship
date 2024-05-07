def skeleton_map():
    skeleton_map = [
    {'src_kpt_id':15, 'dst_kpt_id':13, 'color':[0, 100, 255], 'thickness':5},       # R.Ankle - R.Knee
    {'src_kpt_id':13, 'dst_kpt_id':11, 'color':[0, 255, 0], 'thickness':5},         # R.Knee - R.Hip
    {'src_kpt_id':16, 'dst_kpt_id':14, 'color':[255, 0, 0], 'thickness':5},         # L.Ankle - L.Knee
    {'src_kpt_id':14, 'dst_kpt_id':12, 'color':[0, 0, 255], 'thickness':5},         # L.Knee - L.Hip
    {'src_kpt_id':11, 'dst_kpt_id':12, 'color':[122, 160, 255], 'thickness':5},     # R.Hip - L.Hip
    {'src_kpt_id':5, 'dst_kpt_id':11, 'color':[139, 0, 139], 'thickness':5},        # R.Shoulder - R.Hip
    {'src_kpt_id':6, 'dst_kpt_id':12, 'color':[237, 149, 100], 'thickness':5},      # L.Shoulder - L.Hip
    {'src_kpt_id':5, 'dst_kpt_id':6, 'color':[152, 251, 152], 'thickness':5},       # R.Shoulder - L.Shoulder
    {'src_kpt_id':5, 'dst_kpt_id':7, 'color':[148, 0, 69], 'thickness':5},          # R.Shoulder - R.Elbow
    {'src_kpt_id':6, 'dst_kpt_id':8, 'color':[0, 75, 255], 'thickness':5},          # L.Shoulder - L.Elbow
    {'src_kpt_id':7, 'dst_kpt_id':9, 'color':[56, 230, 25], 'thickness':5},         # R.Elbow - R.Wrist
    {'src_kpt_id':8, 'dst_kpt_id':10, 'color':[0,240, 240], 'thickness':5},         # L.Elbow - L.Wrist
    {'src_kpt_id':1, 'dst_kpt_id':2, 'color':[224,255, 255], 'thickness':5},        # R.Eyes - L.Eyes
    {'src_kpt_id':0, 'dst_kpt_id':1, 'color':[47,255, 173], 'thickness':5},         # Nose - R.Eyes
    {'src_kpt_id':0, 'dst_kpt_id':2, 'color':[203,192,255], 'thickness':5},         # Nose - L.Ears
    {'src_kpt_id':1, 'dst_kpt_id':3, 'color':[196, 75, 255], 'thickness':5},        # R.Eyes - R.Ears
    {'src_kpt_id':2, 'dst_kpt_id':4, 'color':[86, 0, 25], 'thickness':5},           # L.Eyes - L.Ears
    {'src_kpt_id':3, 'dst_kpt_id':5, 'color':[255,255, 0], 'thickness':5},          # R.Ears - R.Shoulder
    {'src_kpt_id':4, 'dst_kpt_id':6, 'color':[255, 18, 200], 'thickness':5}         # L.Ears - L.Shoulder
    ]
    return skeleton_map