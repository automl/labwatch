import json

def hash_dict(storage):
    return hash(json.dumps(storage, sort_keys=True))
    
