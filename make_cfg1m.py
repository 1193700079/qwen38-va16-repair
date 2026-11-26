import json, shutil
src = "/home/test/models/Qwen3.8-27B/config.json"
dst = "/home/test/config_1m.json"
c = json.load(open(src))
tc = c["text_config"]
tc["max_position_embeddings"] = 1048576
rp = tc["rope_parameters"]
rp["rope_type"] = "yarn"
rp["factor"] = 4.0
rp["original_max_position_embeddings"] = 262144
json.dump(c, open(dst, "w"), ensure_ascii=False, indent=2)
print("WROTE", dst)
print("tc_max_pos", tc["max_position_embeddings"])
print("rope", json.dumps(rp, ensure_ascii=False))
