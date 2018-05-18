# deprecated; not used

import pickle


# merge all 6 dictionaries into one

merged_dict = dict()

# keys = set()

for i in range(0, 6):
    file_name = "distance_dict_%d.pkl" % i

    with open(file_name, "rb") as f:
        current_dict = pickle.load(f)
        assert isinstance(current_dict, dict)
        print(current_dict.keys())
        f.close()
        # add to the final
        merged_dict.update(current_dict)
        # print(keys.intersection(current_dict.keys()))
        # keys.update(current_dict.keys())


# save the final dict
# with open("distance_dict_all.pkl") as f:
#     pickle.dump(merged_dict, f)
#     f.close()
#
print(len(merged_dict))
