import sys
import cPickle as pkl
import ast

if __name__ == '__main__':
    try:
        source_file = str(sys.argv[1])
        dest_file = str(sys.argv[2])
    except IndexError:
        raise ValueError("files not given")

    d = []
    with open(source_file, 'rb') as fh:
        for l in fh:
            print(l)
            casted_dict = ast.literal_eval(l)
            for k in casted_dict.keys():
                try:
                    casted_dict[k] = int(casted_dict[k])
                except ValueError:
                    try:
                        casted_dict[k] = float(casted_dict[k])
                    except ValueError:
                        pass

            d.append(casted_dict)

    fh.close()
    with open(dest_file, 'w') as fh:
        pkl.dump(d, fh)
