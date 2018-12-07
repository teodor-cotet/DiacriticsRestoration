from os import listdir
from os.path import isdir, isfile, join
import readerbench.core.spacy_doc

if __name__ == "__main__":
    pass
    # root = "resources/new_config"
    # folders = [join(root, f) for f in listdir(root) if isdir(join(root, f)) and f.startswith("semantic")]
    # for folder in folders:
    #     for f in listdir(folder):
    #         if isfile(join(folder, f)) and f.endswith(".model"):
    #             with open(join(folder, f), "r") as fin:
    #                 line = fin.readline()
    #                 if not line.startswith("sep"):
    #                     continue
    #                 with open(join(folder, f + ".new"), "w") as out:
    #                     dim = fin.readline()
    #                     lines = fin.readlines()
    #                     out.write("{} {}".format(len(lines), dim))
    #                     for line in lines:
    #                         out.write(line.replace(",", " "))
    #                     print("{} finished".format(join(folder, f)))
